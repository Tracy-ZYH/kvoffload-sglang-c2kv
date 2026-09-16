"""CPU-only lifecycle regression tests. Never import a model or contact HTTP."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import types
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "python/sglang/srt/mem_cache"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ledger = load("persistent_ledger_test", CACHE / "history_kv_lifecycle.py")
eviction = load("physical_evictor_test", CACHE / "history_kv_eviction.py")


def method(path, cls, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("method_name", ["streamingllm", "h2o", "snapkv_persistent", "pyramidkv"])
@pytest.mark.parametrize("layout", ["token", "ascend_page", "ascend_fia"])
def test_two_turns_reuse_only_resident_kv_and_new_tokens(method_name, layout):
    # Use nonzero physical allocator slots, independently of logical position.
    row = torch.zeros((1, 64), dtype=torch.int64)
    row[0, :8] = torch.arange(4, 12)
    keys = torch.zeros((64, 1, 1))
    values = torch.zeros_like(keys)
    keys[4:12, 0, 0] = torch.arange(8)
    values[4:12, 0, 0] = torch.arange(8) + 100
    shape = (16,4,1,1) if layout == "ascend_page" else (64,1,1,1)
    cache = SimpleNamespace(start_layer=0, layer_num=1,
        _get_key_buffer=lambda _: keys if layout == "token" else keys.view(shape),
        _get_value_buffer=lambda _: values if layout == "token" else values.view(shape))
    freed = []
    allocator = SimpleNamespace(page_size=4, get_kvcache=lambda: cache,
                                free=lambda x: freed.extend(x.tolist()), available_size=lambda: 64)
    req = SimpleNamespace(req_pool_idx=0, kv_committed_len=8, kv_allocated_len=8,
                          already_computed=8, c2kv_position_correction=0)
    evictor = eviction.PhysicalHistoryKVEvictor(SimpleNamespace(req_to_token=row), allocator)
    result = evictor.evict(req, method=method_name, history_start=2, history_end=8,
                          target_tokens=3, selected_history_indices=[0, 3, 5])
    assert result.success
    assert req.kv_committed_len == 5 and result.new_physical_kv_slots == 8
    assert result.next_rope_position_before == result.next_rope_position_after == 8
    retained = ledger.compact_positions(range(8), 2, 8, [0, 3, 5])
    assert retained == [0, 1, 2, 5, 7]
    assert keys[row[0, :5], 0, 0].tolist() == retained
    assert values[row[0, :5], 0, 0].tolist() == [p+100 for p in retained]
    # Turn 2 uses the *same physical prefix*. Only canonical new positions 8–11
    # are appended; logical old history is not materialized again.
    next_positions = ledger.append_resident_positions(retained, 8, 12)
    row[0, 5:9] = torch.arange(9, 13)
    keys[9:13, 0, 0] = torch.arange(8, 12)
    values[9:13, 0, 0] = torch.arange(8, 12) + 100
    req.kv_committed_len = req.kv_allocated_len = 9
    assert keys[row[0, :9], 0, 0].tolist() == next_positions
    hs, he = ledger.physical_history_range(next_positions, 2, 10)
    result = evictor.evict(req, method=method_name, history_start=hs, history_end=he,
                          target_tokens=3, selected_history_indices=[0, 3, 4])
    assert result.success and result.next_rope_position_after == 12
    final = ledger.compact_positions(next_positions, hs, he, [0, 3, 4])
    assert not {3, 4, 6}.intersection(final)
    assert keys[row[0, :req.kv_committed_len], 0, 0].tolist() == final
    assert values[row[0, :req.kv_committed_len], 0, 0].tolist() == [p+100 for p in final]


def test_within_turn_boundary_may_precede_cached_current_prefix():
    positions = ledger.append_resident_positions([0, 1, 4, 5, 6], 7, 10)
    assert ledger.physical_history_range(positions, 2, 5) == (2, 3)
    assert positions == [0, 1, 4, 5, 6, 7, 8, 9]


def test_ledger_rejects_duplicate_and_resurrected_old_positions():
    for bad in ([0, 0], [0, 8], [2, 1]):
        with pytest.raises(ValueError):
            ledger.append_resident_positions(bad, 8, 12)
    with pytest.raises(ValueError):
        ledger.compact_positions([0, 1, 5], 1, 3, [0, 0])


def test_decode_cleanup_never_frees_prompt_pages_at_nonzero_allocator_offset():
    discard = method(CACHE / "session_aware_cache.py", "SessionAwareCache",
                     "_discard_persistent_decode_suffix", {"torch": torch, "Req": SimpleNamespace})
    row = torch.arange(40, 60).reshape(1, 20)
    freed = []
    self = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=row), page_size=4,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda x: freed.extend(x.tolist())))
    req = SimpleNamespace(origin_input_ids=list(range(5)), kv_committed_len=8, kv_allocated_len=10,
        req_pool_idx=0, persistent_decode_cache_locs=[torch.tensor(44), torch.tensor(49), torch.tensor(50)],
        kv_memory_report={})
    discard(self, req)
    assert freed == [48]  # page 12 only; prompt pages 10 and 11 must survive
    assert row[0, :5].tolist() == [40, 41, 42, 43, 44]
    assert row[0, 5:10].tolist() == [0]*5
    assert req.kv_committed_len == req.kv_allocated_len == 5


@pytest.mark.parametrize("method_name", ["streamingllm", "h2o", "snapkv_persistent", "pyramidkv"])
@pytest.mark.parametrize("streaming", [False, True])
def test_multiround_finish_transfers_session_ownership_without_radix_insert(method_name, streaming):
    # Execute the real release function without importing SGLang/device code.
    path = CACHE / "common.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "release_kv_cache")
    class HybridPool:
        pass
    namespace = {"Req": SimpleNamespace, "BasePrefixCache": object,
                 "HybridReqToTokenPool": HybridPool}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    release = namespace["release_kv_cache"]
    owns = method(CACHE / "session_aware_cache.py", "SessionAwareCache",
                  "owns_finished_request", {"Req": SimpleNamespace,
                  "_is_streaming": lambda r: r.session is not None and r.session.streaming})
    # AST extraction leaves @staticmethod: resolve its function explicitly.
    owns = owns.__func__
    freed, pool_freed, saved = [], [], []
    pool = SimpleNamespace(req_to_token=torch.arange(40, 48).reshape(1, 8),
                           free=lambda r: pool_freed.append(r.req_pool_idx))
    def finish(req, is_insert):
        assert is_insert is False
        saved.append((req.req_pool_idx, req.kv_allocated_len))
        req.req_pool_idx = None  # SessionSlot.save_from_req ownership transfer
    cache = SimpleNamespace(req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda x: freed.extend(x.tolist())),
        owns_finished_request=owns, cache_finished_req=finish,
        dec_lock_ref=lambda _: None)
    req = SimpleNamespace(req_pool_idx=0, c2kv_rounds=[object()],
        c2kv_tree_cache_prefix_len=0, kv_allocated_len=8, last_node=None,
        session=SimpleNamespace(streaming=streaming), history_kv_eviction={"method": method_name})
    release(req, cache, is_insert=False)
    if streaming:
        assert saved == [(0, 8)] and req.req_pool_idx is None
        assert freed == [] and pool_freed == []
        # Session close, not normal request completion, releases these slots.
        cache.token_to_kv_pool_allocator.free(pool.req_to_token[0, :8])
        assert freed == list(range(40, 48))
    else:
        assert saved == [] and pool_freed == [0]
        assert freed == list(range(40, 48))
        assert req.kv_committed_freed and req.kv_overallocated_freed


def test_serving_delta_prefix_mismatch_fails_without_full_prefill_fallback():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    prepare = method(path, "OpenAIServingChat", "_prepare_persistent_history_delta",
                     {"ChatCompletionRequest": object, "List": list, "Optional": __import__('typing').Optional})
    self = SimpleNamespace(_is_persistent_history_request=lambda _: True,
                           _persistent_history_sessions={"s": [0, 1, 2, 3]})
    req = SimpleNamespace(stream=False, session_params={"id": "s"},
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": True, "session_id": "s"},
                           "history_kv_eviction": {"history_start": 1, "history_end": 2}})
    delta, sid, canonical = prepare(self, req, [0, 1, 2, 3, 4, 5])
    assert delta == [4, 5] and sid == "s" and canonical == [0, 1, 2, 3, 4, 5]
    assert req.c2kv_kv_memory_hint['history_kv_eviction']['persistent_delta_history_tokens'] == 0
    with pytest.raises(ValueError, match="PREFIX_MISMATCH"):
        prepare(self, req, [9, 1, 2, 3, 4, 5])
    req.session_params['id'] = 'other'
    with pytest.raises(ValueError, match="ID_MISMATCH"):
        prepare(self, req, [0, 1, 2, 3, 4, 5])


def test_session_match_restores_prefix_and_builds_only_new_history_round(monkeypatch):
    # Inject only the two tiny modules imported by the method under test.
    # No SGLang/model imports, no endpoint, no allocator device initialization.
    rounds_module = types.ModuleType('sglang.srt.managers.schedule_batch')
    class Round:
        def __init__(self, tokens, segments, post_history_kv_eviction=False):
            self.tokens = tokens; self.post_history_kv_eviction = post_history_kv_eviction
    rounds_module.C2KVPrefillRound = Round
    monkeypatch.setitem(sys.modules, rounds_module.__name__, rounds_module)
    monkeypatch.setitem(sys.modules, 'sglang.srt.mem_cache.history_kv_lifecycle', ledger)
    match = method(CACHE/'session_aware_cache.py', 'SessionAwareCache', 'match_prefix', {
        'MatchPrefixParams': object, 'MatchResult': lambda **kw: SimpleNamespace(**kw),
        'torch': torch, '_is_streaming': lambda _: True})
    row = torch.arange(4, 36).reshape(1, 32)
    prior = [0, 1, 2, 5, 7]
    req = SimpleNamespace(session=SimpleNamespace(session_id='s'), kv_memory_report={},
        history_kv_eviction={'persistent_continuation_pending': True,
                            'persistent_protected_prefix_tokens': 2,
                            'persistent_canonical_history_end': 10,
                            'persistent_delta_history_tokens': 2},
        c2kv_kv_memory_hint={'persistent_session_logical_prefix_tokens': 8,
                            'persistent_session_canonical_prompt_tokens': 12},
        origin_input_ids=prior+[8,9,10,11])
    def restore(req):
        req.req_pool_idx=0; req.kv_committed_len=5; req.c2kv_position_correction=3
    slot=SimpleNamespace(req_pool_idx=0, history_kv_resident_positions=prior,
                         restore_to_req=restore, cache_protected_len=0, virtual_node=object())
    self=SimpleNamespace(slots={'s':slot}, req_to_token_pool=SimpleNamespace(req_to_token=row))
    result=match(self,SimpleNamespace(req=req,key=SimpleNamespace(token_ids=req.origin_input_ids)))
    assert result.device_indices.tolist()==[4,5,6,7,8]
    assert req.history_kv_resident_positions==prior+[8,9,10,11]
    assert req.history_kv_eviction['history_start']==2
    assert req.history_kv_eviction['history_end']==7
    assert req.c2kv_rounds[0].tokens==prior+[8,9]
    assert req.c2kv_rounds[1].tokens==[10,11]
    assert not {3,4,6}.intersection(req.history_kv_resident_positions)


@pytest.mark.parametrize("method_name", ["h2o", "snapkv_persistent", "pyramidkv"])
@pytest.mark.parametrize("layout", ["token", "ascend_page", "ascend_fia"])
def test_attention_eviction_scores_cached_resident_keys_not_full_history(method_name, layout):
    collect=method(ROOT/'python/sglang/srt/models/qwen3.py', 'Qwen3Attention',
        '_collect_history_kv_eviction_scores', {'torch':torch, 'ForwardBatch':SimpleNamespace})
    keys=torch.full((32,1,1),1000.)  # nonresident pages must not enter scoring
    keys[[4,8,9,13],0,0]=torch.tensor([0.,1.,4.,7.])
    if layout == "ascend_page":
        keys = keys.reshape(8,4,1,1)
    elif layout == "ascend_fia":
        keys = keys.reshape(32,1,1,1)
    config={'method':method_name,'history_start':1,'history_end':4,
            'resident_logical_positions':[0,1,4,7,8]}
    fb=SimpleNamespace(c2kv_history_kv_eviction_configs=[config],
        forward_mode=SimpleNamespace(is_extend_or_draft_extend_or_mixed=lambda:True),
        extend_seq_lens_cpu=[1],extend_prefix_lens_cpu=[4],req_pool_indices=torch.tensor([0]),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[4,8,9,13,14]])),
        token_to_kv_pool=SimpleNamespace(_get_key_buffer=lambda _:keys))
    self=SimpleNamespace(num_heads=1,num_kv_heads=1,head_dim=1,scaling=1.,attn=SimpleNamespace(layer_id=0))
    collect(self,torch.ones(1,1),torch.tensor([[8.]]),torch.tensor([8]),fb)
    scores=fb.c2kv_history_kv_selection_scores[0]['layers'][0]
    assert scores.numel()==3 and scores.argmax().item()==2
    assert torch.allclose(scores,torch.softmax(torch.tensor([0.,1.,4.,7.]),0)[1:])


def test_missing_persistent_slot_never_falls_back_to_reprefill():
    match = method(CACHE/'session_aware_cache.py', 'SessionAwareCache', 'match_prefix', {
        'MatchPrefixParams': object, 'MatchResult': object, '_is_streaming': lambda _: True})
    req = SimpleNamespace(session=SimpleNamespace(session_id='lost'),
                          history_kv_eviction={'persistent_continuation':True},c2kv_kv_memory_hint={})
    self = SimpleNamespace(slots={})
    with pytest.raises(RuntimeError,match='RESIDENT_CACHE_MISSING'):
        match(self,SimpleNamespace(req=req))


def test_h2o_accumulates_scores_only_for_resident_positions_across_requests():
    select=method(ROOT/'python/sglang/srt/managers/scheduler.py', 'Scheduler',
                  '_select_history_kv_eviction_indices', {'torch':torch,'Optional':Optional})
    config={'method':'h2o','persistent_session':True,'history_start':1,'history_end':4,'target_tokens':1}
    req=SimpleNamespace(history_kv_eviction=config,history_kv_resident_positions=[0,1,4,7,8],
        history_kv_selection_scores={'layers':[torch.tensor([.1,.2,.7])]},
        history_kv_score_state={0:{3:1000.,4:10.}})
    assert select(None,req,config)==[1]  # canonical position 4 retains old score
    assert 3 not in req.history_kv_score_state[0]  # evicted position is not a candidate
    req.history_kv_resident_positions=[0,4,8,9]
    config['history_end']=3
    req.history_kv_selection_scores={'layers':[torch.tensor([.2,.9])]}
    assert select(None,req,config)==[0]  # old resident 4, not resurrected old 7
    assert set(req.history_kv_score_state[0])=={4,8}
