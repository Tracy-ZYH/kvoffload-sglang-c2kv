"""Reference attention must bypass both captured execution paths."""
import ast
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/model_executor/model_runner.py"
QWEN_SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/models/qwen3.py"


def test_reference_eager_gate_and_both_call_sites():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_requires_reference_attention_eager")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), namespace)
    gate = namespace[helper.name]
    assert not gate(SimpleNamespace())
    assert not gate(SimpleNamespace(history_kv_reference_configs=[None]))
    assert gate(SimpleNamespace(history_kv_reference_configs=[{}]))
    assert gate(SimpleNamespace(history_kv_reference_states=[None, object()]))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "ModelRunner")
    for name in ("forward_extend", "_forward_raw"):
        method = next(node for node in runner.body if isinstance(node, ast.FunctionDef)
                      and node.name == name)
        assignment = next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "can_run_graph"
                                  for target in node.targets))
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == helper.name for node in ast.walk(assignment.value))


def test_reference_runtime_forces_explicit_qkv_before_state_exists():
    tree = ast.parse(QWEN_SOURCE.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_requires_reference_runtime_qkv")
    namespace = {"ForwardBatch": object}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(QWEN_SOURCE), "exec"), namespace)
    gate = namespace[helper.name]
    assert not gate(SimpleNamespace())
    assert not gate(SimpleNamespace(history_kv_reference_configs=[None]))
    assert not gate(SimpleNamespace(history_kv_reference_states=[None]))
    assert not gate(SimpleNamespace(
        history_kv_reference_configs=[None],
        history_kv_reference_states=[None],
    ))
    assert gate(SimpleNamespace(history_kv_reference_configs=[{"method": "commitkv"}]))
    assert gate(SimpleNamespace(
        history_kv_reference_configs=[{"method": "agentkv"}],
        history_kv_reference_states=[None],
    ))
    assert gate(SimpleNamespace(
        history_kv_reference_configs=[None],
        history_kv_reference_states=[object()],
    ))
    assert gate(SimpleNamespace(history_kv_reference_states=[None, object()]))

    attention = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "Qwen3Attention")
    forward = next(node for node in attention.body if isinstance(node, ast.FunctionDef)
                   and node.name == "forward")
    assignments = [
        node for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "use_reference_runtime"
                for target in node.targets)
    ]
    assert len(assignments) == 1
    calls = [node for node in ast.walk(assignments[0].value) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Name) and call.func.id == helper.name
               for call in calls)

    native_branch = next(
        node for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and any(isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "forward_prepare_native"
                for statement in node.body)
    )
    assert any(isinstance(node, ast.Name) and node.id == "use_reference_runtime"
               for node in ast.walk(native_branch.test))
