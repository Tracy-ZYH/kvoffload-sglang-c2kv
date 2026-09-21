from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace


HTTP_SERVER = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/entrypoints/http_server.py"
)


def _load_functions(*names):
    tree = ast.parse(HTTP_SERVER.read_text(encoding="utf-8"), filename=str(HTTP_SERVER))
    wanted = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            wanted.append(node)
    assert {node.name for node in wanted} == set(names)
    namespace = {
        "C2KVTokenizeRequest": object,
        "C2KVTokenizeResponse": lambda **fields: SimpleNamespace(
            success=True, error=None, **fields
        ),
    }
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(HTTP_SERVER), "exec"), namespace)
    return namespace


def test_tokenize_endpoint_counts_template_tokens_without_model_work():
    functions = _load_functions("_c2kv_template_ids", "v1_c2kv_tokenize")
    tokenizer = SimpleNamespace(
        bos_token="<bos>",
        apply_chat_template=lambda messages, **_kwargs: (
            "<bos><user>" + messages[0]["content"] + "</user>"
        ),
        encode=lambda text, add_special_tokens=False: list(range(len(text))),
    )

    class Manager:
        def __init__(self):
            self.tokenizer = tokenizer

        async def c2kv_extract(self, **_kwargs):
            raise AssertionError("token-count-only endpoint scheduled model work")

    functions["_global_state"] = SimpleNamespace(tokenizer_manager=Manager())
    functions["_c2kv_flat_tools"] = lambda tools: tools
    request = SimpleNamespace(
        role="user",
        text="x" * 35_939,
        tools=None,
        chat_template_kwargs={"enable_thinking": False},
    )
    response = asyncio.run(functions["v1_c2kv_tokenize"](request))

    assert response.success is True
    assert response.token_count == len("<user>") + 35_939 + len("</user>")
