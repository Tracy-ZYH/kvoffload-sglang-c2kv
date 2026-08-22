#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import requests
from transformers import AutoTokenizer


def post_json(base_url: str, path: str, payload: dict) -> dict:
    response = requests.post(
        f"{base_url}{path}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=120,
    )
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"{path} failed: HTTP {response.status_code}: {body}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prefix-text", default="Alice lives in Beijing.")
    parser.add_argument("--suffix-text", default=" Her favorite number is 7319.")
    parser.add_argument("--checkpoint-id", default="recovery_smoke_c0")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prefix_ids = tokenizer.encode(args.prefix_text, add_special_tokens=True)
    suffix_ids = tokenizer.encode(args.suffix_text, add_special_tokens=False)
    full_ids = prefix_ids + suffix_ids

    print(f"Full prefix tokens: {len(prefix_ids)}")
    prefill = post_json(
        args.base_url,
        "/generate",
        {
            "input_ids": prefix_ids,
            "sampling_params": {"max_new_tokens": 0},
        },
    )
    print("Prefill:", json.dumps(prefill.get("meta_info", prefill), indent=2))

    create = post_json(
        args.base_url,
        "/recovery_checkpoint/create",
        {
            "checkpoint_id": args.checkpoint_id,
            "input_ids": prefix_ids,
            "tier": "host",
            "evict_device_after": True,
            "sync": True,
        },
    )
    print("Create:", json.dumps(create, indent=2))

    status = post_json(
        args.base_url,
        "/recovery_checkpoint/status",
        {"checkpoint_id": args.checkpoint_id},
    )
    print("Status after create:", json.dumps(status, indent=2))

    restore_started = time.perf_counter()
    restore = post_json(
        args.base_url,
        "/recovery_checkpoint/restore",
        {
            "checkpoint_id": args.checkpoint_id,
            "sync": True,
            "pin_device": True,
        },
    )
    print(f"Restore wall latency ms: {(time.perf_counter() - restore_started) * 1000:.3f}")
    print("Restore:", json.dumps(restore, indent=2))

    reuse = post_json(
        args.base_url,
        "/generate",
        {
            "input_ids": full_ids,
            "sampling_params": {"max_new_tokens": 0},
        },
    )
    print("Reuse prefill:", json.dumps(reuse.get("meta_info", reuse), indent=2))

    release = post_json(
        args.base_url,
        "/recovery_checkpoint/release",
        {"checkpoint_id": args.checkpoint_id},
    )
    print("Release:", json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
