from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.mem_cache.radix_cache import TreeNode


class RecoveryCheckpointState(str, Enum):
    CREATING = "creating"
    HOST_READY = "host_ready"
    DEVICE_READY = "device_ready"
    RELEASED = "released"
    INVALID = "invalid"


@dataclass
class RecoveryCheckpoint:
    checkpoint_id: str

    token_count: int
    token_hash: str
    extra_key: Optional[str]

    last_node_id: int
    node_path_ids: List[int]

    session_id: Optional[str]
    segment_id: Optional[int]
    global_step: Optional[int]

    parent_checkpoint_id: Optional[str]

    tier: str
    state: RecoveryCheckpointState

    created_at: float

    host_tokens: int = 0
    device_tokens: int = 0

    backup_tokens: int = 0
    restore_tokens: int = 0

    backup_latency_ms: float = 0.0
    restore_latency_ms: float = 0.0

    requested_host_only_tokens: int = 0
    actual_host_only_tokens: int = 0
    shared_or_locked_device_tokens: int = 0

    restore_success_count: int = 0
    restore_failure_count: int = 0
    create_count: int = 1
    release_count: int = 0

    device_pinned: bool = False
    host_pinned: bool = False
    fallback_reason: str = ""


class RecoveryCheckpointManager:
    """Manage offloaded Full-KV recovery checkpoints.

    The manager owns only checkpoint metadata and radix TreeNode references.
    KV movement and memory ownership stay inside HiRadixCache.
    """

    def __init__(self, tree_cache):
        self.tree_cache = tree_cache
        self.checkpoints: Dict[str, RecoveryCheckpoint] = {}
        self._last_nodes: Dict[str, TreeNode] = {}
        self.generation = 0
        self.metrics: Dict[str, float] = {
            "recovery_checkpoint_create_count": 0,
            "recovery_checkpoint_release_count": 0,
            "recovery_checkpoint_backup_tokens": 0,
            "recovery_checkpoint_backup_latency_ms": 0.0,
            "recovery_checkpoint_restore_tokens": 0,
            "recovery_checkpoint_restore_latency_ms": 0.0,
            "recovery_checkpoint_restore_success": 0,
            "recovery_checkpoint_restore_failure": 0,
            "recovery_checkpoint_host_tokens": 0,
            "recovery_checkpoint_device_tokens": 0,
        }

    @staticmethod
    def token_hash(input_ids: List[int]) -> str:
        payload = json.dumps(list(input_ids), separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _status_dict(self, checkpoint: RecoveryCheckpoint) -> Dict[str, Any]:
        out = asdict(checkpoint)
        out["state"] = checkpoint.state.value
        out["generation"] = self.generation
        return out

    def _count_host_tokens(self, last_node: TreeNode) -> int:
        return sum(
            len(node.host_value)
            for node in self.tree_cache.get_node_path(last_node)
            if node.backuped
        )

    def _count_device_tokens(self, last_node: TreeNode) -> int:
        return sum(
            len(node.value)
            for node in self.tree_cache.get_node_path(last_node)
            if not node.evicted and node.value is not None
        )

    def create(
        self,
        *,
        checkpoint_id: str,
        input_ids: List[int],
        extra_key: Optional[str] = None,
        session_id: Optional[str] = None,
        segment_id: Optional[int] = None,
        global_step: Optional[int] = None,
        parent_checkpoint_id: Optional[str] = None,
        tier: str = "host",
        evict_device_after: bool = True,
        sync: bool = True,
    ) -> Dict[str, Any]:
        if checkpoint_id in self.checkpoints:
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": "DUPLICATE_CHECKPOINT_ID",
            }
        if tier != "host":
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": f"UNSUPPORTED_TIER:{tier}",
            }

        requested_tokens = len(input_ids)
        page_size = max(1, int(getattr(self.tree_cache, "page_size", 1) or 1))
        checkpoint_tokens = (requested_tokens // page_size) * page_size
        checkpoint_input_ids = input_ids[:checkpoint_tokens]

        if checkpoint_tokens <= 0:
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "error": "PREFIX_TOO_SHORT",
                "fallback_reason": "PREFIX_TOO_SHORT",
                "requested_tokens": requested_tokens,
                "checkpoint_tokens": checkpoint_tokens,
                "page_size": page_size,
            }

        last_node = self.tree_cache.find_exact_prefix_node(
            checkpoint_input_ids, extra_key
        )
        if last_node is None:
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": "PREFIX_NOT_FOUND",
                "requested_tokens": requested_tokens,
                "checkpoint_tokens": checkpoint_tokens,
                "page_size": page_size,
            }

        started = time.perf_counter()
        checkpoint = RecoveryCheckpoint(
            checkpoint_id=checkpoint_id,
            token_count=checkpoint_tokens,
            token_hash=self.token_hash(checkpoint_input_ids),
            extra_key=extra_key,
            last_node_id=last_node.id,
            node_path_ids=[node.id for node in self.tree_cache.get_node_path(last_node)],
            session_id=session_id,
            segment_id=segment_id,
            global_step=global_step,
            parent_checkpoint_id=parent_checkpoint_id,
            tier=tier,
            state=RecoveryCheckpointState.CREATING,
            created_at=time.time(),
        )

        backup = self.tree_cache.ensure_prefix_host_backup(last_node, sync=sync)
        checkpoint.backup_tokens = int(backup.get("backup_tokens") or 0)
        checkpoint.backup_latency_ms = (time.perf_counter() - started) * 1000
        self.metrics["recovery_checkpoint_create_count"] += 1
        self.metrics["recovery_checkpoint_backup_tokens"] += checkpoint.backup_tokens
        self.metrics["recovery_checkpoint_backup_latency_ms"] += (
            checkpoint.backup_latency_ms
        )
        if not backup.get("success"):
            checkpoint.state = RecoveryCheckpointState.INVALID
            checkpoint.fallback_reason = backup.get("message") or "BACKUP_FAILED"
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": checkpoint.fallback_reason,
                "requested_tokens": requested_tokens,
                "checkpoint_tokens": checkpoint_tokens,
                "page_size": page_size,
                **self._status_dict(checkpoint),
            }

        checkpoint.host_tokens = self.tree_cache.protect_host_prefix(last_node)
        checkpoint.host_pinned = True
        if evict_device_after:
            demote = self.tree_cache.demote_prefix_to_host(last_node, only_unlocked=True)
            checkpoint.requested_host_only_tokens = int(
                demote.get("requested_host_only_tokens") or 0
            )
            checkpoint.actual_host_only_tokens = int(
                demote.get("actual_host_only_tokens") or 0
            )
            checkpoint.shared_or_locked_device_tokens = int(
                demote.get("shared_or_locked_device_tokens") or 0
            )
        checkpoint.device_tokens = self._count_device_tokens(last_node)
        checkpoint.state = RecoveryCheckpointState.HOST_READY

        self.checkpoints[checkpoint_id] = checkpoint
        self._last_nodes[checkpoint_id] = last_node
        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "requested_tokens": requested_tokens,
            "checkpoint_tokens": checkpoint_tokens,
            "page_size": page_size,
            **self._status_dict(checkpoint),
        }

    def restore(
        self,
        *,
        checkpoint_id: str,
        sync: bool = True,
        pin_device: bool = True,
        mem_quota: Optional[int] = None,
    ) -> Dict[str, Any]:
        checkpoint = self.checkpoints.get(checkpoint_id)
        last_node = self._last_nodes.get(checkpoint_id)
        if checkpoint is None or last_node is None:
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": "UNKNOWN_CHECKPOINT",
            }
        if checkpoint.state in {
            RecoveryCheckpointState.RELEASED,
            RecoveryCheckpointState.INVALID,
        }:
            checkpoint.restore_failure_count += 1
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": f"CHECKPOINT_{checkpoint.state.value.upper()}",
                **self._status_dict(checkpoint),
            }

        started = time.perf_counter()
        restored = self.tree_cache.load_prefix_to_device(
            last_node, sync=sync, mem_quota=mem_quota
        )
        if restored is None:
            restored = {
                "success": False,
                "loaded_from_host_tokens": 0,
                "message": "LOAD_BACK_FAILED",
            }
        checkpoint.restore_latency_ms = (time.perf_counter() - started) * 1000
        checkpoint.restore_tokens += int(restored.get("loaded_from_host_tokens") or 0)
        self.metrics["recovery_checkpoint_restore_tokens"] += int(
            restored.get("loaded_from_host_tokens") or 0
        )
        self.metrics["recovery_checkpoint_restore_latency_ms"] += (
            checkpoint.restore_latency_ms
        )
        checkpoint.host_tokens = self._count_host_tokens(last_node)
        checkpoint.device_tokens = self._count_device_tokens(last_node)
        if not restored.get("success"):
            checkpoint.restore_failure_count += 1
            self.metrics["recovery_checkpoint_restore_failure"] += 1
            checkpoint.fallback_reason = restored.get("message") or "RESTORE_FAILED"
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": checkpoint.fallback_reason,
                **self._status_dict(checkpoint),
            }

        if pin_device and not checkpoint.device_pinned:
            self.tree_cache.inc_lock_ref(last_node)
            checkpoint.device_pinned = True

        checkpoint.restore_success_count += 1
        self.metrics["recovery_checkpoint_restore_success"] += 1
        checkpoint.state = RecoveryCheckpointState.DEVICE_READY
        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "prefix_tokens": checkpoint.token_count,
            "already_device_tokens": int(restored.get("already_device_tokens") or 0),
            "loaded_from_host_tokens": int(
                restored.get("loaded_from_host_tokens") or 0
            ),
            **self._status_dict(checkpoint),
        }

    def release(self, checkpoint_id: str) -> Dict[str, Any]:
        checkpoint = self.checkpoints.get(checkpoint_id)
        last_node = self._last_nodes.get(checkpoint_id)
        if checkpoint is None or last_node is None:
            return {
                "success": False,
                "checkpoint_id": checkpoint_id,
                "fallback_reason": "UNKNOWN_CHECKPOINT",
            }

        if checkpoint.device_pinned:
            self.tree_cache.dec_lock_ref(last_node)
            checkpoint.device_pinned = False
        if checkpoint.host_pinned:
            self.tree_cache.release_host_prefix(last_node)
            checkpoint.host_pinned = False

        checkpoint.release_count += 1
        self.metrics["recovery_checkpoint_release_count"] += 1
        checkpoint.state = RecoveryCheckpointState.RELEASED
        out = {"success": True, **self._status_dict(checkpoint)}
        self.checkpoints.pop(checkpoint_id, None)
        self._last_nodes.pop(checkpoint_id, None)
        return out

    def get_status(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        if checkpoint_id:
            checkpoint = self.checkpoints.get(checkpoint_id)
            if checkpoint is None:
                return {
                    "success": False,
                    "checkpoint_id": checkpoint_id,
                    "fallback_reason": "UNKNOWN_CHECKPOINT",
                }
            last_node = self._last_nodes.get(checkpoint_id)
            if last_node is not None:
                checkpoint.host_tokens = self._count_host_tokens(last_node)
                checkpoint.device_tokens = self._count_device_tokens(last_node)
            return {"success": True, **self._status_dict(checkpoint)}
        self.metrics["recovery_checkpoint_host_tokens"] = sum(
            checkpoint.host_tokens for checkpoint in self.checkpoints.values()
        )
        self.metrics["recovery_checkpoint_device_tokens"] = sum(
            checkpoint.device_tokens for checkpoint in self.checkpoints.values()
        )
        return {
            "success": True,
            "generation": self.generation,
            "metrics": dict(self.metrics),
            "checkpoints": [
                self.get_status(checkpoint_id)
                for checkpoint_id in self.checkpoints.keys()
            ],
        }

    def clear(self) -> Dict[str, Any]:
        ids = list(self.checkpoints.keys())
        released = 0
        for checkpoint_id in ids:
            try:
                self.release(checkpoint_id)
                released += 1
            except Exception:
                checkpoint = self.checkpoints.get(checkpoint_id)
                if checkpoint is not None:
                    checkpoint.state = RecoveryCheckpointState.INVALID
        self.checkpoints.clear()
        self._last_nodes.clear()
        self.generation += 1
        return {"success": True, "released": released, "generation": self.generation}
