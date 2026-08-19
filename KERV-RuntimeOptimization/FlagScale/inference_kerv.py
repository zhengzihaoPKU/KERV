"""FlagScale entrypoint for the released KERV/OpenVLA LIBERO pipeline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def _bool(value: object) -> str:
    return "True" if bool(value) else "False"


def _text(value: object, default: str = "") -> str:
    """Render optional OmegaConf values without leaking YAML ``None``.

    The KERV child entrypoint accepts comma-separated allowlists.  A missing
    allowlist is represented as ``None`` by OmegaConf; passing ``str(None)``
    makes the child parser interpret it as a row named ``None`` rather than
    an empty allowlist.
    """
    if value is None:
        return default
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    runtime = cfg.runtime
    rope_cache = runtime.get("rope_cache", {})
    rope_fusion = runtime.get("rope_fusion", {})
    tree_attention_mask = runtime.get("tree_attention_mask", {})
    draft_topk = runtime.get("draft_logsoftmax_topk", {})
    w8a16 = runtime.get("w8a16", {})
    entrypoint = os.path.join(runtime.source_root, runtime.entrypoint)
    work_dir = Path(runtime.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(runtime.get("python_executable", sys.executable)),
        entrypoint,
        "--model_family",
        runtime.model_family,
        "--pretrained_checkpoint",
        runtime.pretrained_checkpoint,
        "--spec_checkpoint",
        runtime.spec_checkpoint,
        "--task_suite_name",
        runtime.task_suite_name,
        "--num_trials_per_task",
        str(runtime.num_trials_per_task),
        "--max_tasks",
        str(runtime.max_tasks),
        "--max_episode_steps",
        str(runtime.max_episode_steps),
        "--num_steps_wait",
        str(runtime.num_steps_wait),
        "--max_planning_steps",
        str(runtime.max_planning_steps),
        "--center_crop",
        _bool(runtime.center_crop),
        "--use_spec",
        _bool(runtime.use_spec),
        "--use_wandb",
        _bool(runtime.use_wandb),
        "--use_kalman_fallback",
        _bool(runtime.use_kalman_fallback),
        "--use_kalman_tree",
        _bool(runtime.use_kalman_tree),
        "--local_log_dir",
        runtime.local_log_dir,
        "--run_id_note",
        runtime.run_id_note,
        "--use_flag_gems",
        _bool(runtime.flag_gems.enabled),
        "--flag_gems_include",
        runtime.flag_gems.include,
        "--flag_gems_record",
        _bool(runtime.flag_gems.record),
        "--flag_gems_record_once",
        _bool(runtime.flag_gems.record_once),
        "--flag_gems_log_path",
        runtime.flag_gems.log_path,
        "--flag_gems_manifest_path",
        runtime.flag_gems.manifest_path,
        "--flag_gems_rms_norm_backend",
        str(runtime.flag_gems.get("rms_norm_backend", "flag_gems")),
        "--use_flagos_embodied_ops",
        _bool(runtime.get("embodied_ops", {}).get("enabled", False)),
        "--flagos_embodied_ops_include",
        str(runtime.get("embodied_ops", {}).get("include", "")),
        "--flagos_embodied_ops_backend",
        str(runtime.get("embodied_ops", {}).get("backend", "auto")),
        "--flagos_embodied_ops_record",
        _bool(runtime.get("embodied_ops", {}).get("record", False)),
        "--flagos_embodied_ops_record_once",
        _bool(runtime.get("embodied_ops", {}).get("record_once", True)),
        "--flagos_embodied_ops_log_path",
        str(runtime.get("embodied_ops", {}).get("log_path", "")),
        "--flagos_embodied_ops_manifest_path",
        str(runtime.get("embodied_ops", {}).get("manifest_path", "")),
        "--flagos_embodied_ops_strict",
        _bool(runtime.get("embodied_ops", {}).get("strict", True)),
        "--use_flagos_linear_fusion",
        _bool(runtime.linear_fusion.enabled),
        "--flagos_qkv_fusion_rows",
        _text(runtime.linear_fusion.qkv_rows),
        "--flagos_gate_up_fusion_rows",
        _text(runtime.linear_fusion.gate_up_rows),
        "--flagos_qkv_fusion_input_sizes",
        _text(runtime.linear_fusion.qkv_input_sizes),
        "--flagos_gate_up_fusion_input_sizes",
        _text(runtime.linear_fusion.gate_up_input_sizes),
        "--flagos_swiglu_fusion_rows",
        _text(runtime.linear_fusion.get("swiglu_rows", "")),
        "--flagos_swiglu_fusion_input_sizes",
        _text(runtime.linear_fusion.get("swiglu_input_sizes", "")),
        "--flagos_swiglu_fusion_backend",
        str(runtime.linear_fusion.get("swiglu_backend", "native_inplace")),
        "--flagos_add_rms_norm_fusion_rows",
        _text(runtime.linear_fusion.get("add_rms_norm_rows", "")),
        "--flagos_add_rms_norm_fusion_input_sizes",
        _text(runtime.linear_fusion.get("add_rms_norm_input_sizes", "")),
        "--flagos_add_rms_norm_fusion_backend",
        str(runtime.linear_fusion.get("add_rms_norm_backend", "native_inplace")),
        "--flagos_linear_fusion_record",
        _bool(runtime.linear_fusion.record),
        "--flagos_linear_fusion_record_once",
        _bool(runtime.linear_fusion.record_once),
        "--flagos_linear_fusion_log_path",
        str(runtime.linear_fusion.log_path),
        "--flagos_linear_fusion_manifest_path",
        str(runtime.linear_fusion.manifest_path),
        "--use_flagos_rope_cache",
        _bool(rope_cache.get("enabled", False)),
        "--flagos_rope_cache_record",
        _bool(rope_cache.get("record", True)),
        "--flagos_rope_cache_record_once",
        _bool(rope_cache.get("record_once", True)),
        "--flagos_rope_cache_log_path",
        str(rope_cache.get("log_path", "")),
        "--flagos_rope_cache_manifest_path",
        str(rope_cache.get("manifest_path", "")),
        "--use_flagos_rope_fusion",
        _bool(rope_fusion.get("enabled", False)),
        "--flagos_rope_resident_key_write",
        _bool(rope_fusion.get("resident_key_write", True)),
        "--flagos_rope_fusion_record",
        _bool(rope_fusion.get("record", True)),
        "--flagos_rope_fusion_record_once",
        _bool(rope_fusion.get("record_once", True)),
        "--flagos_rope_fusion_log_path",
        str(rope_fusion.get("log_path", "")),
        "--flagos_rope_fusion_manifest_path",
        str(rope_fusion.get("manifest_path", "")),
        "--use_flagos_tree_attention_mask",
        _bool(tree_attention_mask.get("enabled", False)),
        "--flagos_tree_attention_mask_record",
        _bool(tree_attention_mask.get("record", True)),
        "--flagos_tree_attention_mask_record_once",
        _bool(tree_attention_mask.get("record_once", True)),
        "--flagos_tree_attention_mask_log_path",
        str(tree_attention_mask.get("log_path", "")),
        "--flagos_tree_attention_mask_manifest_path",
        str(tree_attention_mask.get("manifest_path", "")),
        "--use_flagos_action_head",
        _bool(runtime.action_head.enabled),
        "--flagos_action_vocab_size",
        str(runtime.action_head.action_vocab_size),
        "--flagos_action_head_record",
        _bool(runtime.action_head.record),
        "--flagos_action_head_record_once",
        _bool(runtime.action_head.record_once),
        "--flagos_action_head_log_path",
        str(runtime.action_head.log_path),
        "--flagos_action_head_manifest_path",
        str(runtime.action_head.manifest_path),
        "--use_flagos_w8a16",
        _bool(w8a16.get("enabled", False)),
        "--flagos_w8a16_groups",
        str(w8a16.get("groups", "verify_mlp,verify_attention,vision_mlp,draft_linear")),
        "--flagos_w8a16_min_speedup",
        str(w8a16.get("min_speedup", 1.05)),
        "--flagos_w8a16_strict",
        _bool(w8a16.get("strict", False)),
        "--flagos_w8a16_manifest_path",
        str(w8a16.get("manifest_path", "")),
        "--use_flagos_draft_logsoftmax_topk",
        _bool(draft_topk.get("enabled", False)),
        "--flagos_draft_logsoftmax_topk_manifest_path",
        str(draft_topk.get("manifest_path", "")),
        "--use_flagos_tree_builder",
        _bool(runtime.tree_builder.enabled),
        "--flagos_tree_builder_manifest_path",
        str(runtime.tree_builder.manifest_path),
        "--flagos_tree_static_runtime",
        _bool(runtime.tree_builder.get("static_runtime", False)),
        "--flagos_tree_verify_max_paths",
        str(runtime.tree_builder.get("verify_max_paths", 0)),
        "--flagos_tree_max_depth",
        str(runtime.tree_builder.get("max_depth", 0)),
        "--flagos_compact_tree_mode",
        str(runtime.tree_builder.get("compact_tree_mode", "off")),
        "--flagos_compact_tree_min_confidence",
        str(runtime.tree_builder.get("compact_tree_min_confidence", 0.0)),
        "--flagos_compact_tree_full_max_paths",
        str(runtime.tree_builder.get("compact_tree_full_max_paths", 0)),
        "--flagos_compact_tree_full_max_depth",
        str(runtime.tree_builder.get("compact_tree_full_max_depth", 0)),
        "--flagos_compact_tree_tight_max_paths",
        str(runtime.tree_builder.get("compact_tree_tight_max_paths", 0)),
        "--flagos_compact_tree_tight_max_depth",
        str(runtime.tree_builder.get("compact_tree_tight_max_depth", 0)),
        "--flagos_compact_tree_tight_min_confidence",
        str(runtime.tree_builder.get("compact_tree_tight_min_confidence", float("inf"))),
        "--flagos_tree_cache_topology",
        _bool(runtime.tree_builder.get("cache_topology", True)),
        "--flagos_tree_operatorized",
        _bool(runtime.tree_builder.get("operatorized", False)),
        "--flagos_draft_tree_operatorized",
        _bool(
            runtime.tree_builder.get(
                "draft_tree_operatorized",
                False,
            )
        ),
        "--flagos_verify_accept_operatorized",
        _bool(runtime.tree_builder.get("verify_accept_operatorized", False)),
        "--flagos_tree_node_buckets",
        str(runtime.tree_builder.get("node_buckets") or ""),
        "--flagos_compile_verifier",
        _bool(runtime.tree_builder.get("compile_verifier", False)),
        "--flagos_compile_verifier_mode",
        str(runtime.tree_builder.get("compile_mode", "reduce-overhead")),
        "--flagos_cuda_graph_max_entries",
        str(runtime.tree_builder.get("cuda_graph_max_entries", 1)),
        "--flagos_cuda_graph_capture_past_length",
        str(runtime.tree_builder.get("cuda_graph_capture_past_length", -1)),
        "--flagos_kalman_batch",
        _bool(runtime.tree_builder.get("kalman_batch", False)),
        "--flagos_kv_commit_batched",
        _bool(runtime.tree_builder.get("kv_commit_batched", False)),
        "--flagos_fused_control_transfer",
        _bool(runtime.tree_builder.get("fused_control_transfer", False)),
        "--flagos_persistent_runtime",
        _bool(runtime.tree_builder.get("persistent_runtime", False)),
        "--flagos_resident_kv_cache",
        _bool(runtime.tree_builder.get("resident_kv_cache", False)),
        "--flagos_persistent_prefix_capacity",
        str(runtime.tree_builder.get("persistent_prefix_capacity", 288)),
        "--flagos_persistent_tree_capacity",
        str(runtime.tree_builder.get("persistent_tree_capacity", 320)),
        "--flagos_exact_node_templates",
        _bool(runtime.tree_builder.get("exact_node_templates", False)),
        "--flagos_fixed_workspace_layout",
        _bool(runtime.tree_builder.get("fixed_workspace_layout", False)),
        "--flagos_prewarm_graph_buckets",
        _bool(runtime.tree_builder.get("prewarm_graph_buckets", False)),
        "--flagos_persistent_input_buffers",
        _bool(runtime.tree_builder.get("persistent_input_buffers", False)),
        "--flagos_static_tree_attention",
        str(runtime.tree_builder.get("static_tree_attention", "off")),
        "--flagos_cuda_warmup_seconds",
        str(runtime.get("cuda_warmup_seconds", 0.0)),
        "--flagos_timing_backend",
        str(runtime.get("timing_backend", "cpu")),
        "--flagos_async_runtime_logging",
        _bool(runtime.get("async_runtime_logging", False)),
        "--flagos_system_optimization",
        _bool(runtime.get("system_optimization", {}).get("enabled", False)),
        "--flagos_prompt_cuda_graph",
        str(runtime.get("system_optimization", {}).get("prompt_cuda_graph", "off")),
        "--flagos_draft_cuda_graph",
        str(runtime.get("system_optimization", {}).get("draft_cuda_graph", "off")),
        "--flagos_shared_graph_pool",
        _bool(runtime.get("system_optimization", {}).get("shared_graph_pool", True)),
        "--flagos_process_level_warmup",
        _bool(runtime.get("system_optimization", {}).get("process_level_warmup", False)),
        "--flagos_persistent_control_buffer",
        _bool(runtime.get("system_optimization", {}).get("persistent_control_buffer", False)),
        "--flagos_persistent_decode_workspace",
        _bool(runtime.get("system_optimization", {}).get("persistent_decode_workspace", False)),
        "--flagos_verifier_gate_during_warmup",
        _bool(runtime.get("system_optimization", {}).get("verifier_gate_during_warmup", False)),
        "--flagos_prompt_token_cache",
        _bool(runtime.get("system_optimization", {}).get("prompt_token_cache", False)),
        "--flagos_pinned_image_buffer",
        _bool(runtime.get("system_optimization", {}).get("pinned_image_buffer", False)),
        "--flagos_async_image_transfer",
        _bool(runtime.get("system_optimization", {}).get("async_image_transfer", False)),
        "--flagos_persistent_draft_kv_cache",
        _bool(runtime.get("system_optimization", {}).get("persistent_draft_kv_cache", False)),
        "--flagos_draft_graph_accept_buckets",
        ",".join(
            str(value)
            for value in runtime.get("system_optimization", {}).get(
                "draft_graph_accept_buckets", [1, 2, 3, 4, 5, 6, 7]
            )
        ),
        "--flagos_commit_draft_cuda_graph",
        str(runtime.get("system_optimization", {}).get("commit_draft_cuda_graph", "off")),
        "--flagos_commit_draft_stream_overlap",
        str(runtime.get("system_optimization", {}).get("commit_draft_stream_overlap", "off")),
        "--flagos_task_level_graph_warmup",
        _bool(runtime.get("system_optimization", {}).get("task_level_graph_warmup", False)),
        "--flagos_inference_mode",
        _bool(runtime.get("system_optimization", {}).get("inference_mode", False)),
        "--flagos_allocator_backend",
        str(runtime.get("system_optimization", {}).get("allocator_backend", "auto")),
        "--flagos_async_logs",
        _bool(runtime.get("system_optimization", {}).get("async_runtime_logging", False)),
        "--flagos_episode_device_warmup",
        str(runtime.get("system_optimization", {}).get("episode_device_warmup", "off")),
        "--flagos_episode_warmup_duration_ms",
        str(runtime.get("system_optimization", {}).get("episode_warmup_duration_ms", "auto")),
        "--flagos_joint_commit_draft_graph",
        str(runtime.get("system_optimization", {}).get("joint_commit_draft_graph", "off")),
        "--flagos_joint_graph_accept_buckets",
        ",".join(
            str(value)
            for value in runtime.get("system_optimization", {}).get(
                "joint_graph_accept_buckets", [1, 2, 3, 4, 5, 6, 7]
            )
        ),
        "--flagos_fixed_draft_input_workspace",
        _bool(runtime.get("system_optimization", {}).get("fixed_draft_input_workspace", False)),
        "--flagos_numa_affinity",
        str(runtime.get("system_optimization", {}).get("numa_affinity", "off")),
    ]
    command.extend(["--collect_spec_errors", _bool(runtime.collect_spec_errors)])
    if runtime.spec_error_output:
        command.extend(["--spec_error_output", runtime.spec_error_output])
    if runtime.spec_history_output:
        command.extend(["--spec_history_output", runtime.spec_history_output])
    if runtime.runtime_stats_path:
        command.extend(["--runtime_stats_path", runtime.runtime_stats_path])
    release_root = Path(__file__).resolve().parents[1]
    allocator_backend = str(
        runtime.get("system_optimization", {}).get("allocator_backend", "auto")
    ).lower()
    if allocator_backend == "cuda_malloc_async":
        # The allocator backend must be selected before the child imports
        # torch/CUDA.  ``auto`` deliberately leaves the deployment default
        # unchanged until the P95 gate has been measured.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    elif allocator_backend not in {"auto", "native"}:
        raise ValueError(
            "runtime.system_optimization.allocator_backend must be "
            "auto, native, or cuda_malloc_async"
        )
    python_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(release_root), str(release_root / "openvla"), python_path)
        if value
    )
    print("[FlagScale/KERV] launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=work_dir, check=True)


if __name__ == "__main__":
    main()
