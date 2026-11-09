"""Copy the exact code and compact artifacts behind the report into code/."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "apart_swarm_submission" / "code"

FILES = {
    "apart_swarm_submission/assemble_code_package.py": "assemble_code_package.py",
    "apart_swarm_submission/plot_six_seed_trajectories.py": "plot_six_seed_trajectories.py",
    "kaggle_deploy/probe_retrain/kernel_full/kernel-metadata.json": "kaggle_reproduction/probe_retrain_kernel_metadata.json",
    "kaggle_deploy/retrained_holdout/kernel_full/kernel-metadata.json": "kaggle_reproduction/holdout_kernel_metadata.json",
    "beehive_sim.py": "beehive_sim.py",
    "bee_feasibility.py": "bee_feasibility.py",
    "bee_menu_qwen_policy.py": "bee_menu_qwen_policy.py",
    "bee_local_qwen_policy.py": "bee_local_qwen_policy.py",
    "generate_bee_sft_dataset.py": "generate_bee_sft_dataset.py",
    "generate_bee_edgecase_sft.py": "generate_bee_edgecase_sft.py",
    "beehive_agents.ipynb": "beehive_agents.ipynb",
    "beehive_agents_qwen35.ipynb": "beehive_agents_qwen35.ipynb",
    "train_qwen35_bee_agents.ipynb": "train_qwen35_bee_agents.ipynb",
    "bee_menu_finetune.ipynb": "bee_menu_finetune.ipynb",
    "beehive_vector_probe_steering.ipynb": "beehive_vector_probe_steering.ipynb",
    "beehive_column_steering.ipynb": "beehive_column_steering.ipynb",
    "beehive_column_steering_4h.ipynb": "beehive_column_steering_4h.ipynb",
    "beehive_mixed_probe_2h.ipynb": "beehive_mixed_probe_2h.ipynb",
    "beehive_direction_audit.ipynb": "beehive_direction_audit.ipynb",
    "beehive_probe_retrained.ipynb": "beehive_probe_retrained.ipynb",
    "beehive_retrained_probe_holdout.ipynb": "beehive_retrained_probe_holdout.ipynb",
    "build_bee_menu_finetune_notebook.py": "build_bee_menu_finetune_notebook.py",
    "build_vector_probe_notebook.py": "build_vector_probe_notebook.py",
    "build_bee_column_steering_notebook.py": "build_bee_column_steering_notebook.py",
    "build_bee_column_steering_4h_notebook.py": "build_bee_column_steering_4h_notebook.py",
    "build_bee_mixed_probe_2h_notebook.py": "build_bee_mixed_probe_2h_notebook.py",
    "build_bee_direction_audit_notebook.py": "build_bee_direction_audit_notebook.py",
    "build_bee_probe_retrain_notebook.py": "build_bee_probe_retrain_notebook.py",
    "build_bee_retrained_holdout_notebook.py": "build_bee_retrained_holdout_notebook.py",
    "MODEL_DESIGN.md": "MODEL_DESIGN.md",
    "TRAINING_DATA.md": "TRAINING_DATA.md",
    "BEE_PROBE_RETRAINED.md": "BEE_PROBE_RETRAINED.md",
    "BEE_RETRAINED_HOLDOUT_100_103.md": "BEE_RETRAINED_HOLDOUT_100_103.md",
    "requirements.txt": "requirements.txt",
    "tests/test_beehive_sim.py": "tests/test_beehive_sim.py",
    "bee_decision_sft_dataset.zip": "data/bee_decision_sft_dataset.zip",
    "data/bee_decision_sft/manifest.json": "data/bee_decision_sft_manifest.json",
    "data/bee_decision_sft/DATASET_CARD.md": "data/DATASET_CARD.md",
    "data/bee_edgecase_sft/edgecase_train.jsonl": "data/edgecase_train.jsonl",
    "data/bee_edgecase_sft/edgecase_validation.jsonl": "data/edgecase_validation.jsonl",
    "data/bee_edgecase_sft/manifest.json": "data/edgecase_manifest.json",
    "kaggle_deploy/beehive_qwen35/dataset/beehive_qwen35_runtime_bundle.zip": "artifacts/runtime/beehive_qwen35_runtime_bundle.zip",
    "kaggle_deploy/beehive_qwen35/dataset/probe_source_metadata.json": "artifacts/runtime/probe_source_metadata.json",
    "kaggle_deploy/menu_adapter/dataset/adapter_model.safetensors": "artifacts/menu_adapter/adapter_model.safetensors",
    "kaggle_deploy/menu_adapter/dataset/adapter_config.json": "artifacts/menu_adapter/adapter_config.json",
    "kaggle_deploy/menu_adapter/dataset/tokenizer.json": "artifacts/menu_adapter/tokenizer.json",
    "kaggle_deploy/menu_adapter/dataset/tokenizer_config.json": "artifacts/menu_adapter/tokenizer_config.json",
    "kaggle_deploy/menu_adapter/dataset/processor_config.json": "artifacts/menu_adapter/processor_config.json",
    "kaggle_deploy/menu_adapter/dataset/chat_template.jinja": "artifacts/menu_adapter/chat_template.jinja",
    "kaggle_deploy/menu_adapter/dataset/training_report.json": "artifacts/menu_adapter/training_report.json",
    "kaggle_deploy/mixed_probe_2h/full_output/bee_mixed_probe_20261107T221054Z/new_probe_layer_13.npz": "artifacts/mixed_probe/new_probe_layer_13.npz",
    "kaggle_deploy/mixed_probe_2h/full_output/bee_mixed_probe_20261107T221054Z/new_probe_layer_14.npz": "artifacts/mixed_probe/new_probe_layer_14.npz",
    "kaggle_deploy/mixed_probe_2h/full_output/bee_mixed_probe_20261107T221054Z/experiment_report.json": "artifacts/mixed_probe/experiment_report.json",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_90_baseline/actions.jsonl": "artifacts/mixed_probe/source_90_baseline/actions.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_90_baseline/vectors.jsonl": "artifacts/mixed_probe/source_90_baseline/vectors.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_90_baseline/metadata.json": "artifacts/mixed_probe/source_90_baseline/metadata.json",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_90_bootstrap_equal/actions.jsonl": "artifacts/mixed_probe/source_90_bootstrap_equal/actions.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_90_bootstrap_equal/vectors.jsonl": "artifacts/mixed_probe/source_90_bootstrap_equal/vectors.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_90_bootstrap_equal/metadata.json": "artifacts/mixed_probe/source_90_bootstrap_equal/metadata.json",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_91_baseline/actions.jsonl": "artifacts/mixed_probe/source_91_baseline/actions.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_91_baseline/vectors.jsonl": "artifacts/mixed_probe/source_91_baseline/vectors.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_91_baseline/metadata.json": "artifacts/mixed_probe/source_91_baseline/metadata.json",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_91_bootstrap_equal/actions.jsonl": "artifacts/mixed_probe/source_91_bootstrap_equal/actions.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_91_bootstrap_equal/vectors.jsonl": "artifacts/mixed_probe/source_91_bootstrap_equal/vectors.jsonl",
    "kaggle_deploy/mixed_probe_2h/replay_source/bee_mixed_probe_20261107T221054Z/source_91_bootstrap_equal/metadata.json": "artifacts/mixed_probe/source_91_bootstrap_equal/metadata.json",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/retrained_probe_layer_13.npz": "artifacts/retrained_probe/retrained_probe_layer_13.npz",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/retrained_probe_layer_14.npz": "artifacts/retrained_probe/retrained_probe_layer_14.npz",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/retrained_probe_report.json": "artifacts/retrained_probe/retrained_probe_report.json",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/retrained_probe_validation.csv": "results/retrained_probe_validation.csv",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/final_step_changes.csv.gz": "results/final_step_changes_114_115.csv.gz",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/final_hive_summary.csv": "results/final_hive_summary_114_115.csv",
    "kaggle_deploy/probe_retrain/full_output/bee_probe_retrained_20261108T081009Z/final_hive_vectors.csv": "results/final_hive_vectors_114_115.csv",
    "kaggle_deploy/retrained_holdout/full_output/bee_retrained_holdout_20261108T160028Z/holdout_report.json": "results/holdout_report_100_103.json",
    "kaggle_deploy/retrained_holdout/full_output/bee_retrained_holdout_20261108T160028Z/holdout_step_changes.csv.gz": "results/holdout_step_changes_100_103.csv.gz",
    "kaggle_deploy/retrained_holdout/full_output/bee_retrained_holdout_20261108T160028Z/holdout_run_summary.csv": "results/holdout_run_summary_100_103.csv",
    "kaggle_deploy/retrained_holdout/full_output/bee_retrained_holdout_20261108T160028Z/holdout_final_vectors.csv": "results/holdout_final_vectors_100_103.csv",
}


def main() -> None:
    hashes = {}
    for source, target in FILES.items():
        src = ROOT / source
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = DEST / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[target] = hashlib.sha256(dst.read_bytes()).hexdigest()
    manifest = {
        "description": "Source paths and SHA-256 hashes for copied research files",
        "files": [{"source": source, "target": target, "sha256": hashes[target]}
                  for source, target in FILES.items()],
    }
    (DEST / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Copied {len(FILES)} files into {DEST}")


if __name__ == "__main__":
    main()
