# Probe-guided bee-swarm experiment: code and artifacts

This directory accompanies *Probe-Guided Steering of an LLM Worker-Bee Swarm
in a Simulated Hive*, an Apart Swarm Alignment Hackathon 2026 report by
Faithful Chukwunwogor (Independent). The report's GitHub URL is a placeholder:
**https://github.com/USERNAME/REPO**. Replace it after uploading this folder.

The code is a synthetic AI-agent testbed, not a biologically validated
honey-bee model. The report's six-seed tables come from saved Kaggle runs.

## Contents

| Stage | Main files | Function |
| --- | --- | --- |
| Beehive | beehive_sim.py, beehive_agents.ipynb, MODEL_DESIGN.md, tests/test_beehive_sim.py | Spatial world, 17 bounded actions, local observations, hidden 14-value observer, action transcripts |
| Local Qwen runtime | bee_local_qwen_policy.py, bee_feasibility.py, bee_menu_qwen_policy.py, beehive_agents_qwen35.ipynb | Batched local-model decisions from one frozen view per batch, feasible menu, sequential application |
| Initial synthetic SFT | generate_bee_sft_dataset.py, train_qwen35_bee_agents.ipynb, data/bee_decision_sft_dataset.zip | 11,900 balanced examples; half of training split used for rank-16 LoRA |
| Continuation training | generate_bee_edgecase_sft.py, bee_menu_finetune.ipynb, data/edgecase files | Failure-heavy menu examples and continued LoRA |
| Probe development | beehive_vector_probe_steering.ipynb, beehive_mixed_probe_2h.ipynb | Earlier probe and mixed-world trajectories used as retraining inputs |
| Main intervention | beehive_probe_retrained.ipynb | Fit new 14-output probe, validate on separate seeds, screen signed directions, test seeds 114–115 |
| Frozen repeat | beehive_retrained_probe_holdout.ipynb | Load same probe, directions, and strength without fitting; test seeds 100–103 |
| Audit and figures | results/, plot_six_seed_trajectories.py | Saved transcripts, vectors, probe errors, six-seed plot source |

The build_bee and build_vector Python scripts generate the notebooks. The two
final notebooks are the completed full-run versions with SMOKE=False.

## Saved inputs and output provenance

- artifacts/menu_adapter contains the continued menu LoRA used in all six
  final hive trials. The Qwen base weights are not included; notebooks load
  unsloth/Qwen3.5-0.8B.
- artifacts/runtime contains the exact simulator runtime ZIP and source-run
  metadata used in Kaggle. The root beehive_sim.py has the same SHA-256 as
  the archived engine.
- artifacts/mixed_probe contains the preceding probe's layer-13/14 matrices,
  report, and all four source trajectory folders: baseline and steered worlds
  for seeds 90 and 91. The final notebooks replay these exact actions and
  verify every saved vector before training or evaluation.
- artifacts/retrained_probe contains the final layer-13/14 matrices and
  report. results contains held-out probe errors and both final action-level
  timestamped vector transcripts, summaries, and per-seed final vectors.
- PACKAGE_MANIFEST.json lists each copied file's original path and SHA-256.

Kaggle notebook pages (some datasets and outputs may be private):

- Continuation training: https://www.kaggle.com/code/faithfulchukwunwogor/beehive-qwen35-feasible-menu-finetune
- Source trajectories: https://www.kaggle.com/code/faithfulchukwunwogor/beehive-mixed-probe-two-hour
- Retrained probe and seeds 114–115: https://www.kaggle.com/code/faithfulchukwunwogor/beehive-probe-retrained
- Frozen repeat on seeds 100–103: https://www.kaggle.com/code/faithfulchukwunwogor/beehive-retrained-probe-holdout

## Reproduction sequence

For the closest rerun of the reported numbers, use the archived adapter,
earlier probe, and all four source trajectories already in artifacts; run
the two final notebooks in steps 5–6. Steps 1–4 explain how those inputs
were produced. Repeating training in steps 1–4 can produce new weights and
is a separate replication attempt.

1. Generate the initial action dataset with generate_bee_sft_dataset.py, or
   use the included ZIP. It uses the simulator's own observation and action
   schema.
2. Run train_qwen35_bee_agents.ipynb on a GPU with Unsloth. It trains rank-16
   LoRA for two epochs on 4,760 action-stratified training rows.
3. Use generate_bee_edgecase_sft.py or the included edge-case JSONL, then run
   bee_menu_finetune.ipynb for one epoch of continuation training.
4. Run beehive_mixed_probe_2h.ipynb to regenerate the four seed-90/91 source
   worlds and earlier probe if testing the full historical pipeline. The
   archived outputs required for the final notebooks are already included.
5. Run beehive_probe_retrained.ipynb with the runtime, continued adapter, and
   preceding mixed-probe output attached. It trains the 14-output probe on
   2,623 examples, selects layers 13/14, calibrates signed combinations, and
   evaluates seeds 114/115.
6. Run beehive_retrained_probe_holdout.ipynb with the same inputs and the
   retrained-probe output attached. It replays original source examples to
   recover fixed activation-norm scalars, then evaluates seeds 100–103. It
   does not refit the probe or direction weights.

The notebooks search recursively under /kaggle/input for their input
filenames. Batch size is 16; 50 adults act once per 15-minute round; two
simulated hours equal 400 decisions per arm. The observer snapshots the
14-vector after each action.

### Exact Kaggle rerun of the reported final experiment

Use the Kaggle account that can read the three inputs named below. The two
JSON files in kaggle_reproduction preserve the original attachment lists.
The simplest route from this GitHub folder is to upload the entire code
directory as **one Kaggle Dataset**. Open each final notebook as Copy & Edit,
select a GPU, attach only that uploaded dataset, and leave Internet on for
its first-cell package installation and base-model download. All source
trajectory folders, the continued LoRA adapter, earlier probe, and saved
retrained probe are already inside this package. Do not attach the original
Kaggle kernel outputs at the same time: their duplicate filenames make the
notebooks' recursive input search ambiguous. Alternatively, use the original
Kaggle notebook pages with their existing input attachments and no upload of
this package. Keep the full-run settings in either route.

**First, reproduce seeds 114–115.** Open beehive-probe-retrained. The package
provides the inputs originally attached as the beehive-qwen35-runtime and
beehive-qwen35-menu-adapter datasets and beehive-mixed-probe-two-hour output.
Its full-run settings are:
SMOKE=False, VALIDATION_SEEDS=(110,111,112), SCREEN_SEEDS=(110,111),
FINAL_SEEDS=(114,115), HOURS=2.0, BATCH_SIZE=16, MAX_NEW_TOKENS=160.
The notebook replays four seed-90/91 source worlds and asserts zero vector
error, fits 2,623 examples, selects layers 13 and 14, reports probe error on
seeds 111–112, screens signed directions, fixes strength 0.045, then writes
final_step_changes.csv.gz, final_hive_summary.csv, final_hive_vectors.csv,
retrained_probe_layer_13.npz, retrained_probe_layer_14.npz, and
retrained_probe_report.json in its output directory. Check the two final
scores per arm against the report table before continuing.

**Second, reproduce seeds 100–103 without refitting.** Open
beehive-retrained-probe-holdout with only the same package attached. Its
saved retrained probe and report are under artifacts/retrained_probe.
Its settings are SMOKE=False,
FINAL_SEEDS=(100,101,102,103), HOURS=2.0, BATCH_SIZE=16,
MAX_NEW_TOKENS=160, FRACTION=0.045, POSITION=all. The notebook loads the
saved probe, signs, and weights. It replays the original 2,623 training
examples only to recover two fixed activation norms, then runs 12 final arms
and writes holdout_step_changes.csv.gz, holdout_run_summary.csv,
holdout_final_vectors.csv, and holdout_report.json. Its baseline arms must
match the prior same-seed baseline vectors exactly.

**Verify scores.** The bundled verify_report_results.py audits 7,200
post-action snapshots and prints baseline/equal/learned scores after actions
200 and 400. Run it from this code directory with
python3 verify_report_results.py. The expected two-hour mean scores are
0.795420073 baseline, 0.795738923 equal, and 0.796138766 learned. The
learned paired mean gain is +0.000718693 and is positive on five of six seeds.
To check a new Kaggle rerun, replace the two files under results whose names
end in step_changes_100_103.csv.gz and step_changes_114_115.csv.gz with
the new outputs, preserving those filenames, then rerun the verifier.
Timestamped folder names and runtime can differ; exact floating-point scores
may vary with library or model revisions, because the notebooks install
current package versions and do not pin a base-model commit. The archived
transcripts and adapter are the authoritative source for the report's numbers.

To regenerate the figure from included results, run the plot script with:
--latest results/holdout_step_changes_100_103.csv.gz,
--prior results/final_step_changes_114_115.csv.gz, and
--output ../figures/six_seed_mean_trajectories.png.
The LaTeX source is ../report.tex; compile it from the parent directory with
tectonic report.tex --outdir . or another TeX engine. The PDF is ../report.pdf.

## Provenance relative to the sprint

The simulator, synthetic SFT generator, and first Qwen LoRA predate the
November 6–8, 2026 sprint (saved artifacts date to November 2–3). The local
menu and continuation-training assets were prepared by November 5. The
retrained probe, causal direction screen, and paired six-seed evaluation are
the main sprint work and immediate follow-up. Earlier probe and mixed-world
notebooks are included because the main experiment uses their saved data.
The beehive design cites TaskSelSim, ApisRAM, SIMBACA, and Bee++; no source
code from those projects was copied.

The action teacher and metric are synthetic. Physically executable labels
do not prove biological correctness or optimal choices. Seeds 100–103 had
been used in earlier experiments with other directions, so the six-seed
result remains exploratory rather than fully untouched confirmation.
