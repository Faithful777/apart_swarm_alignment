from pathlib import Path
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''# Vector probe steering for beehive agents

This notebook loads the trained **Qwen3.5-0.8B LoRA** bee policy, finds decoder layers whose response features predict the 14-component **immediate action change** in hive state, and compares three six-hour runs: baseline, mean probe direction, and backprop-learned weighted probe direction. Every run starts from the same evaluation seed and uses frozen observations in GPU batches of up to 16 bees. The probe labels and global state vectors remain researcher-only; agents receive only their local observations.

The method extends [Activation Steering With Mean Response Probes](https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes): it keeps response-activation pooling, layer screening, and inference-time forward hooks, while replacing the scalar classifier with a 14-output signed regression probe. The training data are the previous complete baseline run; its logged actions are replayed exactly to recover observations. Final comparisons use a different world seed. This is an exploratory single-seed experiment, not evidence of biological validity or a causal interpretation of the probe weights.''')

md(r'''## 1. Runtime and settings

Enable a Kaggle GPU and internet access, and attach the private `beehive-qwen35-runtime` dataset. Set `SMOKE=True` to run the short execution check. The published full notebook uses `SMOKE=False`.''')
code(r'''%pip install --upgrade --quiet unsloth unsloth_zoo "transformers>=5.0.0" "trl>=0.27.0" datasets accelerate peft scikit-learn''')
code(r'''from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, math, sys, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from IPython.display import display, FileLink
from unsloth import FastLanguageModel
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_ROOT = Path('/kaggle/input')
sim_files = list(INPUT_ROOT.rglob('beehive_sim.py'))
audit_files = list(INPUT_ROOT.rglob('probe_source_actions.jsonl'))
metadata_files = list(INPUT_ROOT.rglob('probe_source_metadata.json'))
adapters = [p.parent for p in INPUT_ROOT.rglob('adapter_config.json')
            if (p.parent / 'adapter_model.safetensors').is_file()]
if not (sim_files and audit_files and metadata_files and adapters):
    raise FileNotFoundError('Attach the beehive-qwen35-runtime dataset with simulator, adapter, and probe source audit.')
SIM_DIR, ADAPTER_DIR = sim_files[0].parent, adapters[0]
sys.path.insert(0, str(SIM_DIR))
from beehive_sim import Config, World, Simulation, VECTOR_NAMES, SYSTEM_PROMPT, validate_action, weighted_research_score
from bee_local_qwen_policy import BeeLocalQwenPolicy
SOURCE_AUDIT = [json.loads(line) for line in audit_files[0].open()]
SOURCE_META = json.loads(metadata_files[0].read_text())
assert len(SOURCE_AUDIT) == SOURCE_META['events_recorded'] == 1200
assert len(VECTOR_NAMES) == 14
print('Simulator:', SIM_DIR)
print('Adapter:', ADAPTER_DIR)
print('Logged training actions:', len(SOURCE_AUDIT))''')
code(r'''SMOKE = False                 # Set True for a short end-to-end execution check.
BATCH_SIZE = 16
N_SELECTED_LAYERS = 2
PROBE_SOURCE_LIMIT = 1200 if not SMOKE else 72
CALIBRATION_HOURS = 0.5 if not SMOKE else 0.25
EVALUATION_HOURS = 6.0 if not SMOKE else 0.25
STRENGTH_GRID = [0.005, 0.015, 0.03] if not SMOKE else [0.005]
CALIBRATION_SEED = 44
EVALUATION_SEED = 43
MAX_NEW_TOKENS = 160
RUN_ROOT = Path('/kaggle/working') / ('vector_probe_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
RUN_ROOT.mkdir(parents=True, exist_ok=False)
print('Run root:', RUN_ROOT)
print('Smoke:', SMOKE, 'feature examples:', PROBE_SOURCE_LIMIT,
      'evaluation hours:', EVALUATION_HOURS)''')

md(r'''## 2. Reconstruct probe examples from the prior run

For action *i*, the target is `post_action_vector − pre_action_vector`, as logged in `immediate_action_delta`. It is signed and has 14 entries. An unchanged attribute has target **exactly zero**. Background changes between decisions are excluded. The simulator replays all saved decisions and checks every resulting vector against the original run before any feature is used.''')
code(r'''class ReplayPolicy:
    label = 'exact_recorded_action_replay'
    def __init__(self, rows):
        self.rows, self.index, self.observations = rows, 0, []
        self.last_model = 'recorded local Qwen decisions'
        self.last_usage = {}
    def decide_batch(self, observations):
        expected = self.rows[self.index:self.index + len(observations)]
        assert len(expected) == len(observations)
        assert [o['self']['id'] for o in observations] == [r['bee_id'] for r in expected]
        self.observations.extend(copy.deepcopy(observations))
        self.index += len(expected)
        return [r['action'] for r in expected]

source_config = Config(**SOURCE_META['calibration']['config'])
replay = ReplayPolicy(SOURCE_AUDIT)
replayed = Simulation(World(source_config), replay).run(decision_batch_size=BATCH_SIZE)
assert len(replayed) == len(SOURCE_AUDIT) == len(replay.observations)
# Exact post-action vectors and outcomes are the provenance check.
assert all(row['vector'] == original['vector'] for row, original in
           zip(replayed, [json.loads(line) for line in (audit_files[0].parent / 'probe_source_vectors.jsonl').open()]))
indices = np.arange(len(SOURCE_AUDIT))
if PROBE_SOURCE_LIMIT < len(indices):
    indices = np.unique(np.linspace(0, len(indices) - 1, PROBE_SOURCE_LIMIT, dtype=int))
cut1, cut2 = int(0.60 * len(indices)), int(0.80 * len(indices))
train_pos = np.arange(0, cut1)
valid_pos = np.arange(cut1, cut2)
weight_pos = np.arange(cut2, len(indices))
targets = np.asarray([SOURCE_AUDIT[i]['immediate_action_delta'] for i in indices], dtype=np.float32)
assert targets.shape == (len(indices), len(VECTOR_NAMES))
assert np.isfinite(targets).all()
print('Time-blocked splits:', len(train_pos), 'probe train;', len(valid_pos),
      'layer validation;', len(weight_pos), 'weight calibration')
print('Nonzero target entries:', int(np.count_nonzero(targets)), '/', targets.size)''')

md(r'''## 3. Load the trained model and discover its decoder layers

The notebook identifies the text decoder `ModuleList` from the live model. It screens **every decoder layer**. Each feature is the mean residual activation over the canonical JSON action response in a teacher-forced forward pass; this is the response representation available after the action text is formed. The action comes from the recorded model output. Canonical JSON is used because the original run saved parsed actions rather than raw whitespace.''')
code(r'''base_policy = BeeLocalQwenPolicy.from_adapter(
    ADAPTER_DIR, max_seq_length=3072, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE)
model, processor = base_policy.model, base_policy.tokenizer
model.eval()

def text_decoder_candidates(model):
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 8:
            continue
        first = module[0]
        if not any(hasattr(first, field) for field in ('self_attn', 'linear_attn', 'mlp')):
            continue
        score = len(module) + 100 * ('language_model' in name) + 50 * ('text' in name)
        score -= 100 * any(word in name for word in ('visual', 'vision', 'image'))
        candidates.append((score, name, module))
    return sorted(candidates, key=lambda item: item[0], reverse=True)

candidates = text_decoder_candidates(model)
if not candidates:
    raise RuntimeError('No text decoder ModuleList found; inspect model.named_modules().')
for score, name, layers in candidates[:5]:
    print(f'candidate {name}: {len(layers)} layers; score {score}')
_, LAYER_PATH, DECODER_LAYERS = candidates[0]
N_LAYERS = len(DECODER_LAYERS)
print('Selected text decoder:', LAYER_PATH, 'with', N_LAYERS, 'layers')''')

md(r'''## 4. Extract mean response features and rank layers

A forward hook pools only response-token activations, so it does not retain full hidden-state tensors. Probe training uses a ridge-regularized linear map with 14 outputs. Targets are scaled by each attribute's typical nonzero change. Rows with any real change get extra weight; the remaining entries, including all unchanged attributes, still train toward zero. Layer selection uses the later time-blocked validation segment. The table reports changed-component sign accuracy, change-detection F1, and weighted error against a zero predictor.''')
code(r'''def encoded_chat(messages, add_generation_prompt):
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
        return_tensors='pt', return_dict=True)
    return {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}

pooled = {}
response_span = [0, 0]
handles = []
for layer_i, layer in enumerate(DECODER_LAYERS):
    def capture(_module, _inputs, output, layer_i=layer_i):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f'Unexpected decoder layer output at layer {layer_i}')
        start, end = response_span
        if hidden.shape[0] != 1 or end <= start or end > hidden.shape[1]:
            raise RuntimeError('Response activation span is outside decoder output')
        pooled[layer_i] = hidden[0, start:end].float().mean(dim=0).detach().cpu().numpy()
    handles.append(layer.register_forward_hook(capture))

features = None
extract_start = time.perf_counter()
try:
    for local_i, source_i in enumerate(indices):
        observation = replay.observations[int(source_i)]
        prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(observation, ensure_ascii=False, separators=(',', ':'), allow_nan=False)},
        ]
        action_text = json.dumps(SOURCE_AUDIT[int(source_i)]['action'], ensure_ascii=False,
                                 separators=(',', ':'), allow_nan=False)
        prompt_encoded = encoded_chat(prompt, True)
        full_encoded = encoded_chat(prompt + [{'role': 'assistant', 'content': action_text}], False)
        pids, fids = prompt_encoded['input_ids'][0], full_encoded['input_ids'][0]
        if not torch.equal(pids, fids[:len(pids)]):
            raise RuntimeError('Chat-template prompt is not a prefix of the action transcript')
        response_span[:] = [len(pids), len(fids) - 1]  # omit final chat end token
        if response_span[1] <= response_span[0] or len(fids) > 3072:
            raise RuntimeError('Empty or overlong response span')
        pooled.clear()
        with torch.inference_mode():
            result = model(**{k: v.to(model.device) for k, v in full_encoded.items()}, use_cache=False)
        del result
        if len(pooled) != N_LAYERS:
            raise RuntimeError(f'Captured {len(pooled)} of {N_LAYERS} decoder layers')
        if features is None:
            dim = len(pooled[0])
            features = np.empty((len(indices), N_LAYERS, dim), dtype=np.float32)
        for layer_i in range(N_LAYERS):
            features[local_i, layer_i] = pooled[layer_i]
        if (local_i + 1) % 100 == 0 or local_i + 1 == len(indices):
            print(f'Features {local_i + 1}/{len(indices)} in {(time.perf_counter()-extract_start)/60:.1f} min')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
finally:
    for handle in handles: handle.remove()
assert features is not None and np.isfinite(features).all()
print('Feature tensor:', features.shape)''')

code(r'''# A nonzero target is calibrated by the 75th percentile of its nonzero training changes.
scales = []
for j in range(targets.shape[1]):
    nonzero = np.abs(targets[train_pos, j])
    nonzero = nonzero[nonzero > 1e-10]
    scales.append(max(float(np.percentile(nonzero, 75)), 1e-4) if len(nonzero) else 1.0)
scales = np.asarray(scales, dtype=np.float32)
y_scaled = targets / scales
row_weights = 1.0 + 3.0 * np.any(np.abs(targets[train_pos]) > 1e-10, axis=1).astype(np.float32)
probe_models, layer_rows = {}, []
for layer_i in range(N_LAYERS):
    x_train = features[train_pos, layer_i]
    mu = x_train.mean(axis=0)
    sd = np.maximum(x_train.std(axis=0), 1e-3)
    x_fit = (x_train - mu) / sd
    probe = Ridge(alpha=100.0, solver='lsqr', max_iter=1000)
    probe.fit(x_fit, y_scaled[train_pos], sample_weight=row_weights)
    pred = probe.predict((features[valid_pos, layer_i] - mu) / sd)
    truth = y_scaled[valid_pos]
    changed = np.abs(targets[valid_pos]) > 1e-10
    entry_weights = np.where(changed, 4.0, 1.0)
    weighted_mae = float(np.sum(entry_weights * np.abs(pred - truth)) / entry_weights.sum())
    zero_mae = float(np.sum(entry_weights * np.abs(truth)) / entry_weights.sum())
    sign_accuracy = float(np.mean(np.sign(pred[changed]) == np.sign(truth[changed]))) if changed.any() else float('nan')
    change_f1 = float(f1_score(changed.ravel(), (np.abs(pred) > 0.1).ravel(), zero_division=0))
    layer_rows.append({'layer': layer_i, 'weighted_mae': weighted_mae,
                       'zero_predictor_mae': zero_mae, 'changed_sign_accuracy': sign_accuracy,
                       'change_f1': change_f1})
    probe_models[layer_i] = (probe, mu, sd)
layer_table = pd.DataFrame(layer_rows).sort_values('weighted_mae').reset_index(drop=True)
SELECTED_LAYERS = layer_table.head(N_SELECTED_LAYERS)['layer'].astype(int).tolist()
print('Top decoder layers:', SELECTED_LAYERS)
display(layer_table.head(10))
layer_table.to_csv(RUN_ROOT / 'layer_screen.csv', index=False)''')

md(r'''## 5. Derive the two vector-probe steering directions

For each selected layer, the trained probe is `predicted_delta = h @ W + b`, with **W shaped `[hidden_width, 14]`**. The mean direction is `mean(W, axis=1)`. The learned direction is `W @ softmax(theta)`, where 14 nonnegative weights sum to one. Backprop trains `theta` on changed actions in the separate calibration block to predict their actual mean immediate change across attributes that changed during probe training. Attributes with no observed training effect receive zero learned weight. The intervention uses the **positive** direction, since each probe output is signed so higher means improvement. Both directions are L2-normalized before strength calibration. A small threshold, chosen on validation, sets near-zero probe predictions to exact zero for reporting; it does not change W or the steering direction.''')
code(r'''directions = {'mean': {}, 'learned': {}}
learned_attribute_weights = {}
probe_reporting = {}
for layer_i in SELECTED_LAYERS:
    probe, mu, sd = probe_models[layer_i]
    # Undo input and target scaling so W maps native activations to raw 0–1 vector changes.
    W = (probe.coef_.T / sd[:, None]) * scales[None, :]
    bias = (probe.intercept_ - (mu / sd) @ probe.coef_.T) * scales
    pred_valid = features[valid_pos, layer_i] @ W + bias
    thresholds = np.zeros(len(VECTOR_NAMES), dtype=np.float32)
    for j in range(len(VECTOR_NAMES)):
        true = targets[valid_pos, j]
        estimate = pred_valid[:, j]
        candidates = np.linspace(0.0, 0.5 * scales[j], 21)
        penalties = []
        for threshold in candidates:
            sparse = np.where(np.abs(estimate) <= threshold, 0.0, estimate)
            weight = np.where(np.abs(true) > 1e-10, 4.0, 1.0)
            penalties.append(float(np.mean(weight * np.abs(sparse - true))))
        thresholds[j] = candidates[int(np.argmin(penalties))]
    sparse_valid = np.where(np.abs(pred_valid) <= thresholds[None, :], 0.0, pred_valid)
    unchanged = np.abs(targets[valid_pos]) <= 1e-10
    changed = ~unchanged
    probe_reporting[layer_i] = {
        'unchanged_exact_zero_rate': float(np.mean(sparse_valid[unchanged] == 0.0)),
        'changed_detection_recall': float(np.mean(sparse_valid[changed] != 0.0)),
        'thresholds': thresholds.tolist(),
    }
    mean_direction = W.mean(axis=1)
    active = np.any(np.abs(targets[train_pos]) > 1e-10, axis=0)
    if not active.any():
        raise RuntimeError('No hive attributes changed in probe training data')
    predicted_cal = (features[weight_pos, layer_i] @ W + bias)[:, active]
    actual_cal = targets[weight_pos][:, active]
    changed_rows = np.any(np.abs(actual_cal) > 1e-10, axis=1)
    if changed_rows.sum() < 3:
        raise RuntimeError('Too few changed actions to learn attribute weights')
    actual_score_delta = actual_cal[changed_rows].mean(axis=1)
    normalizer = max(float(np.quantile(np.abs(actual_score_delta), 0.9)), 1e-5)
    P = torch.tensor(predicted_cal[changed_rows] / normalizer, dtype=torch.float32)
    Y = torch.tensor(actual_score_delta / normalizer, dtype=torch.float32)
    theta = torch.nn.Parameter(torch.zeros(int(active.sum()), dtype=torch.float32))
    optimizer = torch.optim.Adam([theta], lr=0.05)
    for step in range(300):
        optimizer.zero_grad()
        active_weights = torch.softmax(theta, dim=0)
        loss = F.smooth_l1_loss(P @ active_weights, Y) + 0.01 * ((active_weights - 1 / int(active.sum())) ** 2).sum()
        loss.backward()
        optimizer.step()
    weights = np.zeros(len(VECTOR_NAMES), dtype=np.float32)
    weights[active] = torch.softmax(theta.detach(), dim=0).cpu().numpy()
    learned_direction = W @ weights
    for method, raw in [('mean', mean_direction), ('learned', learned_direction)]:
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm < 1e-12:
            raise RuntimeError(f'Probe at layer {layer_i} has no usable {method} direction')
        directions[method][layer_i] = (raw / norm).astype(np.float32)
    learned_attribute_weights[layer_i] = weights.tolist()
    np.savez(RUN_ROOT / f'probe_layer_{layer_i}.npz', W=W, bias=bias, feature_mean=mu,
             feature_std=sd, target_scales=scales, thresholds=thresholds,
             mean_direction=directions['mean'][layer_i],
             learned_direction=directions['learned'][layer_i], learned_weights=weights)
    print(f'Layer {layer_i}: learned weights', {name: round(float(v), 3) for name, v in zip(VECTOR_NAMES, weights)})
    print('  exact-zero rate on truly unchanged entries:', round(probe_reporting[layer_i]['unchanged_exact_zero_rate'], 3),
          'changed-entry recall:', round(probe_reporting[layer_i]['changed_detection_recall'], 3),
          'direction cosine:', round(float(np.dot(directions['mean'][layer_i], directions['learned'][layer_i])), 4))

# A strength fraction is a fraction of the median hidden-state L2 norm, split over selected layers.
activation_norms = {layer_i: float(np.median(np.linalg.norm(features[train_pos, layer_i], axis=1)))
                    for layer_i in SELECTED_LAYERS}
assert all(np.isfinite(x) and x > 0 for x in activation_norms.values())''')

md(r'''## 6. Intervention mechanism and strength calibration

A forward hook adds the fixed direction to **every token activation** at each selected decoder layer during greedy generation. The hook is removed after each batch. If a model response is malformed or violates the action schema, its raw proposal is journalled and the simulator records a rejected action with an unchanged immediate vector; no replacement decision is invented. Strengths are fractions of the training activation norm and are selected on a short, separate world seed. Both methods use the same strength grid and the same calibration world. Calibration chooses the highest final equal-weight hive score among runs that complete and retain valid JSON actions. No hive vector or score is passed to the agents during generation.''')
code(r'''class SteeredBeePolicy(BeeLocalQwenPolicy):
    def __init__(self, model, tokenizer, layer_directions, fraction, method):
        super().__init__(model, tokenizer, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE)
        self.layer_directions = layer_directions
        self.fraction = float(fraction)
        self.method = method
        self.label = f'local_qwen35_probe_{method}_fraction_{fraction}'
        self.hook_calls = 0
    @staticmethod
    def _proposal_from_text(text):
        stripped = text.lstrip()
        try:
            proposal, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError as exc:
            return {'raw_model_output': text, 'parse_error': f'{exc.msg} at {exc.lineno}:{exc.colno}'}
        if not isinstance(proposal, dict) or stripped[end:].strip():
            return {'raw_model_output': text, 'parse_error': 'Non-object JSON or extra response text'}
        return proposal  # Preserve schema-invalid JSON verbatim for the simulator's rejected-action audit.
    def _decide_allow_invalid(self, observations):
        self.last_response_id = None
        self.last_batch_size = len(observations)
        self.last_sub_batch_sizes = []
        texts, input_counts, output_counts = [], [], []
        for start in range(0, len(observations), self.batch_size):
            generated = self._generate_with_oom_split(observations[start:start + self.batch_size])
            texts.extend(generated[0])
            input_counts.extend(int(v) for v in generated[1])
            output_counts.extend(int(v) for v in generated[2])
        proposals = [self._proposal_from_text(text) for text in texts]
        self.last_usage_by_decision = [
            {'input_tokens': a, 'output_tokens': b, 'total_tokens': a + b}
            for a, b in zip(input_counts, output_counts)
        ]
        self.last_usage = {key: sum(item[key] for item in self.last_usage_by_decision)
                           for key in ('input_tokens', 'output_tokens', 'total_tokens')}
        return proposals
    def decide_batch(self, observations):
        handles = []
        try:
            for layer_i, unit_direction in self.layer_directions.items():
                shift_size = self.fraction * activation_norms[layer_i] / math.sqrt(len(self.layer_directions))
                cached = {}
                def steer(_module, _inputs, output, unit_direction=unit_direction,
                          shift_size=shift_size, cached=cached):
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                        raise RuntimeError('Steering hook received an unexpected layer output')
                    key = (str(hidden.device), hidden.dtype)
                    if key not in cached:
                        cached[key] = torch.as_tensor(unit_direction, device=hidden.device,
                                                      dtype=hidden.dtype).view(1, 1, -1) * shift_size
                    if hidden.shape[-1] != cached[key].shape[-1]:
                        raise RuntimeError('Steering direction and decoder width differ')
                    self.hook_calls += 1
                    steered = hidden + cached[key]
                    if isinstance(output, tuple): return (steered, *output[1:])
                    if isinstance(output, list): return [steered, *output[1:]]
                    return steered
                handles.append(DECODER_LAYERS[layer_i].register_forward_hook(steer))
            result = self._decide_allow_invalid(observations)
            if self.layer_directions and self.hook_calls == 0:
                raise RuntimeError('Steering hook did not execute during generation')
            return result
        finally:
            for handle in handles: handle.remove()

def config_for(seed, hours):
    data = dict(SOURCE_META['calibration']['config'])
    data['seed'] = int(seed)
    data['duration_hours'] = float(hours)
    data['max_actions'] = int(data['n_bees'] * math.ceil(hours * 60 / data['round_minutes']))
    return Config(**data)

def run_arm(name, seed, hours, method=None, fraction=0.0):
    world = World(config_for(seed, hours))
    policy = SteeredBeePolicy(model, processor,
                              directions[method] if method is not None else {},
                              fraction, method or 'baseline')
    output_dir = RUN_ROOT / name
    simulation = Simulation(world, policy, output_dir=output_dir)
    transcript = simulation.run(progress_every=250, decision_batch_size=BATCH_SIZE)
    if method is not None and policy.hook_calls == 0:
        raise RuntimeError(f'Intervention {method} never reached its selected layers')
    initial, final = simulation.baseline, transcript[-1]['vector']
    outcomes = Counter(event['outcome']['status'] for event in simulation.audit)
    summary = {
        'name': name, 'method': method or 'baseline', 'fraction': fraction,
        'seed': seed, 'duration_hours': hours, 'actions': len(transcript),
        'stop_reason': json.loads((output_dir / 'metadata.json').read_text())['stop_reason'],
        'hook_calls': getattr(policy, 'hook_calls', 0),
        'applied': outcomes['applied'], 'rejected': outcomes['rejected'],
        'schema_invalid': sum(validate_action(event['action']) is not None for event in simulation.audit),
        'initial_vector': initial, 'final_vector': final,
        'initial_score': weighted_research_score(initial, [1.0] * len(VECTOR_NAMES)),
        'final_score': weighted_research_score(final, [1.0] * len(VECTOR_NAMES)),
        'output_dir': str(output_dir),
    }
    print(name, 'actions:', len(transcript), 'applied:', outcomes['applied'],
          'rejected:', outcomes['rejected'], 'schema invalid:', summary['schema_invalid'], 'score:', round(summary['final_score'], 5),
          'hooks:', summary['hook_calls'])
    return summary, transcript

calibration_results = []
short_baseline, _ = run_arm('calibration_baseline', CALIBRATION_SEED, CALIBRATION_HOURS)
for method in ('mean', 'learned'):
    for fraction in STRENGTH_GRID:
        name = f'calibration_{method}_{str(fraction).replace(".", "p")}'
        try:
            result, _ = run_arm(name, CALIBRATION_SEED, CALIBRATION_HOURS, method, fraction)
            calibration_results.append(result)
        except Exception as exc:
            calibration_results.append({'name': name, 'method': method, 'fraction': fraction,
                                        'failed': type(exc).__name__ + ': ' + str(exc)[:300]})
            print(name, 'failed:', type(exc).__name__, str(exc)[:300])
chosen_strengths = {}
for method in ('mean', 'learned'):
    completed = [r for r in calibration_results if r['method'] == method and 'final_score' in r]
    if not completed:
        raise RuntimeError(f'All calibration strengths failed for {method}; inspect saved partial runs.')
    best = max(completed, key=lambda r: (r['final_score'], r['applied']))
    chosen_strengths[method] = best['fraction']
print('Short baseline score:', short_baseline['final_score'])
print('Selected strengths:', chosen_strengths)''')

md(r'''## 7. Three matched full-system runs

Each arm creates a fresh world with the same evaluation seed, same 50 bees, same physical configuration, same six-hour horizon, same batch size, and the same frozen-view decision semantics. Runs are ordered baseline, mean intervention, learned-weight intervention. Every action attempt writes one timestamped 14-component vector to `vectors.jsonl` and its outcome to `actions.jsonl`.''')
code(r'''results = {}
transcripts = {}
for name, method, fraction in [
    ('baseline', None, 0.0),
    ('mean_intervention', 'mean', chosen_strengths['mean']),
    ('learned_intervention', 'learned', chosen_strengths['learned']),
]:
    results[name], transcripts[name] = run_arm(name, EVALUATION_SEED,
                                               EVALUATION_HOURS, method, fraction)
assert all(r['actions'] == config_for(EVALUATION_SEED, EVALUATION_HOURS).max_actions
           for r in results.values())
assert all(r['stop_reason'] == 'duration_elapsed' for r in results.values())
assert all(len(row['vector']) == len(VECTOR_NAMES) and
           all(0 <= value <= 1 for value in row['vector'])
           for transcript in transcripts.values() for row in transcript)
assert results['baseline']['initial_vector'] == results['mean_intervention']['initial_vector'] == results['learned_intervention']['initial_vector']
print('All three matched transcripts completed.')''')

md(r'''## 8. Report the state vectors and save artifacts

The table shows all 14 final components; `baseline`, `mean_intervention`, and `learned_intervention` each also have the full time-stamped vector and action audit under the run root. The score uses equal positive weights across all attributes. A single evaluation seed supports a direct controlled comparison here, while broader claims require more seeds.''')
code(r'''comparison = pd.DataFrame({
    'attribute': list(VECTOR_NAMES),
    'initial': results['baseline']['initial_vector'],
    'baseline': results['baseline']['final_vector'],
    'mean_intervention': results['mean_intervention']['final_vector'],
    'learned_intervention': results['learned_intervention']['final_vector'],
})
for name in ('mean_intervention', 'learned_intervention'):
    comparison[name + '_minus_baseline'] = comparison[name] - comparison['baseline']
display(comparison)
score_table = pd.DataFrame([
    {key: result[key] for key in ('name', 'actions', 'applied', 'rejected', 'initial_score',
                                  'final_score', 'fraction', 'hook_calls', 'schema_invalid')}
    for result in results.values()
])
display(score_table)
comparison.to_csv(RUN_ROOT / 'final_vector_comparison.csv', index=False)
score_table.to_csv(RUN_ROOT / 'run_summary.csv', index=False)
report = {
    'source_reference': 'https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes',
    'vector_names': list(VECTOR_NAMES), 'source_events': len(SOURCE_AUDIT),
    'probe_examples': len(indices), 'split_sizes': {'train': len(train_pos), 'validation': len(valid_pos),
                                                     'weight_calibration': len(weight_pos)},
    'selected_layers': SELECTED_LAYERS,
    'selected_layer_metrics': layer_table[layer_table.layer.isin(SELECTED_LAYERS)].to_dict('records'),
    'probe_zero_reporting': probe_reporting,
    'learned_attribute_weights': learned_attribute_weights,
    'strength_grid': STRENGTH_GRID, 'strength_calibration': calibration_results,
    'chosen_strengths': chosen_strengths, 'evaluation_seed': EVALUATION_SEED,
    'results': results,
}
(RUN_ROOT / 'experiment_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
fig, ax = plt.subplots(figsize=(11, 5))
for name, transcript in transcripts.items():
    x = [datetime.fromisoformat(row['timestamp']) for row in transcript]
    y = [weighted_research_score(row['vector'], [1.0] * len(VECTOR_NAMES)) for row in transcript]
    ax.plot(x, y, label=name, linewidth=1.6)
ax.set(xlabel='Simulated time', ylabel='Equal-weight hive score', title='Matched beehive trajectories')
ax.legend(); ax.grid(alpha=0.25); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(RUN_ROOT / 'score_trajectories.png', dpi=160)
plt.show()
print('Experiment files:', RUN_ROOT)
for filename in ('experiment_report.json', 'final_vector_comparison.csv', 'run_summary.csv', 'score_trajectories.png'):
    display(FileLink(str(RUN_ROOT / filename)))
for name in results:
    print(name, 'vectors:', RUN_ROOT / name / 'vectors.jsonl',
          'actions:', RUN_ROOT / name / 'actions.jsonl')''')

md(r'''## Interpretation limits

The probe predicts logged **immediate** action effects, including zero for rejected or inert actions; the three final vectors include both actions and background drift. A linear probe may find correlations that do not yield useful steering. Calibration uses a separate world seed but the full comparison still has one evaluation seed. The learned attribute weights optimize how probe outputs estimate equal-weight action benefit; they are not learned through simulator gradients. Direction strength and layer choice are preselected before the final comparison. The model is a synthetic research system, not a validated biological model.''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'beehive_experiment': {'model': 'Qwen3.5-0.8B LoRA', 'probe_outputs': 14,
                            'steering_methods': ['mean', 'learned_weighted_mean'],
                            'batch_size': 16},
})
out = Path('/home/faithful777/apart_swarm_alignment/beehive_vector_probe_steering.ipynb')
nbf.write(nb, out)
print(out, 'cells', len(cells))
