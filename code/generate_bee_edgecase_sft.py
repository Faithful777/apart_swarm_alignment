"""Create locally feasible bee decision SFT rows from exact prior rollouts."""
from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path

from beehive_sim import Config, Simulation, SYSTEM_PROMPT, World
from bee_feasibility import feasible_actions, teacher_action, physical_signature

ROOT = Path(__file__).resolve().parent
SEED42 = ROOT / 'kaggle_deploy/beehive_qwen35/full_run_output/runs/qwen_local_20261103T135246_463284Z'
SEED43 = ROOT / 'kaggle_deploy/vector_probe/full_output/vector_probe_20261104T144424_251913Z/baseline'
OUTPUT = ROOT / 'data/bee_edgecase_sft'
MENU_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    '\nYour observation includes feasible_actions, a local list of executable actions. '
    'Choose exactly one item from that list and copy its six JSON fields exactly. '
    'The list does not contain the colony-wide hive vector or scores. '
    'All bees in a decision batch see one frozen state.\n'
)

class Replay:
    label = 'edgecase_source_replay'
    def __init__(self, rows):
        self.rows = rows
        self.index = 0
        self.observations = []
    def decide_batch(self, observations):
        batch = self.rows[self.index:self.index + len(observations)]
        assert [o['self']['id'] for o in observations] == [r['bee_id'] for r in batch]
        self.observations.extend(copy.deepcopy(observations))
        self.index += len(batch)
        return [r['action'] for r in batch]

def reconstruct(rows_path: Path, vectors_path: Path, config: Config):
    rows = [json.loads(line) for line in rows_path.open()]
    expected = [json.loads(line) for line in vectors_path.open()]
    policy = Replay(rows)
    replayed = Simulation(World(config), policy).run(decision_batch_size=16)
    if len(rows) != len(expected) or len(replayed) != len(expected):
        raise RuntimeError('Replay length mismatch')
    if any(a['vector'] != b['vector'] for a, b in zip(replayed, expected)):
        raise RuntimeError('Replay differs from saved trajectory')
    return rows, policy.observations

def examples(rows, observations, selected):
    output = []
    for i in selected:
        source = rows[i]
        observation = observations[i]
        menu = feasible_actions(observation)
        target = teacher_action(observation, menu, source['action'])
        assert target in menu
        enriched = copy.deepcopy(observation)
        enriched['feasible_actions'] = menu
        output.append({
            'prompt': [
                {'role': 'system', 'content': MENU_SYSTEM_PROMPT},
                {'role': 'user', 'content': json.dumps(enriched, ensure_ascii=False, separators=(',', ':'), allow_nan=False)},
            ],
            'completion': [
                {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False, separators=(',', ':'), allow_nan=False)},
            ],
        })
    return output

def main():
    metadata = json.loads((SEED42 / 'metadata.json').read_text())
    cfg42 = Config(**metadata['calibration']['config'])
    cfg43_data = dict(metadata['calibration']['config'])
    cfg43_data['seed'] = 43
    rows42, observations42 = reconstruct(SEED42 / 'actions.jsonl', SEED42 / 'vectors.jsonl', cfg42)
    rows43, observations43 = reconstruct(SEED43 / 'actions.jsonl', SEED43 / 'vectors.jsonl', Config(**cfg43_data))
    rng = random.Random(5407)
    failures42 = [i for i, row in enumerate(rows42) if row['outcome']['status'] == 'rejected']
    applied42 = [i for i, row in enumerate(rows42) if row['outcome']['status'] == 'applied']
    failures43 = [i for i, row in enumerate(rows43) if row['outcome']['status'] == 'rejected']
    applied43 = [i for i, row in enumerate(rows43) if row['outcome']['status'] == 'applied']
    train_ids = failures42 + rng.sample(applied42, min(400, len(applied42)))
    validation_ids = rng.sample(failures43, min(100, len(failures43))) + rng.sample(applied43, min(100, len(applied43)))
    rng.shuffle(train_ids)
    rng.shuffle(validation_ids)
    train = examples(rows42, observations42, train_ids)
    # A separate feasible teacher rollout supplies task diversity that the
    # rejected-action source lacks (especially forage, brood care, and guard).
    class Demonstrator:
        label = 'menu_teacher_diversity_source'
        def __init__(self): self.observations = []
        def decide_batch(self, observations):
            self.observations.extend(copy.deepcopy(observations))
            return [teacher_action(obs, feasible_actions(obs)) for obs in observations]
    cfg44_data = dict(metadata['calibration']['config'])
    cfg44_data['seed'] = 44
    demonstrator = Demonstrator()
    demo_sim = Simulation(World(Config(**cfg44_data)), demonstrator)
    demo_sim.run(decision_batch_size=16)
    groups = {}
    for index, row in enumerate(demo_sim.audit):
        groups.setdefault(row['action']['action'], []).append(index)
    selected_demo = []
    for action_name in sorted(groups):
        choices = groups[action_name][:]
        rng.shuffle(choices)
        selected_demo.extend(choices[:min(60, len(choices))])
    remainder = [i for i in range(len(demo_sim.audit)) if i not in set(selected_demo)]
    rng.shuffle(remainder)
    selected_demo.extend(remainder[:max(0, 500 - len(selected_demo))])
    rng.shuffle(selected_demo)
    train.extend(examples(demo_sim.audit, demonstrator.observations, selected_demo))
    rng.shuffle(train)
    validation = examples(rows43, observations43, validation_ids)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, records in [('train', train), ('validation', validation)]:
        with (OUTPUT / f'edgecase_{name}.jsonl').open('w', encoding='utf-8') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    def counts(records):
        return dict(Counter(json.loads(row['completion'][0]['content'])['action'] for row in records))
    manifest = {
        'dataset': 'bee_local_feasible_menu_sft',
        'source_seeds': {'failed_action_train': 42, 'feasible_teacher_train': 44, 'validation': 43},
        'source_rows': {'failed_action_train': len(rows42), 'feasible_teacher_train': len(demo_sim.audit), 'validation': len(rows43)},
        'rows': {'train': len(train), 'validation': len(validation)},
        'source_rejected_in_training': len(failures42),
        'feasible_teacher_rows_in_training': len(selected_demo),
        'action_counts': {'train': counts(train), 'validation': counts(validation)},
        'menu_size_limit': 12,
        'target_rule': 'keep applicable model task; otherwise choose a locally feasible role-aware demonstration',
        'validation_note': 'one independent world seed, with 100 rejected and 100 applied source cases',
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
