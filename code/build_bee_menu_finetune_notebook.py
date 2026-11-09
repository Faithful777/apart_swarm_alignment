from pathlib import Path
import json
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Continue Qwen3.5-0.8B bee LoRA on locally feasible actions

This notebook trains on exact simulator observations enriched with a bounded menu of locally executable actions. Targets are heuristic demonstrations, not optimal labels. Seed 42 provides failure-heavy cases and seed 44 contributes stratified feasible-teacher cases; seed 43 provides held-out validation. The global 14-component hive state never appears in prompts. A separate simulator experiment evaluates whether this training improves colony outcomes.''')
code('''%pip install --upgrade --quiet unsloth unsloth_zoo "transformers>=5.0.0" "trl>=0.27.0" datasets accelerate peft''')
code('''from pathlib import Path
from datetime import datetime, timezone
import json, math, random, sys, zipfile
import numpy as np
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

ROOT = Path('/kaggle/input')
train_path = next(ROOT.rglob('edgecase_train.jsonl'))
valid_path = next(ROOT.rglob('edgecase_validation.jsonl'))
archive = next(ROOT.rglob('beehive_qwen35_runtime_bundle.zip'), None)
work = Path('/kaggle/working/bee_menu_finetune')
work.mkdir(parents=True, exist_ok=True)
if archive is not None:
    with zipfile.ZipFile(archive) as z:
        z.extractall(work / 'runtime')
    adapter = work / 'runtime/adapter'
else:
    adapter = next(p.parent for p in ROOT.rglob('adapter_config.json')
                   if (p.parent / 'adapter_model.safetensors').is_file())
train_rows = [json.loads(line) for line in train_path.open()]
valid_rows = [json.loads(line) for line in valid_path.open()]
assert len(train_rows) >= 985 and len(valid_rows) == 200
assert all('feasible_actions' in json.loads(row['prompt'][1]['content']) for row in train_rows + valid_rows)
print('Rows:', len(train_rows), len(valid_rows), 'adapter:', adapter)''')
code('''SMOKE = False
SEED = 719
MAX_LENGTH = 4096
TRAIN_ROWS = 48 if SMOKE else len(train_rows)
VALID_ROWS = 16 if SMOKE else len(valid_rows)
random.Random(SEED).shuffle(train_rows)
random.Random(SEED + 1).shuffle(valid_rows)
train_rows = train_rows[:TRAIN_ROWS]
valid_rows = valid_rows[:VALID_ROWS]
dataset_train = Dataset.from_list(train_rows)
dataset_valid = Dataset.from_list(valid_rows)
torch.manual_seed(SEED)
print('Mode:', 'smoke' if SMOKE else 'full')''')
code('''model, processor = FastLanguageModel.from_pretrained(
    model_name=str(adapter), max_seq_length=MAX_LENGTH, dtype=None,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
)
FastLanguageModel.for_training(model)
# Existing PEFT adapter weights should be trainable. Stop if a library update freezes them.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable == 0:
    for name, p in model.named_parameters():
        if 'lora_' in name: p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 1_000_000, f'Existing LoRA adapter is frozen: {trainable}'
print('Trainable adapter parameters:', trainable)''')
code('''args = SFTConfig(
    output_dir=str(work / 'checkpoints'), max_length=MAX_LENGTH,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, num_train_epochs=1,
    max_steps=3 if SMOKE else -1, learning_rate=5e-5,
    lr_scheduler_type='cosine', warmup_ratio=0.03,
    weight_decay=0.01, optim='adamw_8bit', logging_steps=10,
    eval_strategy='no' if SMOKE else 'epoch',
    save_strategy='no', completion_only_loss=True,
    bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True, max_grad_norm=1.0,
    seed=SEED, data_seed=SEED, dataset_num_proc=1, report_to='none',
)
trainer = SFTTrainer(model=model, processing_class=processor,
    train_dataset=dataset_train, eval_dataset=dataset_valid, args=args)
print('Optimizer steps:', 3 if SMOKE else math.ceil(TRAIN_ROWS / 8))
result = trainer.train()
print(result)
metrics = trainer.evaluate()
print('Held-out metrics:', metrics)''')
code('''FastLanguageModel.for_inference(model)
saved_adapter = work / 'adapter'
model.save_pretrained(str(saved_adapter))
processor.save_pretrained(str(saved_adapter))
report = {'mode': 'smoke' if SMOKE else 'full', 'train_seeds': [42,44],
    'validation_seed': 43, 'train_rows': TRAIN_ROWS, 'validation_rows': VALID_ROWS,
    'max_length': MAX_LENGTH, 'epochs': 1, 'learning_rate': 5e-5,
    'trainable_parameters': trainable, 'training_loss': result.training_loss,
    'validation_metrics': metrics, 'adapter': str(saved_adapter)}
(work / 'training_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))''')

out = Path('bee_menu_finetune.ipynb')
nbf.write(nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
    'language_info': {'name':'python'}}), out)
meta = {'id':'faithfulchukwunwogor/beehive-qwen35-feasible-menu-finetune',
        'title':'Beehive Qwen35 Feasible Menu Finetune',
        'code_file':out.name,'language':'python','kernel_type':'notebook',
        'is_private':'true','enable_gpu':'true','enable_internet':'true',
        'machine_shape':'','dataset_sources':['faithfulchukwunwogor/beehive-qwen35-runtime'],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]}
folder = Path('kaggle_deploy/menu_finetune/kernel')
folder.mkdir(parents=True, exist_ok=True)
nbf.write(nbf.read(out, as_version=4), folder/out.name)
(folder/'kernel-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
smoke = Path('kaggle_deploy/menu_finetune/kernel_smoke')
smoke.mkdir(parents=True, exist_ok=True)
smoke_nb = nbf.read(out, as_version=4)
for cell in smoke_nb.cells:
    if cell.cell_type == 'code' and 'SMOKE = False' in cell.source:
        cell.source = cell.source.replace('SMOKE = False', 'SMOKE = True')
nbf.write(smoke_nb, smoke / 'bee_menu_finetune.ipynb')
smoke_meta = dict(meta)
smoke_meta['id'] = 'faithfulchukwunwogor/beehive-qwen35-feasible-menu-finetune-smoke'
smoke_meta['title'] = 'Beehive Qwen35 Feasible Menu Finetune Smoke'
(smoke/'kernel-metadata.json').write_text(json.dumps(smoke_meta, indent=2)+'\n')
print(out, folder)
