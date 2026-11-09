"""Build the bounded per-attribute bee probe steering experiment."""
from pathlib import Path
import json
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md('''# Individual and combined hive-attribute steering

The saved 14-output probes from the completed follow-up run are reused without
retraining. Six columns have nonzero labels. This notebook tests each normalized
column alone, an equal combination of normalized columns, the earlier raw-column
sum, and a combination whose nonnegative weights are learned from paired
calibration runs. A single fixed intervention strength and position are used
throughout, so direction construction is the main comparison.

The 14-component hive vector remains researcher-only. The bees receive only
their local observations and feasible-action menus. A batch of 16 decisions
sees one frozen world state; actions execute in order, with a vector snapshot
after each action. The main evaluation uses new seeds 70–73. This is a synthetic
simulator experiment, not a biological validation.''')

code('''%pip install --upgrade --quiet unsloth unsloth_zoo "transformers>=5.0.0" "trl>=0.27.0" datasets accelerate peft scikit-learn''')

code('''from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, math, sys, time, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unsloth import FastLanguageModel

INPUT_ROOT=Path('/kaggle/input')
runtime_archive=next(INPUT_ROOT.rglob('beehive_qwen35_runtime_bundle.zip'),None)
menu_module=next(INPUT_ROOT.rglob('bee_menu_qwen_policy.py'))
probe_files={layer:next(INPUT_ROOT.rglob(f'probe_layer_{layer}.npz')) for layer in (19,20)}
adapter_files=[p for p in INPUT_ROOT.rglob('adapter_model.safetensors')
               if 'menu' in str(p.parent).lower()]
if not adapter_files: raise FileNotFoundError('Attach the private continued menu-LoRA adapter dataset')
ADAPTER_DIR=adapter_files[0].parent
RUN_ROOT=Path('/kaggle/working')/('bee_columns_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
RUN_ROOT.mkdir(parents=True,exist_ok=False)
if runtime_archive is not None:
    RUNTIME_DIR=RUN_ROOT/'runtime'
    with zipfile.ZipFile(runtime_archive) as z: z.extractall(RUNTIME_DIR)
else:
    RUNTIME_DIR=next(INPUT_ROOT.rglob('beehive_sim.py')).parent
sys.path[:0]=[str(menu_module.parent),str(RUNTIME_DIR)]
from beehive_sim import Config, World, Simulation, VECTOR_NAMES, weighted_research_score, validate_action
from bee_menu_qwen_policy import BeeMenuQwenPolicy
SOURCE_META=json.loads(next(INPUT_ROOT.rglob('probe_source_metadata.json')).read_text())
SOURCE_CONFIG=dict(SOURCE_META['calibration']['config'])
assert len(VECTOR_NAMES)==14
print('Adapter',ADAPTER_DIR,'probe files',probe_files,'output',RUN_ROOT)''')

code('''SMOKE=False
BATCH_SIZE=16
MAX_NEW_TOKENS=160
CAL_SEEDS=(54,) if SMOKE else (54,55)
FINAL_SEEDS=(70,) if SMOKE else (70,71,72,73)
CAL_HOURS=0.10 if SMOKE else 0.50
FINAL_HOURS=0.10 if SMOKE else 1.00
FRACTION=0.03
POSITION='all'
SELECTED_LAYERS=(19,20)

model,processor=FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),max_seq_length=4096,dtype=None,
    load_in_4bit=False,load_in_16bit=True,full_finetuning=False)
FastLanguageModel.for_inference(model)
model.eval()

def decoder_candidates(model):
    found=[]
    for name,module in model.named_modules():
        if not isinstance(module,nn.ModuleList) or len(module)<8: continue
        if not any(hasattr(module[0],field) for field in ('self_attn','linear_attn','mlp')): continue
        score=len(module)+100*('language_model' in name)+50*('text' in name)
        score-=100*any(word in name for word in ('visual','vision','image'))
        found.append((score,name,module))
    return sorted(found,reverse=True)
_,LAYER_PATH,DECODER_LAYERS=decoder_candidates(model)[0]
assert len(DECODER_LAYERS)>max(SELECTED_LAYERS)

W={layer:np.load(probe_files[layer])['W'].astype(np.float32) for layer in SELECTED_LAYERS}
norms={layer:np.linalg.norm(matrix,axis=0) for layer,matrix in W.items()}
active=np.flatnonzero(np.all(np.stack([norms[layer]>1e-10 for layer in SELECTED_LAYERS]),axis=0))
assert active.tolist()==[2,3,8,10,11,12],active
ACTIVE=active[:2] if SMOKE else active
U={layer:W[layer][:,ACTIVE]/norms[layer][ACTIVE] for layer in SELECTED_LAYERS}
assert all(np.allclose(np.linalg.norm(U[layer],axis=0),1,atol=1e-5) for layer in SELECTED_LAYERS)

def normalized(vector):
    vector=np.asarray(vector,dtype=np.float32)
    size=float(np.linalg.norm(vector))
    assert np.isfinite(size) and size>1e-10
    return vector/size

def build_directions(weights):
    weights=np.asarray(weights,dtype=np.float32)
    assert weights.shape==(len(ACTIVE),) and np.all(weights>=0) and weights.sum()>0
    return {layer:normalized(U[layer]@weights) for layer in SELECTED_LAYERS}

DIRS={f'attribute_{j}':{layer:U[layer][:,k].copy() for layer in SELECTED_LAYERS}
      for k,j in enumerate(ACTIVE)}
DIRS['equal_normalized']=build_directions(np.ones(len(ACTIVE)))
DIRS['raw_sum']={layer:normalized(W[layer][:,ACTIVE].sum(axis=1)) for layer in SELECTED_LAYERS}
for layer in SELECTED_LAYERS:
    cosine=float(DIRS['equal_normalized'][layer]@DIRS['raw_sum'][layer])
    print('Layer',layer,'active columns',[(int(j),VECTOR_NAMES[j],round(float(norms[layer][j]),5)) for j in ACTIVE],
          'equal-vs-raw cosine',round(cosine,4))
print('Decoder',LAYER_PATH,'layers',len(DECODER_LAYERS),'smoke',SMOKE)''')

md('''## 1. Fixed-size, paired interventions

Each column is normalized separately. All arms use the same positive strength
(3% of a fixed layer activation norm), all-token hook position, two layers,
50 bees, and decision batches of at most 16. The baseline calibration run
measures the layer norm once; the same values are used for every arm. Thus the
raw, equal-normalized, and individually steered arms have the same perturbation
budget per layer. A change in action is counted against the matching seed's
baseline, not against a different world.''')

code('''def config_for(seed,hours):
    data=dict(SOURCE_CONFIG)
    data.update(seed=int(seed),duration_hours=float(hours),
                max_actions=int(data['n_bees']*math.ceil(hours*60/data['round_minutes'])),
                model='Qwen3.5-0.8B continued menu LoRA; saved vector probe columns')
    if SMOKE: data['max_actions']=min(data['max_actions'],24)
    return Config(**data)

class NormCapturePolicy(BeeMenuQwenPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.samples={layer:[] for layer in SELECTED_LAYERS}
    def decide_batch(self,observations):
        handles=[]
        try:
            for layer_i in SELECTED_LAYERS:
                def capture(_module,_inputs,output,layer_i=layer_i):
                    if len(self.samples[layer_i])>=256: return
                    hidden=output[0] if isinstance(output,(tuple,list)) else output
                    values=hidden[:,-1,:].float().norm(dim=-1).detach().cpu().tolist()
                    self.samples[layer_i].extend(values[:256-len(self.samples[layer_i])])
                handles.append(DECODER_LAYERS[layer_i].register_forward_hook(capture))
            return super().decide_batch(observations)
        finally:
            for handle in handles: handle.remove()

activation_norms={}
class ColumnSteeredPolicy(BeeMenuQwenPolicy):
    def __init__(self,directions):
        super().__init__(model,processor,MAX_NEW_TOKENS,BATCH_SIZE)
        self.directions=directions
        self.hook_calls=0
        self.label='column_steered_menu'
    def decide_batch(self,observations):
        handles=[]
        try:
            for layer_i,unit in self.directions.items():
                size=FRACTION*activation_norms[layer_i]/math.sqrt(len(self.directions))
                cached={}
                def steer(_module,_inputs,output,unit=unit,size=size,cached=cached):
                    hidden=output[0] if isinstance(output,(tuple,list)) else output
                    key=(str(hidden.device),hidden.dtype)
                    if key not in cached:
                        cached[key]=torch.as_tensor(unit,device=hidden.device,dtype=hidden.dtype).view(1,1,-1)*size
                    shift=cached[key]
                    assert hidden.shape[-1]==shift.shape[-1]
                    self.hook_calls+=1
                    if POSITION=='response' and hidden.shape[1]>1:
                        altered=hidden.clone(); altered[:,-1:,:]+=shift
                    else:
                        altered=hidden+shift
                    if isinstance(output,tuple): return (altered,*output[1:])
                    if isinstance(output,list): return [altered,*output[1:]]
                    return altered
                handles.append(DECODER_LAYERS[layer_i].register_forward_hook(steer))
            result=super().decide_batch(observations)
            assert self.hook_calls>0
            return result
        finally:
            for handle in handles: handle.remove()

def run_arm(name,seed,hours,directions=None,capture_norm=False):
    world=World(config_for(seed,hours))
    if capture_norm:
        policy=NormCapturePolicy(model,processor,MAX_NEW_TOKENS,BATCH_SIZE)
    elif directions is None:
        policy=BeeMenuQwenPolicy(model,processor,MAX_NEW_TOKENS,BATCH_SIZE)
    else:
        policy=ColumnSteeredPolicy(directions)
    output_dir=RUN_ROOT/name
    sim=Simulation(world,policy,output_dir=output_dir)
    transcript=sim.run(progress_every=200,decision_batch_size=BATCH_SIZE)
    audit=sim.audit
    outcomes=Counter(r['outcome']['status'] for r in audit)
    initial=sim.baseline; final=transcript[-1]['vector']
    immediate=float(sum(np.mean(r['immediate_action_delta']) for r in audit))
    background=float(sum(np.mean(r['background_delta']) for r in audit))
    row={'seed':int(seed),'arm':name.rsplit('_',1)[-1],
         'name':name,'hours':hours,'actions':len(audit),
         'applied':outcomes['applied'],'rejected':outcomes['rejected'],
         'schema_invalid':sum(validate_action(r['action']) is not None for r in audit),
         'initial_vector':initial,'final_vector':final,
         'initial_score':weighted_research_score(initial,[1.0]*14),
         'final_score':weighted_research_score(final,[1.0]*14),
         'immediate_score_change':immediate,'background_score_change':background,
         'hook_calls':getattr(policy,'hook_calls',0),
         'output_dir':str(output_dir)}
    assert len(audit)==config_for(seed,hours).max_actions
    assert abs(row['initial_score']+immediate+background-row['final_score'])<1e-6
    assert all(len(x['vector'])==14 and all(np.isfinite(x['vector'])) and
               all(0<=v<=1 for v in x['vector']) for x in transcript)
    if directions is not None: assert row['hook_calls']>0
    print(name,'score',round(row['final_score'],6),'applied',row['applied'],
          'rejected',row['rejected'],flush=True)
    return row,policy

def action_list(row):
    return [json.loads(line)['action'] for line in
            (Path(row['output_dir'])/'actions.jsonl').open()]

def annotate_changes(row,baseline):
    left,right=action_list(row),action_list(baseline)
    assert len(left)==len(right)
    row['changed_actions_vs_baseline']=sum(a!=b for a,b in zip(left,right))
    row['score_difference_vs_baseline']=row['final_score']-baseline['final_score']
    row['vector_difference_vs_baseline']=(np.asarray(row['final_vector'])-
                                          np.asarray(baseline['final_vector'])).tolist()
    return row''')

md('''## 2. Measure individual and simultaneous effects on calibration seeds

The six individual directions, equal-normalized sum, and raw-column sum are
tested against paired baselines on seeds 54 and 55. Those seeds alone determine
the learned weights. No final-evaluation seed is used for direction selection,
weight learning, or strength tuning.''')

code('''calibration=[]; cal_baselines={}
for seed in CAL_SEEDS:
    baseline,policy=run_arm(f'cal_{seed}_baseline',seed,CAL_HOURS,
                            capture_norm=(seed==CAL_SEEDS[0]))
    if seed==CAL_SEEDS[0]:
        activation_norms.update({layer:float(np.median(policy.samples[layer]))
                                 for layer in SELECTED_LAYERS})
        assert all(np.isfinite(v) and v>0 for v in activation_norms.values())
        print('Fixed activation norms',activation_norms)
    baseline['arm']='baseline'; baseline['changed_actions_vs_baseline']=0
    baseline['score_difference_vs_baseline']=0.0
    baseline['vector_difference_vs_baseline']=[0.0]*14
    cal_baselines[seed]=baseline; calibration.append(baseline)
    for arm,directions in DIRS.items():
        row,_=run_arm(f'cal_{seed}_{arm}',seed,CAL_HOURS,directions)
        row['arm']=arm; calibration.append(annotate_changes(row,baseline))
cal_table=pd.DataFrame([{k:v for k,v in row.items() if k not in ('initial_vector','final_vector','vector_difference_vs_baseline')}
                        for row in calibration])
cal_table.to_csv(RUN_ROOT/'calibration_summary.csv',index=False)
print(cal_table[['seed','arm','final_score','score_difference_vs_baseline','changed_actions_vs_baseline']].to_string(index=False))''')

md('''## 3. Learn a bounded combination from measured hive scores

The measured paired final-score effects of the individual directions are
standardized and passed through a softmax. A 25% uniform component shrinks the
result toward equal weights. This nonnegative, sum-to-one rule was fixed before
the run; it needs no extra model training or large black-box search. It uses
actual simulator outcomes, rather than fitting weights to the probe's own
predictions. The resulting combination is also checked on the calibration
seeds, but its weights are not adjusted after that check.''')

code('''effects=np.asarray([
    np.mean([next(r for r in calibration if r['seed']==seed and r['arm']==f'attribute_{j}')
             ['score_difference_vs_baseline'] for seed in CAL_SEEDS])
    for j in ACTIVE],dtype=np.float64)
effect_scale=max(float(np.std(effects)),1e-5)
logits=2.0*(effects-effects.mean())/effect_scale
soft=np.exp(logits-logits.max()); soft/=soft.sum()
learned_weights=0.25/len(ACTIVE)+0.75*soft
assert np.all(learned_weights>0) and abs(learned_weights.sum()-1)<1e-8
DIRS['learned_combination']=build_directions(learned_weights)
print('Measured individual paired score effects',dict(zip([VECTOR_NAMES[j] for j in ACTIVE],effects.tolist())))
print('Learned weights',dict(zip([VECTOR_NAMES[j] for j in ACTIVE],learned_weights.tolist())))
for seed in CAL_SEEDS:
    row,_=run_arm(f'cal_{seed}_learned_combination',seed,CAL_HOURS,DIRS['learned_combination'])
    row['arm']='learned_combination'
    calibration.append(annotate_changes(row,cal_baselines[seed]))
print('Learned calibration scores',[(r['seed'],r['score_difference_vs_baseline'],r['changed_actions_vs_baseline'])
                                    for r in calibration if r['arm']=='learned_combination'])''')

md('''## 4. Evaluate every direction and combination on four new seeds

All arms run for one simulated hour, with 200 action snapshots each. The final
seeds 70–73 were not used in the prior probe, continuation training, or this
calibration. The final table reports score and each vector component, and keeps
immediate action contributions separate from background time drift. Four seeds
are descriptive, not proof of a biological or broadly general effect.''')

code('''final_runs=[]
for seed in FINAL_SEEDS:
    baseline,_=run_arm(f'final_{seed}_baseline',seed,FINAL_HOURS)
    baseline['arm']='baseline'; baseline['changed_actions_vs_baseline']=0
    baseline['score_difference_vs_baseline']=0.0
    baseline['vector_difference_vs_baseline']=[0.0]*14
    final_runs.append(baseline)
    for arm,directions in DIRS.items():
        row,_=run_arm(f'final_{seed}_{arm}',seed,FINAL_HOURS,directions)
        row['arm']=arm; final_runs.append(annotate_changes(row,baseline))
print('Final arms',len(final_runs),'total actions',sum(r['actions'] for r in final_runs))''')

code('''def percentile_bootstrap(values,draws=10000):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(904)
    means=rng.choice(values,size=(draws,len(values)),replace=True).mean(axis=1)
    return np.quantile(means,[0.025,0.975]).tolist()

summary_rows=[]
for row in final_runs:
    summary_rows.append({k:v for k,v in row.items()
                         if k not in ('initial_vector','final_vector','vector_difference_vs_baseline')})
summary=pd.DataFrame(summary_rows)
summary.to_csv(RUN_ROOT/'final_run_summary.csv',index=False)
vector_rows=[]
for row in final_runs:
    for j,name in enumerate(VECTOR_NAMES):
        vector_rows.append({'seed':row['seed'],'arm':row['arm'],'attribute':name,
                            'initial':row['initial_vector'][j],
                            'final':row['final_vector'][j],
                            'minus_baseline':row['vector_difference_vs_baseline'][j]})
pd.DataFrame(vector_rows).to_csv(RUN_ROOT/'final_vectors.csv',index=False)

paired={}
for arm in DIRS:
    rows=[r for r in final_runs if r['arm']==arm]
    differences=[r['score_difference_vs_baseline'] for r in rows]
    paired[arm]={'score_differences':differences,
                 'mean_score_difference':float(np.mean(differences)),
                 'bootstrap_95pct_interval':percentile_bootstrap(differences),
                 'changed_actions':sum(r['changed_actions_vs_baseline'] for r in rows),
                 'mean_final_vector':np.mean([r['final_vector'] for r in rows],axis=0).tolist(),
                 'mean_vector_difference':np.mean([r['vector_difference_vs_baseline'] for r in rows],axis=0).tolist()}
report={'model':'Qwen3.5-0.8B continued bee-menu LoRA',
        'probe_source':'saved layer 19 and 20 vector probes; no retraining',
        'vector_names':list(VECTOR_NAMES),'active_indices':ACTIVE.tolist(),
        'selected_layers':list(SELECTED_LAYERS),'fraction':FRACTION,'position':POSITION,
        'fixed_activation_norms':activation_norms,
        'calibration_seeds':list(CAL_SEEDS),'calibration_hours':CAL_HOURS,
        'final_seeds':list(FINAL_SEEDS),'final_hours':FINAL_HOURS,'smoke':SMOKE,
        'learned_weights':dict(zip([VECTOR_NAMES[j] for j in ACTIVE],learned_weights.tolist())),
        'individual_calibration_effects':dict(zip([VECTOR_NAMES[j] for j in ACTIVE],effects.tolist())),
        'calibration_runs':calibration,'final_runs':final_runs,'paired':paired}
(RUN_ROOT/'experiment_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(summary[['seed','arm','final_score','score_difference_vs_baseline','changed_actions_vs_baseline']].to_string(index=False))
print('Paired summary',json.dumps({k:{'mean':v['mean_score_difference'],
                                     'interval':v['bootstrap_95pct_interval'],
                                     'changed_actions':v['changed_actions']}
                                    for k,v in paired.items()},indent=2))
print('Artifacts',RUN_ROOT)''')


def write_notebook(path, smoke):
    copied = [nbf.v4.new_markdown_cell(c.source) if c.cell_type == 'markdown'
              else nbf.v4.new_code_cell(c.source) for c in cells]
    if smoke:
        for cell in copied:
            if cell.cell_type == 'code' and 'SMOKE=False' in cell.source:
                cell.source = cell.source.replace('SMOKE=False', 'SMOKE=True', 1)
                break
    nb = nbf.v4.new_notebook(cells=copied, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


write_notebook(ROOT / 'beehive_column_steering.ipynb', False)
write_notebook(ROOT / 'kaggle_deploy/column_steering/kernel_smoke/beehive_column_steering.ipynb', True)
write_notebook(ROOT / 'kaggle_deploy/column_steering/kernel_full/beehive_column_steering.ipynb', False)

base_metadata = {
    'id': 'faithfulchukwunwogor/beehive-probe-column-steering',
    'title': 'Beehive Probe Column Steering',
    'code_file': 'beehive_column_steering.ipynb',
    'language': 'python', 'kernel_type': 'notebook', 'is_private': 'true',
    'enable_gpu': 'true', 'enable_internet': 'true', 'machine_shape': '',
    'dataset_sources': [
        'faithfulchukwunwogor/beehive-qwen35-runtime',
        'faithfulchukwunwogor/beehive-qwen35-menu-adapter',
        'faithfulchukwunwogor/beehive-menu-probe-weights'],
    'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
for stage, slug in [('kernel_smoke', 'beehive-probe-column-steering-smoke'),
                    ('kernel_full', 'beehive-probe-column-steering')]:
    metadata = dict(base_metadata, id='faithfulchukwunwogor/' + slug,
                    title='Beehive Probe Column Steering' + (' Smoke' if stage == 'kernel_smoke' else ''))
    path = ROOT / 'kaggle_deploy/column_steering' / stage / 'kernel-metadata.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
print('Built column-steering Kaggle notebooks and metadata')
