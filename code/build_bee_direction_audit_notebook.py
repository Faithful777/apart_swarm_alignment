"""Build a Kaggle notebook to validate and causally screen the saved bee probe."""
from pathlib import Path
import json
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
base = nbf.read(ROOT / "beehive_mixed_probe_2h.ipynb", as_version=4)
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md("""# Probe direction audit on fresh two-hour worlds

This notebook audits the saved mixed-environment probe from the completed
two-hour run. It computes the overlap of all eight normalized attribute
columns. Fresh seeds 110–113 then run baseline and new equal-normalized
steering for two simulated hours; the saved probe is evaluated separately by
hour and environment. The validation reports when those paired environments
actually diverged. Its dense regression scores are distinct from the earlier
probe's change-mask scores because that mask was not saved with the probe.

For a limited causal screen, seeds 110–111 also provide one frozen 16-bee
decision batch per simulated round. Candidate directions act on the **same
local observations and frozen world** as baseline. Cloned worlds execute the
batch of proposed actions in order. The screen measures changed decisions and
their paired one-batch hive-vector effect. Exact duplicate or opposite probe
columns are represented by signed axes, avoiding repeated GPU runs. The two
best positive candidates have their steering strength checked at 0.015 and
0.045 as well as 0.03. Two entirely new seeds, 114–115, then test up to two
selected directions in complete two-hour hives. If no candidate improves the
one-batch screen, the notebook avoids that costly final stage.

This is a causal test of **actions under intervention**, but probe prediction
and action influence remain different quantities. The one-batch screen is a
selection set; the fresh final seeds are the subsequent hive-outcome test.
""")

cells.append(nbf.v4.new_code_cell(base.cells[1].source))

imports = base.cells[2].source
imports = imports.replace("from collections import Counter", "from collections import Counter, defaultdict")
imports = imports.replace("import copy, json, math, sys, time, zipfile", "import copy, csv, gzip, json, math, sys, time, zipfile")
imports = imports.replace("from sklearn.linear_model import Ridge, LogisticRegression\nfrom sklearn.metrics import f1_score\n", "")
imports = imports.replace(
    "probe_files={layer:next(INPUT_ROOT.rglob(f'probe_layer_{layer}.npz')) for layer in (19,20)}",
    "PROBE_ROOT=next(INPUT_ROOT.rglob('new_probe_layer_14.npz')).parent\n"
    "probe_files={layer:PROBE_ROOT/f'new_probe_layer_{layer}.npz' for layer in (13,14)}\n"
    "PRIOR_REPORT=json.loads((PROBE_ROOT/'experiment_report.json').read_text())",
)
imports = imports.replace("('bee_mixed_probe_'+", "('bee_direction_audit_'+")
imports = imports.replace(
    "from beehive_sim import Config, World, Simulation, VECTOR_NAMES, weighted_research_score, validate_action",
    "from beehive_sim import Config, World, Observer, Simulation, VECTOR_NAMES, weighted_research_score, validate_action",
)
code(imports)

code("""SMOKE=False
BATCH_SIZE=16
MAX_NEW_TOKENS=160
VALIDATION_SEEDS=(110,) if SMOKE else (110,111,112,113)
SCREEN_SEEDS=(110,) if SMOKE else (110,111)
FINAL_SEEDS=(114,) if SMOKE else (114,115)
HOURS=0.10 if SMOKE else 2.0
FRACTION=0.03
POSITION='all'
SELECTED_LAYERS=tuple(PRIOR_REPORT['selected_layers'])
ACTIVE=tuple(PRIOR_REPORT['active_indices'])
assert set(SELECTED_LAYERS)=={13,14} and len(ACTIVE)==8
activation_norms={int(k):float(v) for k,v in PRIOR_REPORT['fixed_activation_norms'].items()}

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

def normalized(vector):
    vector=np.asarray(vector,dtype=np.float32)
    size=float(np.linalg.norm(vector))
    assert np.isfinite(size) and size>1e-10
    return vector/size

W={layer:np.load(probe_files[layer])['W'].astype(np.float32)
   for layer in SELECTED_LAYERS}
BIAS={layer:np.load(probe_files[layer])['bias'].astype(np.float32)
      for layer in SELECTED_LAYERS}
U={layer:W[layer][:,ACTIVE]/np.linalg.norm(W[layer][:,ACTIVE],axis=0)
   for layer in SELECTED_LAYERS}
assert all(W[layer].shape[1]==14 and U[layer].shape[1]==8 for layer in SELECTED_LAYERS)
def combo(weights):
    return {layer:normalized(U[layer]@np.asarray(weights,dtype=np.float32))
            for layer in SELECTED_LAYERS}
EQUAL_DIR=combo(np.ones(len(ACTIVE)))
prior_weights=np.asarray([PRIOR_REPORT['learned_weights'][VECTOR_NAMES[j]] for j in ACTIVE])
LEARNED_DIR=combo(prior_weights)
print('Probe layers',SELECTED_LAYERS,'active',[(j,VECTOR_NAMES[j]) for j in ACTIVE],
      'text decoder',LAYER_PATH,'smoke',SMOKE)
""")

md("""## 1. Direction geometry

Cosines near +1 mean two columns encode effectively the same direction;
cosines near −1 mean they oppose. The singular-value effective rank measures
how many distinct axes the eight normalized columns actually span. The screen
later uses one signed axis for each nearly duplicate or opposite pair.
""")

code("""geometry=[]
for layer in SELECTED_LAYERS:
    gram=U[layer].T@U[layer]
    singular=np.linalg.svd(U[layer],compute_uv=False)
    energy=singular**2/sum(singular**2)
    effective_rank=float(np.exp(-sum(energy*np.log(np.maximum(energy,1e-20)))))
    print('Layer',layer,'effective rank',round(effective_rank,3),
          'singular values',np.round(singular,4).tolist())
    for ai,j in enumerate(ACTIVE):
        for bi,k in enumerate(ACTIVE):
            if bi<=ai: continue
            cosine=float(gram[ai,bi])
            geometry.append({'layer':layer,'attribute_a':VECTOR_NAMES[j],
                             'attribute_b':VECTOR_NAMES[k],'cosine':cosine})
            if abs(cosine)>=0.8:
                print(' ',VECTOR_NAMES[j],VECTOR_NAMES[k],round(cosine,4))
pd.DataFrame(geometry).to_csv(RUN_ROOT/'direction_cosines.csv',index=False)
""")

md("""## 2. New held-out baseline and steered worlds

The policy still sees only its local observation and feasible menu. The saved
new equal-normalized direction is applied at layers 13 and 14 using the same
0.03 strength and fixed norms as the completed experiment. The baseline run
also captures selected frozen batches for the causal screen. Every run spans
both hours in the full notebook. Smoke caps each arm at 24 actions.
""")

cells.append(nbf.v4.new_code_cell(base.cells[5].source.replace('activation_norms={}\n', '')))

code("""class CheckpointPolicy(CapturingPolicy):
    def __init__(self,world,collect):
        super().__init__(model,processor,MAX_NEW_TOKENS,BATCH_SIZE,capture=True)
        self.world=world;self.collect=bool(collect);self.calls=0;self.checkpoints=[]
        self.label='baseline_checkpoint_menu'
    def decide_batch(self,observations):
        keep=self.collect and self.calls%4==0
        if keep:
            checkpoint={'seed':self.world.config.seed,'batch_id':self.calls,
                        'hour':1+int((self.world.minutes-1e-8)//60),
                        'world':copy.deepcopy(self.world),
                        'observations':copy.deepcopy(observations)}
        actions=super().decide_batch(observations)
        if keep:
            checkpoint['baseline_actions']=copy.deepcopy(actions)
            self.checkpoints.append(checkpoint)
        self.calls+=1
        return actions

validation=[];checkpoints=[];source_rows=[]
for seed in VALIDATION_SEEDS:
    world=World(config_for(seed,HOURS))
    baseline_policy=CheckpointPolicy(world,seed in SCREEN_SEEDS)
    baseline_sim=Simulation(world,baseline_policy,output_dir=RUN_ROOT/f'validation_{seed}_baseline')
    baseline_sim.run(progress_every=200,decision_batch_size=BATCH_SIZE)
    baseline_actions=[row['action'] for row in baseline_sim.audit]
    baseline_vectors=np.asarray([row['vector'] for row in baseline_sim.transcript])
    assert len(baseline_actions)==config_for(seed,HOURS).max_actions
    baseline_entry={'seed':seed,'arm':'baseline','actions':len(baseline_actions),
                    'initial_score':float(np.mean(baseline_sim.baseline)),
                    'final_score':float(np.mean(baseline_vectors[-1])),
                    'changed_actions_vs_baseline':0,'different_state_steps':0}
    validation.append(baseline_entry)
    for cp in baseline_policy.checkpoints:
        matching=[row for row in baseline_sim.audit if row['decision_batch_id']==cp['batch_id']]
        assert matching and len(matching)==len(cp['baseline_actions'])
        cp['expected_after']=baseline_sim.transcript[matching[-1]['event_id']-1]['vector']
        checkpoints.append(cp)

    steered,steered_policy=run_arm(f'validation_{seed}_equal',seed,HOURS,
                                   EQUAL_DIR,capture=True)
    steered_actions=action_list(steered)
    steered_vectors=np.asarray([json.loads(line)['vector'] for line in
                  (Path(steered['output_dir'])/'vectors.jsonl').open()])
    assert len(steered_actions)==len(baseline_actions)
    changed=sum(a!=b for a,b in zip(steered_actions,baseline_actions))
    different=np.any(np.abs(steered_vectors-baseline_vectors)>1e-9,axis=1)
    for hour in (1,2):
        if SMOKE and hour==2: continue
        lo,hi=(0,200) if hour==1 else (200,400)
        validation.append({'seed':seed,'arm':'equal_normalized','hour':hour,
                           'changed_actions_vs_baseline':sum(
                               a!=b for a,b in zip(steered_actions[lo:hi],baseline_actions[lo:hi])),
                           'different_state_steps':int(different[lo:hi].sum()),
                           'actions':len(steered_actions[lo:hi]),
                           'initial_score':float(np.mean(baseline_sim.baseline)),
                           'final_score':float(np.mean(steered_vectors[min(hi,len(steered_vectors))-1]))})
    print('Fresh seed',seed,'changed actions',changed,
          'different state steps',int(different.sum()),flush=True)

    for arm,observations,audits in (
        ('baseline',baseline_policy.observations,baseline_sim.audit),
        ('equal_normalized',steered_policy.observations,
         [json.loads(line) for line in (Path(steered['output_dir'])/'actions.jsonl').open()])):
        assert len(observations)==len(audits)
        for i,(observation,audit) in enumerate(zip(observations,audits)):
            if i%2: continue
            source_rows.append({'seed':seed,'arm':arm,'hour':1+i//200,
                                'observation':observation,'action':audit['action'],
                                'target':audit['immediate_action_delta'],
                                'diverged_seed_hour':bool(different[(0 if 1+i//200==1 else 200):
                                                                          (200 if 1+i//200==1 else 400)].any())})
pd.DataFrame(validation).to_csv(RUN_ROOT/'fresh_world_summary.csv',index=False)
assert checkpoints and len(source_rows)>0
if not SMOKE:
    assert set(row['hour'] for row in source_rows)=={1,2}
    assert all(sum(r['different_state_steps']>0 for r in validation
                   if r.get('hour')==hour and r['arm']=='equal_normalized')>=1
               for hour in (1,2)), 'No genuinely diverged steered validation world in both hours'
print('Frozen batches',len(checkpoints),'neutral probe examples',len(source_rows))
""")

md("""## 3. Evaluate the saved probe on neutral response activations

Teacher-forcing uses each newly observed local view and its actual chosen
action, with no steering hook during feature extraction. The saved linear
probe weights predict all 14 immediate hive-vector components. Scales come
from the original probe-training source rows, seeds 90–91, and are fixed
before these fresh-world predictions. Dense regression error is reported
against an always-zero control. The original change-mask classifier was not
saved, so this validation does not claim to reproduce its F1 or exact-zero
metric.
""")

code("""old_steps=PROBE_ROOT/'all_step_changes.csv.gz'
nonzero_by_name={name:[] for name in VECTOR_NAMES}
with gzip.open(old_steps,'rt') as stream:
    for row in csv.DictReader(stream):
        if row['phase']!='source' or int(row['seed']) not in (90,91): continue
        for name in VECTOR_NAMES:
            amount=abs(float(row['immediate_'+name]))
            if amount>1e-10:nonzero_by_name[name].append(amount)
SCALES=np.asarray([max(float(np.percentile(nonzero_by_name[name],75)),1e-4)
                   if nonzero_by_name[name] else 1.0 for name in VECTOR_NAMES],dtype=np.float32)

def encoded_chat(messages,add_generation_prompt):
    enc=processor.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=add_generation_prompt,return_tensors='pt',return_dict=True)
    return {k:v for k,v in enc.items() if isinstance(v,torch.Tensor)}

pooled={};response_span=[0,0];handles=[]
for layer_i in SELECTED_LAYERS:
    def capture(_module,_inputs,output,layer_i=layer_i):
        hidden=output[0] if isinstance(output,(tuple,list)) else output
        start,end=response_span
        pooled[layer_i]=hidden[0,start:end].float().mean(dim=0).detach().cpu().numpy()
    handles.append(DECODER_LAYERS[layer_i].register_forward_hook(capture))
features={layer:np.empty((len(source_rows),W[layer].shape[0]),dtype=np.float32)
          for layer in SELECTED_LAYERS}
started=time.perf_counter()
try:
    for index,row in enumerate(source_rows):
        enriched=dict(row['observation'],feasible_actions=feasible_actions(row['observation']))
        prompt=[{'role':'system','content':MENU_SYSTEM_PROMPT},
                {'role':'user','content':json.dumps(enriched,ensure_ascii=False,
                                                  separators=(',',':'),allow_nan=False)}]
        answer=json.dumps(row['action'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
        p=encoded_chat(prompt,True);f=encoded_chat(prompt+[{'role':'assistant','content':answer}],False)
        pids,fids=p['input_ids'][0],f['input_ids'][0]
        assert torch.equal(pids,fids[:len(pids)])
        response_span[:]=[len(pids),len(fids)-1]
        assert response_span[1]>response_span[0] and len(fids)<=4096
        pooled.clear()
        with torch.inference_mode():
            result=model(**{k:v.to(model.device) for k,v in f.items()},use_cache=False)
        del result
        for layer_i in SELECTED_LAYERS:features[layer_i][index]=pooled[layer_i]
        if (index+1)%100==0:
            print('Probe validation features',index+1,'/',len(source_rows),'minutes',
                  round((time.perf_counter()-started)/60,1),flush=True)
finally:
    for handle in handles:handle.remove()
truth=np.asarray([row['target'] for row in source_rows],dtype=np.float32)
assert truth.shape==(len(source_rows),14)

def dense_error(pred,truth):
    weights=np.where(np.abs(truth)>1e-10,4.0,1.0)
    return float(np.sum(weights*np.abs((pred-truth)/SCALES))/weights.sum())

metric_rows=[]
for layer in SELECTED_LAYERS:
    pred=features[layer]@W[layer]+BIAS[layer]
    assert np.isfinite(pred).all()
    for arm in ('baseline','equal_normalized'):
        for hour in (1,2):
            if SMOKE and hour==2: continue
            for divergent_only in (False,True):
                indices=[i for i,row in enumerate(source_rows)
                         if row['arm']==arm and row['hour']==hour and
                            (not divergent_only or row['diverged_seed_hour'])]
                if not indices: continue
                y=truth[indices];p=pred[indices]
                changed=np.abs(y)>1e-10
                metric_rows.append({'layer':layer,'arm':arm,'hour':hour,
                    'diverged_seed_hours_only':divergent_only,'examples':len(indices),
                    'nonzero_target_entries':int(changed.sum()),
                    'weighted_scaled_mae':dense_error(p,y),
                    'always_zero_weighted_scaled_mae':dense_error(np.zeros_like(y),y),
                    'changed_sign_accuracy':float(np.mean(np.sign(p[changed])==
                                                   np.sign(y[changed]))) if changed.any() else None})
pd.DataFrame(metric_rows).to_csv(RUN_ROOT/'fresh_probe_metrics.csv',index=False)
print(pd.DataFrame(metric_rows).to_string(index=False))
""")

md("""## 4. Paired causal action screen at frozen decision batches

Columns for brood nutrition and pathogen suppression are nearly exact
opposites or duplicates of pollen reserve and hygiene. We screen both signs
of six representative axes, plus the saved equal and learned combinations.
Every candidate gets the same observations, menu, baseline actions, and
cloned physical state. A batch result is the post-action hive-vector
difference after its 16 ordered actions; changed actions also receive
one-bee isolated counterfactual comparisons. The screen uses at most 16
frozen batches across two fresh seeds, eight in each simulated hour.
""")

code("""REPRESENTATIVES=(2,3,4,8,10,11)
assert all(j in ACTIVE for j in REPRESENTATIVES)
def single_direction(index,sign):
    column=ACTIVE.index(index)
    return {layer:U[layer][:,column].copy()*sign for layer in SELECTED_LAYERS}

candidate_configs={}
for j in REPRESENTATIVES:
    for sign,label in ((1,'plus'),(-1,'minus')):
        candidate_configs[f'{VECTOR_NAMES[j]}_{label}']={
            'directions':single_direction(j,sign),'strength':0.03,
            'attribute':VECTOR_NAMES[j],'sign':sign}
candidate_configs['equal_normalized']={'directions':EQUAL_DIR,'strength':0.03,
                                       'attribute':'combination','sign':1}
candidate_configs['learned_combination']={'directions':LEARNED_DIR,'strength':0.03,
                                         'attribute':'combination','sign':1}

def branch_batch(cp,actions):
    clone=copy.deepcopy(cp['world']);observer=Observer(clone)
    before=np.asarray(observer.snapshot(),dtype=np.float64)
    statuses=[]
    for observation,action in zip(cp['observations'],actions):
        outcome=clone.execute(observation['self']['id'],action,
                              min(clone.config.round_minutes,HOURS*60))
        statuses.append(outcome['status'])
    after=np.asarray(observer.snapshot(),dtype=np.float64)
    return after-before,after,statuses

def isolated_difference(cp,index,candidate_action):
    baseline_action=cp['baseline_actions'][index]
    baseline=copy.deepcopy(cp['world']);intervention=copy.deepcopy(cp['world'])
    bee_id=cp['observations'][index]['self']['id']
    baseline.execute(bee_id,baseline_action,min(baseline.config.round_minutes,HOURS*60))
    intervention.execute(bee_id,candidate_action,
                         min(intervention.config.round_minutes,HOURS*60))
    return np.asarray(Observer(intervention).snapshot())-np.asarray(Observer(baseline).snapshot())

baseline_branches={}
for ci,cp in enumerate(checkpoints):
    delta,after,statuses=branch_batch(cp,cp['baseline_actions'])
    assert np.allclose(after,cp['expected_after'],rtol=0,atol=1e-8),(
        cp['seed'],cp['batch_id'],np.max(np.abs(after-cp['expected_after'])))
    baseline_branches[ci]=delta
print('Replayed',len(checkpoints),'frozen baseline batches exactly')

screen_rows=[];decision_rows=[]
def evaluate_candidate(name,config):
    global FRACTION
    FRACTION=float(config['strength'])
    policy=ColumnSteeredPolicy(config['directions'])
    batch_effects=[];changed_total=0;isolated_effects=[];positive=negative=0
    for ci,cp in enumerate(checkpoints):
        actions=policy.decide_batch(cp['observations'])
        assert len(actions)==len(cp['baseline_actions'])
        delta,after,statuses=branch_batch(cp,actions)
        difference=delta-baseline_branches[ci]
        score=float(np.mean(difference));batch_effects.append(score)
        positive+=score>1e-10;negative+=score< -1e-10
        for i,(baseline_action,action) in enumerate(zip(cp['baseline_actions'],actions)):
            if action==baseline_action: continue
            changed_total+=1
            isolated=isolated_difference(cp,i,action)
            isolated_effects.append(float(isolated.mean()))
            decision_rows.append({'candidate':name,'strength':FRACTION,
                'seed':cp['seed'],'hour':cp['hour'],'batch_id':cp['batch_id'],
                'bee_id':cp['observations'][i]['self']['id'],
                'baseline_action':baseline_action['action'],'steered_action':action['action'],
                'isolated_score_difference':float(isolated.mean()),
                'isolated_vector_difference':json.dumps(isolated.tolist()),
                'batch_score_difference':score})
    result={'candidate':name,'strength':FRACTION,'attribute':config['attribute'],
            'sign':config['sign'],'batches':len(checkpoints),'decisions':sum(
                len(cp['observations']) for cp in checkpoints),
            'changed_actions':changed_total,'mean_batch_score_effect':float(np.mean(batch_effects)),
            'positive_batches':int(positive),'negative_batches':int(negative),
            'mean_isolated_changed_action_effect':float(np.mean(isolated_effects))
                if isolated_effects else 0.0,'hook_calls':policy.hook_calls}
    screen_rows.append(result)
    print(name,'strength',FRACTION,'changed',changed_total,
          'mean batch score effect',round(result['mean_batch_score_effect'],8),flush=True)
    return result

for name,config in candidate_configs.items():
    evaluate_candidate(name,config)
eligible=sorted((r for r in screen_rows if r['mean_batch_score_effect']>0 and
                 r['changed_actions']>=3 and r['candidate'] not in
                 ('equal_normalized','learned_combination')),
                key=lambda r:r['mean_batch_score_effect'],reverse=True)
shortlist=[r['candidate'] for r in eligible[:2]]
if SMOKE and not shortlist:
    # Exercise strength tuning and complete-hive execution even if the tiny
    # smoke checkpoint contains no beneficial changed action.
    shortlist=['adult_energy_sufficiency_plus']
for name in shortlist:
    for strength in (0.015,0.045):
        config=dict(candidate_configs[name],strength=strength)
        evaluate_candidate(f'{name}_strength_{strength}',config)
pd.DataFrame(screen_rows).to_csv(RUN_ROOT/'causal_screen.csv',index=False)
pd.DataFrame(decision_rows).to_csv(RUN_ROOT/'causal_changed_decisions.csv',index=False)
print('Shortlist',shortlist)
""")

md("""## 5. Fresh two-hour hive test of selected directions

The screen may find no positive candidate; in that case no new full hive run
is warranted. Otherwise the best tested strength for each shortlisted signed
axis is fixed before evaluation on new seeds 114–115. These seeds were absent
from probe training, probe validation, and the causal screen. The final score
and complete 14-value vector are reported alongside the baseline for each
seed; action-level transcripts remain in the Kaggle output.
""")

code("""selected=[]
for base_name in shortlist:
    options=[r for r in screen_rows if r['candidate']==base_name or
             r['candidate'].startswith(base_name+'_strength_')]
    winner=max(options,key=lambda r:r['mean_batch_score_effect'])
    config=dict(candidate_configs[base_name],strength=winner['strength'])
    selected.append((base_name,config,winner))
print('Selected for fresh hive test',[(n,c['strength'],r['mean_batch_score_effect'])
                                      for n,c,r in selected])

final_rows=[]
if selected:
    for seed in FINAL_SEEDS:
        baseline,_=run_arm(f'final_{seed}_baseline',seed,HOURS)
        baseline['arm']='baseline';baseline['changed_actions_vs_baseline']=0
        baseline['score_difference_vs_baseline']=0.0
        final_rows.append(baseline)
        for name,config,winner in selected:
            FRACTION=float(config['strength'])
            row,_=run_arm(f'final_{seed}_{name}',seed,HOURS,config['directions'])
            row['arm']=name
            final_rows.append(annotate_changes(row,baseline))
    pd.DataFrame([{k:v for k,v in row.items() if k not in
                  ('initial_vector','final_vector','vector_difference_vs_baseline')}
                  for row in final_rows]).to_csv(RUN_ROOT/'final_hive_summary.csv',index=False)
    pd.DataFrame([{'seed':row['seed'],'arm':row['arm'],'attribute':VECTOR_NAMES[j],
                   'final':row['final_vector'][j]}
                  for row in final_rows for j in range(14)]).to_csv(
                      RUN_ROOT/'final_hive_vectors.csv',index=False)
print('Fresh hive final rows',len(final_rows))
""")

code("""report={'smoke':SMOKE,'probe_source':str(PROBE_ROOT),
        'probe_layers':list(SELECTED_LAYERS),'active_indices':list(ACTIVE),
        'validation_seeds':list(VALIDATION_SEEDS),'screen_seeds':list(SCREEN_SEEDS),
        'final_seeds':list(FINAL_SEEDS),'hours':HOURS,
        'fresh_worlds':validation,'probe_metrics':metric_rows,
        'direction_cosines':geometry,'causal_screen':screen_rows,
        'shortlist':shortlist,
        'selected_final_directions':[{'name':name,'strength':config['strength']}
                                     for name,config,_ in selected],
        'final_hive_runs':final_rows}
(RUN_ROOT/'direction_audit_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('Artifacts',RUN_ROOT)
""")


def write_notebook(path, smoke):
    copied = [nbf.v4.new_markdown_cell(cell.source) if cell.cell_type == "markdown"
              else nbf.v4.new_code_cell(cell.source) for cell in cells]
    if smoke:
        for cell in copied:
            if cell.cell_type == "code" and "SMOKE=False" in cell.source:
                cell.source = cell.source.replace("SMOKE=False", "SMOKE=True", 1)
                break
    notebook = nbf.v4.new_notebook(cells=copied, metadata=base.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


write_notebook(ROOT / "beehive_direction_audit.ipynb", False)
for smoke, stage in ((True, "kernel_smoke"), (False, "kernel_full")):
    folder = ROOT / "kaggle_deploy/direction_audit" / stage
    folder.mkdir(parents=True, exist_ok=True)
    write_notebook(folder / "beehive_direction_audit.ipynb", smoke)
    suffix = "-smoke" if smoke else ""
    metadata = {
        "id": "faithfulchukwunwogor/beehive-direction-audit" + suffix,
        "title": "Beehive Direction Audit" + (" Smoke" if smoke else ""),
        "code_file": "beehive_direction_audit.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true", "machine_shape": "",
        "dataset_sources": [
            "faithfulchukwunwogor/beehive-qwen35-runtime",
            "faithfulchukwunwogor/beehive-qwen35-menu-adapter",
        ],
        "competition_sources": [],
        "kernel_sources": ["faithfulchukwunwogor/beehive-mixed-probe-two-hour"],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Built fresh-world probe validation and causal direction audit notebooks")
