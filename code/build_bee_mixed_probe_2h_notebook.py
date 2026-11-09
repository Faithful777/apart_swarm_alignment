"""Build one Kaggle notebook: mixed-state probe, calibrated steering, paired two-hour run."""
from pathlib import Path
import json
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
base = nbf.read(ROOT / 'beehive_column_steering_4h.ipynb', as_version=4)
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md('''# Probe training on steered and unsteered hive states

This is one executable experiment. It starts with the existing continued
Qwen3.5-0.8B bee adapter and the saved layer-19/20 vector probes. The old probes
are used **only to generate a steered source environment**. For each of three
source seeds, baseline and old equal-normalized steering each run for two
simulated hours. A new 14-output probe is then fitted to the local observations,
chosen actions, and immediate hive-vector effects from both environments and
both hours. Teacher-forced feature extraction uses neutral model activations
for every source example, keeping the new probe's activation coordinates
comparable across the two environment types.

The notebook screens all text decoder layers, reports probe accuracy separately
by hour and environment on held-out source examples, and constructs new
individual and equal-normalized attribute directions. Separate two-hour
calibration runs learn nonnegative combination weights from measured hive
outcomes. Finally, four fresh seeds compare baseline, equal-weight steering,
and learned-weight steering for two simulated hours each. Every bee sees only
its local view and feasible-action menu. Every action receives a timestamped
14-component researcher-only vector snapshot.

The probe target is still the **immediate change after one action**. The new
training distribution includes states reached under steering; this notebook
does not turn the probe into a future-action outcome predictor.''')

cells.append(nbf.v4.new_code_cell(base.cells[1].source))

imports = base.cells[2].source
imports = imports.replace('import json, math, sys, time, zipfile',
                          'import copy, json, math, sys, time, zipfile')
imports = imports.replace('import torch.nn as nn',
                          'import torch.nn as nn\nfrom sklearn.linear_model import Ridge, LogisticRegression\nfrom sklearn.metrics import f1_score')
imports = imports.replace("('bee_columns_'+", "('bee_mixed_probe_'+")
imports = imports.replace('from bee_menu_qwen_policy import BeeMenuQwenPolicy',
                          'from bee_menu_qwen_policy import BeeMenuQwenPolicy, MENU_SYSTEM_PROMPT\n'
                          'from bee_feasibility import feasible_actions')
code(imports)

code('''SMOKE=False
BATCH_SIZE=16
MAX_NEW_TOKENS=160
SOURCE_SEEDS=(90,91,92)
SOURCE_HOURS=0.10 if SMOKE else 2.0
CAL_SEEDS=(93,) if SMOKE else (93,94)
CAL_HOURS=0.10 if SMOKE else 2.0
FINAL_SEEDS=(100,) if SMOKE else (100,101,102,103)
FINAL_HOURS=0.10 if SMOKE else 2.0
FRACTION=0.03
POSITION='all'
BOOTSTRAP_LAYERS=(19,20)
SELECTED_LAYERS=BOOTSTRAP_LAYERS

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
N_LAYERS=len(DECODER_LAYERS)
assert N_LAYERS>max(BOOTSTRAP_LAYERS)

def normalized(vector):
    vector=np.asarray(vector,dtype=np.float32)
    size=float(np.linalg.norm(vector))
    assert np.isfinite(size) and size>1e-10
    return vector/size

old_W={layer:np.load(probe_files[layer])['W'].astype(np.float32)
       for layer in BOOTSTRAP_LAYERS}
old_norms={layer:np.linalg.norm(old_W[layer],axis=0) for layer in BOOTSTRAP_LAYERS}
old_active=np.flatnonzero(np.all(np.stack(
    [old_norms[layer]>1e-10 for layer in BOOTSTRAP_LAYERS]),axis=0))
assert old_active.tolist()==[2,3,8,10,11,12]
BOOTSTRAP_DIRS={layer:normalized((old_W[layer][:,old_active]/
                                  old_norms[layer][old_active]).sum(axis=1))
                for layer in BOOTSTRAP_LAYERS}
print('Text decoder',LAYER_PATH,N_LAYERS,'layers; bootstrap columns',old_active.tolist(),
      'smoke',SMOKE)''')

md('''## 1. Run baseline and steered source environments over both hours

The initial steering only changes how source worlds evolve. It uses the old
equal-normalized direction at the previously tested 0.03 strength. Each bee's
model still receives its local observation and menu. The first baseline run
measures fixed activation norms; the same norms are used for all source steered
runs. Full runs have 400 actions per source arm, covering both simulated hours.
The smoke run caps each arm at 24 actions to catch code and data errors quickly.''')

code('''def config_for(seed,hours):
    data=dict(SOURCE_CONFIG)
    data.update(seed=int(seed),duration_hours=float(hours),
                max_actions=int(data['n_bees']*math.ceil(hours*60/data['round_minutes'])),
                model='Qwen3.5-0.8B continued menu LoRA; mixed-state probe experiment')
    if SMOKE: data['max_actions']=min(data['max_actions'],24)
    return Config(**data)

class CapturingPolicy(BeeMenuQwenPolicy):
    def __init__(self,*args,capture=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.capture=bool(capture); self.observations=[]
    def decide_batch(self,observations):
        if self.capture: self.observations.extend(copy.deepcopy(observations))
        return super().decide_batch(observations)

class NormCapturePolicy(CapturingPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.samples={layer:[] for layer in SELECTED_LAYERS}
    def decide_batch(self,observations):
        handles=[]
        try:
            for layer_i in SELECTED_LAYERS:
                def capture_norm(_module,_inputs,output,layer_i=layer_i):
                    if len(self.samples[layer_i])>=256: return
                    hidden=output[0] if isinstance(output,(tuple,list)) else output
                    values=hidden[:,-1,:].float().norm(dim=-1).detach().cpu().tolist()
                    self.samples[layer_i].extend(values[:256-len(self.samples[layer_i])])
                handles.append(DECODER_LAYERS[layer_i].register_forward_hook(capture_norm))
            return super().decide_batch(observations)
        finally:
            for handle in handles: handle.remove()

activation_norms={}
class ColumnSteeredPolicy(CapturingPolicy):
    def __init__(self,directions,capture=False):
        super().__init__(model,processor,MAX_NEW_TOKENS,BATCH_SIZE,capture=capture)
        self.directions=directions; self.hook_calls=0
        self.label='mixed_probe_column_steered_menu'
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

def run_arm(name,seed,hours,directions=None,capture=False,capture_norm=False):
    world=World(config_for(seed,hours))
    if capture_norm:
        policy=NormCapturePolicy(model,processor,MAX_NEW_TOKENS,BATCH_SIZE,capture=capture)
    elif directions is None:
        policy=CapturingPolicy(model,processor,MAX_NEW_TOKENS,BATCH_SIZE,capture=capture)
    else:
        policy=ColumnSteeredPolicy(directions,capture=capture)
    output_dir=RUN_ROOT/name
    sim=Simulation(world,policy,output_dir=output_dir)
    transcript=sim.run(progress_every=200,decision_batch_size=BATCH_SIZE)
    audit=sim.audit; outcomes=Counter(r['outcome']['status'] for r in audit)
    initial=sim.baseline; final=transcript[-1]['vector']
    immediate=float(sum(np.mean(r['immediate_action_delta']) for r in audit))
    background=float(sum(np.mean(r['background_delta']) for r in audit))
    row={'seed':int(seed),'name':name,'hours':hours,'actions':len(audit),
         'applied':outcomes['applied'],'rejected':outcomes['rejected'],
         'schema_invalid':sum(validate_action(r['action']) is not None for r in audit),
         'initial_vector':initial,'final_vector':final,
         'initial_score':weighted_research_score(initial,[1.0]*14),
         'final_score':weighted_research_score(final,[1.0]*14),
         'immediate_score_change':immediate,'background_score_change':background,
         'hook_calls':getattr(policy,'hook_calls',0),'output_dir':str(output_dir)}
    assert len(audit)==config_for(seed,hours).max_actions
    assert abs(row['initial_score']+immediate+background-row['final_score'])<1e-6
    assert all(len(x['vector'])==14 and all(np.isfinite(x['vector'])) and
               all(0<=v<=1 for v in x['vector']) for x in transcript)
    if directions is not None: assert row['hook_calls']>0
    if capture: assert len(policy.observations)==len(audit)
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

code('''source_runs=[]; source_summaries=[]
for seed in SOURCE_SEEDS:
    baseline,policy=run_arm(f'source_{seed}_baseline',seed,SOURCE_HOURS,
                            capture=True,capture_norm=(seed==SOURCE_SEEDS[0]))
    if seed==SOURCE_SEEDS[0]:
        activation_norms.update({layer:float(np.median(policy.samples[layer]))
                                 for layer in BOOTSTRAP_LAYERS})
        assert all(np.isfinite(v) and v>0 for v in activation_norms.values())
        print('Bootstrap activation norms',activation_norms)
    baseline['arm']='baseline'; baseline['changed_actions_vs_baseline']=0
    baseline['score_difference_vs_baseline']=0.0
    baseline['vector_difference_vs_baseline']=[0.0]*14
    source_runs.append((baseline,policy));source_summaries.append(baseline)
    steered,policy=run_arm(f'source_{seed}_bootstrap_equal',seed,SOURCE_HOURS,
                           BOOTSTRAP_DIRS,capture=True)
    steered['arm']='bootstrap_equal'; annotate_changes(steered,baseline)
    source_runs.append((steered,policy));source_summaries.append(steered)
source_changed=sum(r['changed_actions_vs_baseline'] for r in source_summaries)
if not SMOKE: assert source_changed>0,'Bootstrap steering did not diversify source states'
print('Source actions',sum(r['actions'] for r in source_summaries),
      'bootstrap changed actions',source_changed)
pd.DataFrame([{k:v for k,v in row.items() if k not in
              ('initial_vector','final_vector','vector_difference_vs_baseline')}
              for row in source_summaries]).to_csv(RUN_ROOT/'source_run_summary.csv',index=False)''')

md('''## 2. Extract neutral response activations and fit the 14-output probe

The probe input is the model's response representation given the current
local observation and the action that the source policy chose. Source states
come from both baseline and steered worlds, in both hour windows. The probe
target is the immediate post-action minus pre-action 14-vector. Source seeds
90 and 91 train the probe. Seed 92 supplies alternating events for layer
selection and a separate report split; each split covers both environment
types and both hours. A fixed 0.5 change-mask threshold makes predicted
unchanged attributes exactly zero. The report includes hour-by-environment
metrics on the held-out events.''')

code('''observations=[]; action_rows=[]; source_seed_ids=[]; source_arm_ids=[]; source_hours=[]
for summary,policy in source_runs:
    rows=[json.loads(line) for line in
          (Path(summary['output_dir'])/'actions.jsonl').open()]
    assert len(rows)==len(policy.observations)==summary['actions']
    observations.extend(policy.observations);action_rows.extend(rows)
    source_seed_ids.extend([summary['seed']]*len(rows))
    source_arm_ids.extend([summary['arm']]*len(rows))
    per_hour=int(SOURCE_CONFIG['n_bees']*60/SOURCE_CONFIG['round_minutes'])
    source_hours.extend([1+(int(row['event_id'])-1)//per_hour for row in rows])
source_seed_ids=np.asarray(source_seed_ids)
source_arm_ids=np.asarray(source_arm_ids)
source_hours=np.asarray(source_hours)
targets=np.asarray([row['immediate_action_delta'] for row in action_rows],dtype=np.float32)
assert targets.shape==(len(observations),14)
train_pos=np.flatnonzero(np.isin(source_seed_ids,SOURCE_SEEDS[:2]))
held=np.flatnonzero(source_seed_ids==SOURCE_SEEDS[-1])
screen_pos,report_pos=held[::2],held[1::2]
assert len(train_pos)>0 and len(screen_pos)>0 and len(report_pos)>0
if not SMOKE:
    for positions in (train_pos,screen_pos,report_pos):
        assert set(source_hours[positions])=={1,2}
        assert set(source_arm_ids[positions])=={'baseline','bootstrap_equal'}
print('Probe splits',len(train_pos),len(screen_pos),len(report_pos))
print('Source condition counts',pd.DataFrame({'seed':source_seed_ids,'arm':source_arm_ids,
                                              'hour':source_hours}).value_counts().to_string())

def encoded_chat(messages,add_generation_prompt):
    enc=processor.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=add_generation_prompt,return_tensors='pt',return_dict=True)
    return {k:v for k,v in enc.items() if isinstance(v,torch.Tensor)}

pooled={}; response_span=[0,0]; handles=[]
for layer_i,layer in enumerate(DECODER_LAYERS):
    def capture(_module,_inputs,output,layer_i=layer_i):
        hidden=output[0] if isinstance(output,(tuple,list)) else output
        start,end=response_span
        pooled[layer_i]=hidden[0,start:end].float().mean(dim=0).detach().cpu().numpy()
    handles.append(layer.register_forward_hook(capture))
features=None; feature_start=time.perf_counter()
try:
    for index,(obs,row) in enumerate(zip(observations,action_rows)):
        enriched=dict(obs,feasible_actions=feasible_actions(obs))
        prompt=[{'role':'system','content':MENU_SYSTEM_PROMPT},
                {'role':'user','content':json.dumps(enriched,ensure_ascii=False,
                                                  separators=(',',':'),allow_nan=False)}]
        action_text=json.dumps(row['action'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
        prompt_enc=encoded_chat(prompt,True)
        full_enc=encoded_chat(prompt+[{'role':'assistant','content':action_text}],False)
        pids,fids=prompt_enc['input_ids'][0],full_enc['input_ids'][0]
        assert torch.equal(pids,fids[:len(pids)])
        response_span[:]=[len(pids),len(fids)-1]
        assert response_span[1]>response_span[0] and len(fids)<=4096
        pooled.clear()
        with torch.inference_mode():
            result=model(**{k:v.to(model.device) for k,v in full_enc.items()},use_cache=False)
        del result
        if features is None:
            features=np.empty((len(observations),N_LAYERS,len(pooled[0])),dtype=np.float32)
        for layer_i in range(N_LAYERS): features[index,layer_i]=pooled[layer_i]
        if (index+1)%100==0:
            print('Features',index+1,'/',len(observations),'minutes',
                  round((time.perf_counter()-feature_start)/60,1),flush=True)
finally:
    for handle in handles: handle.remove()
assert features is not None and np.isfinite(features).all()
print('Feature shape',features.shape)''')

code('''scales=[]
for j in range(14):
    nz=np.abs(targets[train_pos,j]);nz=nz[nz>1e-10]
    scales.append(max(float(np.percentile(nz,75)),1e-4) if len(nz) else 1.0)
scales=np.asarray(scales,dtype=np.float32)
y_scaled=targets/scales
row_weights=1+3*np.any(np.abs(targets[train_pos])>1e-10,axis=1).astype(np.float32)

def weighted_mae(pred,truth):
    weights=np.where(np.abs(truth)>1e-10,4.0,1.0)
    return float(np.sum(weights*np.abs(pred-truth))/weights.sum())

layer_models={};screen_rows=[]
for layer_i in range(N_LAYERS):
    x=features[train_pos,layer_i]
    mu=x.mean(axis=0);sd=np.maximum(x.std(axis=0),1e-3)
    ridge=Ridge(alpha=100.0,solver='lsqr',max_iter=1000)
    ridge.fit((x-mu)/sd,y_scaled[train_pos],sample_weight=row_weights)
    pred=ridge.predict((features[screen_pos,layer_i]-mu)/sd)
    screen_rows.append({'layer':layer_i,'weighted_scaled_mae':weighted_mae(pred,y_scaled[screen_pos])})
    layer_models[layer_i]=(ridge,mu,sd)
screen_table=pd.DataFrame(screen_rows).sort_values('weighted_scaled_mae')
SELECTED_LAYERS=tuple(screen_table.head(2).layer.astype(int).tolist())
screen_table.to_csv(RUN_ROOT/'layer_screen.csv',index=False)
print('Selected layers',SELECTED_LAYERS,'screen leaders',screen_table.head(5).to_dict('records'))

train_change_counts=np.count_nonzero(np.abs(targets[train_pos])>1e-10,axis=0)
ACTIVE=np.flatnonzero(train_change_counts>=(1 if SMOKE else 5))
assert len(ACTIVE)>=1,train_change_counts
W={};U={};probe_metrics=[];probe_controls={}
for layer_i in SELECTED_LAYERS:
    ridge,mu,sd=layer_models[layer_i]
    x_train=(features[train_pos,layer_i]-mu)/sd
    x_report=(features[report_pos,layer_i]-mu)/sd
    signed=ridge.predict(x_report)*scales
    sparse=np.zeros_like(signed)
    for j in range(14):
        changed=(np.abs(targets[train_pos,j])>1e-10).astype(int)
        if len(np.unique(changed))<2:
            prob=np.full(len(report_pos),float(changed[0]))
        else:
            clf=LogisticRegression(C=0.1,class_weight='balanced',max_iter=300,
                                   solver='liblinear',random_state=271)
            clf.fit(x_train,changed)
            prob=clf.predict_proba(x_report)[:,1]
        sparse[:,j]=np.where(prob>=0.5,signed[:,j],0)
    W[layer_i]=(ridge.coef_.T/sd[:,None])*scales[None,:]
    bias=(ridge.intercept_-(mu/sd)@ridge.coef_.T)*scales
    norms=np.linalg.norm(W[layer_i][:,ACTIVE],axis=0)
    assert np.all(np.isfinite(norms)) and np.all(norms>1e-10)
    U[layer_i]=W[layer_i][:,ACTIVE]/norms
    np.savez(RUN_ROOT/f'new_probe_layer_{layer_i}.npz',W=W[layer_i],bias=bias,
             active_indices=ACTIVE,train_change_counts=train_change_counts)
    for arm in ('baseline','bootstrap_equal','all'):
        for hour in (1,2):
            mask=np.ones(len(report_pos),dtype=bool)
            if arm!='all':mask &= source_arm_ids[report_pos]==arm
            mask &= source_hours[report_pos]==hour
            if not mask.any():continue
            pred=sparse[mask];truth=targets[report_pos][mask]
            truth_change=np.abs(truth)>1e-10;pred_change=pred!=0
            probe_metrics.append({'layer':layer_i,'arm':arm,'hour':hour,'examples':int(mask.sum()),
                'nonzero_target_entries':int(truth_change.sum()),
                'weighted_scaled_mae':weighted_mae(pred/scales,truth/scales),
                'change_f1':float(f1_score(truth_change.ravel(),pred_change.ravel(),zero_division=0)),
                'unchanged_exact_zero_rate':float(np.mean(~pred_change[~truth_change])),
                'changed_sign_accuracy':float(np.mean(np.sign(pred[truth_change])==
                                                   np.sign(truth[truth_change]))) if truth_change.any() else None})
    probe_controls[str(layer_i)]={'always_zero_weighted_scaled_mae':weighted_mae(
        np.zeros_like(targets[report_pos]),y_scaled[report_pos])}
probe_table=pd.DataFrame(probe_metrics)
probe_table.to_csv(RUN_ROOT/'probe_hour_environment_metrics.csv',index=False)
print('Active steerable attributes',[(int(j),VECTOR_NAMES[j],int(train_change_counts[j])) for j in ACTIVE])
print(probe_table.to_string(index=False))

def build_directions(weights):
    weights=np.asarray(weights,dtype=np.float32)
    assert weights.shape==(len(ACTIVE),) and np.all(weights>=0) and weights.sum()>0
    return {layer:normalized(U[layer]@weights) for layer in SELECTED_LAYERS}

DIRS={f'attribute_{j}':{layer:U[layer][:,k].copy() for layer in SELECTED_LAYERS}
      for k,j in enumerate(ACTIVE)}
DIRS['equal_normalized']=build_directions(np.ones(len(ACTIVE)))
activation_norms.update({layer:float(np.median(np.linalg.norm(features[train_pos,layer],axis=1)))
                         for layer in SELECTED_LAYERS})
assert all(np.isfinite(activation_norms[layer]) and activation_norms[layer]>0
           for layer in SELECTED_LAYERS)
print('New directions and activation norms ready')''')

md('''## 3. Calibrate new attribute-direction weights over two-hour worlds

The calibration seeds are independent of probe-source and final seeds. Each
individual new direction is run against its paired baseline for the same
two-hour horizon as the final evaluation. Paired final-score effects are
standardized, softmax weighted, and shrunk 25% toward uniform weights. This
learns a nonnegative combination from measured simulator outcomes without
backpropagating through discrete bee choices. The calibration rule and 0.03
strength are fixed before the final seeds are evaluated.''')

cells.append(nbf.v4.new_code_cell(base.cells[7].source))
cells.append(nbf.v4.new_code_cell(base.cells[9].source))

md('''## 4. Paired two-hour final evaluation on new seeds

Seeds 100–103 were not used for the probe, layer selection, or combination
weights. Each of baseline, equal-normalized steering, and outcome-weighted
steering acts once per living bee every fifteen simulated minutes for two
hours. The action order, background updates, researcher-only vector snapshots,
and local feasible menu match the previous notebook.''')

final_code=base.cells[11].source.replace(
    'for arm,directions in DIRS.items():',
    "for arm in ('equal_normalized','learned_combination'):\n        directions=DIRS[arm]")
code(final_code)

summary_code=base.cells[12].source.replace(
    'for arm in DIRS:',
    "for arm in ('equal_normalized','learned_combination'):")
summary_code=summary_code.replace(
    "'probe_source':'saved layer 19 and 20 vector probes; no retraining',",
    "'probe_source':'new mixed-environment immediate vector probe',\n"
    "        'source_seeds':list(SOURCE_SEEDS),'source_hours':SOURCE_HOURS,\n"
    "        'source_runs':source_summaries,\n"
    "        'probe_split':{'train':len(train_pos),'layer_screen':len(screen_pos),\n"
    "                       'independent_report':len(report_pos)},\n"
    "        'probe_train_change_counts':dict(zip(VECTOR_NAMES,train_change_counts.tolist())),\n"
    "        'probe_hour_environment_metrics':probe_metrics,\n"
    "        'probe_controls':probe_controls,")
code(summary_code)

md('''## 5. Preserve every step change and overlay the three trajectories

The timestamped CSV contains every source, calibration, and final action's
immediate, background, and total change for all fourteen attributes. The
notebook saves one plot per final seed with baseline, equal-weight, and
outcome-weighted trajectories on the same axes, plus their mean score
trajectory across seeds.''')

plot_code=base.cells[14].source.replace(
    "all_runs=[('calibration',row) for row in calibration]+[('final',row) for row in final_runs]",
    "all_runs=[('source',row) for row in source_summaries]+[('calibration',row) for row in calibration]+[('final',row) for row in final_runs]")
code(plot_code)


def write_notebook(path,smoke):
    copied=[(nbf.v4.new_markdown_cell(c.source) if c.cell_type=='markdown'
             else nbf.v4.new_code_cell(c.source)) for c in cells]
    if smoke:
        for cell in copied:
            if cell.cell_type=='code' and 'SMOKE=False' in cell.source:
                cell.source=cell.source.replace('SMOKE=False','SMOKE=True',1)
                break
    nb=nbf.v4.new_notebook(cells=copied,metadata=base.metadata)
    path.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,path)


write_notebook(ROOT/'beehive_mixed_probe_2h.ipynb',False)
for smoke,stage in ((True,'kernel_smoke'),(False,'kernel_full')):
    folder=ROOT/'kaggle_deploy/mixed_probe_2h'/stage
    folder.mkdir(parents=True,exist_ok=True)
    write_notebook(folder/'beehive_mixed_probe_2h.ipynb',smoke)
    suffix='-smoke' if smoke else ''
    metadata={
        'id':'faithfulchukwunwogor/beehive-mixed-probe-two-hour'+suffix,
        'title':'Beehive Mixed Probe Two Hour'+(' Smoke' if smoke else ''),
        'code_file':'beehive_mixed_probe_2h.ipynb',
        'language':'python','kernel_type':'notebook','is_private':'true',
        'enable_gpu':'true','enable_internet':'true','machine_shape':'',
        'dataset_sources':[
            'faithfulchukwunwogor/beehive-qwen35-runtime',
            'faithfulchukwunwogor/beehive-qwen35-menu-adapter',
            'faithfulchukwunwogor/beehive-menu-probe-weights'],
        'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built mixed-environment two-hour probe notebook and Kaggle deployments')
