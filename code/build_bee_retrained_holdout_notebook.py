"""Build an evaluation-only four-seed notebook for the frozen retrained probe."""
from pathlib import Path
import json
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
base = nbf.read(ROOT / 'beehive_probe_retrained.ipynb', as_version=4)
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md('''# Frozen retrained-probe steering on seeds 100–103

This is an **evaluation-only** notebook. It loads the completed probe weights
and the signs, nonnegative combination weights, layers, and 0.045 strength
selected in the preceding experiment. No probe regression, classifier,
direction sign, direction weight, or steering-strength fit occurs here.

The preceding run did not export its two activation-norm scalars. This notebook
reconstructs those scalars exactly by replaying the same four logged source
arms and the same controlled counterfactual examples, then forwarding their
action-conditioned prompts through the unchanged model. It checks the exact
source replay and training-example counts. Only the two fixed scalars are
recovered; no probe weights or decisions are learned from these examples.

Four paired seeds, 100–103, then compare baseline, equal signed steering,
and learned-weight signed steering for two simulated hours, 400 bee decisions
per arm. Each bee still receives only its local observation and feasible menu.
The researcher-only 14-component hive state is snapshotted after every
decision. The notebook saves all final vectors, a timestamped step-change
transcript, paired score differences, and overlaid trajectories.

Seeds 100–103 were used in earlier experiments with other probe directions.
They were not used to fit the retrained probe or its new direction settings,
but they are not wholly untouched study seeds. Four paired seeds improve
precision over the preceding two-seed result; they do not establish a broad
biological or simulator-general effect.
''')

cells.append(nbf.v4.new_code_cell(base.cells[1].source))

imports = base.cells[2].source
imports = imports.replace('from sklearn.linear_model import Ridge, LogisticRegression\n', '')
imports = imports.replace("('bee_probe_retrained_'+", "('bee_retrained_holdout_'+")
imports = imports.replace(
    "PRIOR_REPORT=json.loads((PROBE_ROOT/'experiment_report.json').read_text())",
    "PRIOR_REPORT=json.loads((PROBE_ROOT/'experiment_report.json').read_text())\n"
    "TRAINED_ROOT=next(INPUT_ROOT.rglob('retrained_probe_layer_13.npz')).parent\n"
    "TRAINED_REPORT=json.loads((TRAINED_ROOT/'retrained_probe_report.json').read_text())",
)
code(imports)

code('''SMOKE=False
BATCH_SIZE=16
MAX_NEW_TOKENS=160
FINAL_SEEDS=(100,) if SMOKE else (100,101,102,103)
HOURS=0.10 if SMOKE else 2.0
SELECTED_LAYERS=tuple(TRAINED_REPORT['selected_layers'])
assert SELECTED_LAYERS==(13,14)
assert TRAINED_REPORT['final_strength']=={
    'new_equal_combination':0.045,'new_learned_combination':0.045}
FRACTION=0.045
POSITION='all'

model,processor=FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_DIR),max_seq_length=4096,dtype=None,
    load_in_4bit=False,load_in_16bit=True,full_finetuning=False)
FastLanguageModel.for_inference(model)
model.eval()

def decoder_candidates(model):
    found=[]
    for name,module in model.named_modules():
        if not isinstance(module,nn.ModuleList) or len(module)<8:continue
        if not any(hasattr(module[0],field) for field in
                   ('self_attn','linear_attn','mlp')):continue
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

ACTIVE=tuple(TRAINED_REPORT['representatives'])
W={layer:np.load(TRAINED_ROOT/f'retrained_probe_layer_{layer}.npz')['W']
   .astype(np.float32) for layer in SELECTED_LAYERS}
U={layer:W[layer][:,ACTIVE]/np.linalg.norm(W[layer][:,ACTIVE],axis=0)
   for layer in SELECTED_LAYERS}
assert len(ACTIVE)==8
activation_norms={}
print('Loaded frozen probe layers',SELECTED_LAYERS,'active',ACTIVE,
      'adapter',ADAPTER_DIR,'trained output',TRAINED_ROOT,'smoke',SMOKE)
''')

md('''## 1. Reconstruct the fixed activation norms without probe fitting

Source seeds 90 and 91 are replayed exactly, with baseline and old steered
worlds. The same targeted and controlled cases are constructed as in the
training notebook. In full mode the notebook asserts the prior count of
2,623 examples and every example-kind count. Neutral response activations
are pooled only at layers 13 and 14. Their median norms across the full
training examples are the two scalars used by the preceding intervention.
Smoke mode uses a small subset to check the code path; only full mode
reconstructs the final scalars.
''')

cells.append(nbf.v4.new_code_cell(base.cells[5].source))
cells.append(nbf.v4.new_code_cell(base.cells[6].source))
cells.append(nbf.v4.new_code_cell(base.cells[7].source))

code('''if not SMOKE:
    assert len(train_examples)==TRAINED_REPORT['training_examples']==2623
    assert dict(Counter(row['kind'] for row in train_examples))==\
           TRAINED_REPORT['training_kind_counts']
    assert all(row['max_vector_error']==0 for row in replay_rows)

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
norm_samples={layer:[] for layer in SELECTED_LAYERS}
started=time.perf_counter()
try:
    for index,row in enumerate(train_examples):
        enriched=dict(row['observation'],feasible_actions=feasible_actions(row['observation']))
        prompt=[{'role':'system','content':MENU_SYSTEM_PROMPT},
                {'role':'user','content':json.dumps(enriched,ensure_ascii=False,
                                                  separators=(',',':'),allow_nan=False)}]
        answer=json.dumps(row['action'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
        p=encoded_chat(prompt,True)
        f=encoded_chat(prompt+[{'role':'assistant','content':answer}],False)
        pids,fids=p['input_ids'][0],f['input_ids'][0]
        assert torch.equal(pids,fids[:len(pids)])
        response_span[:]=[len(pids),len(fids)-1]
        assert response_span[1]>response_span[0] and len(fids)<=4096
        pooled.clear()
        with torch.inference_mode():
            output=model(**{k:v.to(model.device) for k,v in f.items()},use_cache=False)
        del output
        for layer_i in SELECTED_LAYERS:
            norm_samples[layer_i].append(float(np.linalg.norm(pooled[layer_i])))
        if (index+1)%100==0:
            print('Norm reconstruction',index+1,'/',len(train_examples),'minutes',
                  round((time.perf_counter()-started)/60,1),flush=True)
finally:
    for handle in handles:handle.remove()
activation_norms.update({layer:float(np.median(norm_samples[layer]))
                         for layer in SELECTED_LAYERS})
assert all(np.isfinite(activation_norms[layer]) and activation_norms[layer]>0
           for layer in SELECTED_LAYERS)
norm_record={'source_training_examples':len(train_examples),
             'probe_layers':list(SELECTED_LAYERS),
             'activation_norms':activation_norms,
             'method':'median neutral teacher-forced response-feature norm',
             'source_replay_errors':[row['max_vector_error'] for row in replay_rows]}
(RUN_ROOT/'reconstructed_activation_norms.json').write_text(json.dumps(norm_record,indent=2))
print('Reconstructed fixed activation norms',activation_norms)
''')

md('''## 2. Load frozen signed directions and evaluate four paired seeds

The saved sign and weight dictionaries are loaded unchanged. Each normalized
probe column receives its saved sign before combination. The equal and learned
combinations are each renormalized at both selected layers, then injected at
the fixed 0.045 fraction of the reconstructed activation norms. The policy
and simulator use the same 50 bees, 15-minute rounds, batch size 16,
two-hour horizon, and frozen-world batch semantics as the prior experiment.
''')

code('''def build_combo(weight_by_index):
    weights=np.asarray([weight_by_index.get(j,0.0) for j in ACTIVE],dtype=np.float32)
    assert np.any(weights!=0)
    return {layer:normalized(U[layer]@weights) for layer in SELECTED_LAYERS}

signs={j:int(TRAINED_REPORT['chosen_signs'][VECTOR_NAMES[j]]) for j in ACTIVE}
equal_weights={j:signs[j] for j in ACTIVE}
learned_weights={j:signs[j]*float(TRAINED_REPORT['learned_weights'][VECTOR_NAMES[j]])
                 for j in ACTIVE}
EQUAL_DIR=build_combo(equal_weights)
LEARNED_DIR=build_combo(learned_weights)
assert all(abs(float(np.linalg.norm(direction))-1)<1e-5
           for combo in (EQUAL_DIR,LEARNED_DIR) for direction in combo.values())
print('Frozen signs',signs,'learned weights',learned_weights,
      'strength',FRACTION)

final_rows=[]
for seed in FINAL_SEEDS:
    baseline,_=run_arm(f'holdout_{seed}_baseline',seed,HOURS)
    baseline['arm']='baseline';baseline['changed_actions_vs_baseline']=0
    baseline['score_difference_vs_baseline']=0.0
    baseline['vector_difference_vs_baseline']=[0.0]*14
    final_rows.append(baseline)
    for name,direction in (('new_equal_combination',EQUAL_DIR),
                           ('new_learned_combination',LEARNED_DIR)):
        row,_=run_arm(f'holdout_{seed}_{name}',seed,HOURS,direction)
        row['arm']=name
        final_rows.append(annotate_changes(row,baseline))

summary=pd.DataFrame([{k:v for k,v in row.items() if k not in
                       ('initial_vector','final_vector','vector_difference_vs_baseline')}
                       for row in final_rows])
summary.to_csv(RUN_ROOT/'holdout_run_summary.csv',index=False)
pd.DataFrame([{'seed':row['seed'],'arm':row['arm'],'attribute':VECTOR_NAMES[j],
               'initial':row['initial_vector'][j],'final':row['final_vector'][j],
               'minus_baseline':row['vector_difference_vs_baseline'][j]}
              for row in final_rows for j in range(14)]).to_csv(
                  RUN_ROOT/'holdout_final_vectors.csv',index=False)
print(summary[['seed','arm','final_score','score_difference_vs_baseline',
               'changed_actions_vs_baseline']].to_string(index=False))
''')

md('''## 3. Export every vector step and overlay trajectories

The transcript contains each final arm's timestamp, action, immediate,
background, and total 14-component step change, and post-action state.
The notebook audits the arithmetic and saves three overlaid score
trajectories across the four paired seeds.
''')

code('''step_rows=[]
for row in final_rows:
    output_dir=Path(row['output_dir'])
    actions=[json.loads(line) for line in (output_dir/'actions.jsonl').open()]
    vectors=[json.loads(line) for line in (output_dir/'vectors.jsonl').open()]
    metadata=json.loads((output_dir/'metadata.json').read_text())
    assert metadata['completed'] and len(actions)==len(vectors)==row['actions']
    start=pd.Timestamp(metadata['initial_timestamp'])
    previous=np.asarray(metadata['initial_vector'],dtype=np.float64)
    for action,vector in zip(actions,vectors):
        immediate=np.asarray(action['immediate_action_delta'],dtype=np.float64)
        background=np.asarray(action['background_delta'],dtype=np.float64)
        current=np.asarray(vector['vector'],dtype=np.float64)
        assert immediate.shape==background.shape==current.shape==(14,)
        assert np.allclose(previous+immediate+background,current,rtol=0,atol=1e-8)
        assert np.all((0<=current)&(current<=1))
        item={'seed':row['seed'],'arm':row['arm'],'event_id':action['event_id'],
              'bee_id':action['bee_id'],'timestamp':action['timestamp'],
              'sim_hour':(pd.Timestamp(action['timestamp'])-start).total_seconds()/3600,
              'action':action['action']['action'],'outcome':action['outcome']['status'],
              'hive_score':float(current.mean())}
        for j,name in enumerate(VECTOR_NAMES):
            item['value_'+name]=float(current[j])
            item['immediate_'+name]=float(immediate[j])
            item['background_'+name]=float(background[j])
            item['total_'+name]=float(immediate[j]+background[j])
        step_rows.append(item);previous=current
    assert np.allclose(previous,row['final_vector'],rtol=0,atol=1e-8)
steps=pd.DataFrame(step_rows)
assert len(steps)==sum(row['actions'] for row in final_rows)
steps.to_csv(RUN_ROOT/'holdout_step_changes.csv.gz',index=False,compression='gzip')

import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
colors={'baseline':'#555555','new_equal_combination':'#087e8b',
        'new_learned_combination':'#c17c00'}
for arm,color in colors.items():
    part=steps[steps.arm==arm]
    mean=part.groupby('event_id').agg({'sim_hour':'mean','hive_score':'mean'})
    ax.plot(mean.sim_hour,mean.hive_score,label=arm,color=color,linewidth=1.5)
ax.set(xlabel='Simulated hours',ylabel='Mean 14-attribute hive score',
       title='Frozen retrained-probe steering on seeds 100–103')
ax.legend();ax.grid(alpha=0.2)
fig.savefig(RUN_ROOT/'holdout_mean_score_trajectories.png',dpi=150)
plt.close(fig)
''')

code('''def bootstrap_interval(values,draws=10000):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(121)
    means=rng.choice(values,size=(draws,len(values)),replace=True).mean(axis=1)
    return np.quantile(means,[0.025,0.975]).tolist()

paired={}
for arm in ('new_equal_combination','new_learned_combination'):
    rows=[row for row in final_rows if row['arm']==arm]
    differences=[row['score_difference_vs_baseline'] for row in rows]
    paired[arm]={'score_differences':differences,
                 'mean_score_difference':float(np.mean(differences)),
                 'bootstrap_95pct_interval':bootstrap_interval(differences),
                 'changed_actions':sum(row['changed_actions_vs_baseline'] for row in rows),
                 'mean_final_vector':np.mean([row['final_vector'] for row in rows],axis=0).tolist()}
report={'model':'continued Qwen3.5-0.8B bee menu LoRA',
        'probe_source':str(TRAINED_ROOT),
        'probe_layers':list(SELECTED_LAYERS),
        'reconstructed_activation_norms':activation_norms,
        'strength':FRACTION,'position':POSITION,
        'signs':dict((VECTOR_NAMES[j],signs[j]) for j in ACTIVE),
        'learned_weights':dict((VECTOR_NAMES[j],abs(learned_weights[j])) for j in ACTIVE),
        'no_probe_retraining_or_refitting':True,
        'seed_history_note':'100–103 were used in earlier experiments, not in retrained-probe fitting or calibration',
        'smoke':SMOKE,'seeds':list(FINAL_SEEDS),'hours':HOURS,
        'final_runs':final_rows,'paired':paired,'step_records':len(steps)}
(RUN_ROOT/'holdout_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('Paired summary',json.dumps({k:{'mean':v['mean_score_difference'],
                                    'interval':v['bootstrap_95pct_interval'],
                                    'changed_actions':v['changed_actions']}
                                   for k,v in paired.items()},indent=2))
print('Artifacts',RUN_ROOT)
''')


def write_notebook(path, smoke):
    copied=[nbf.v4.new_markdown_cell(c.source) if c.cell_type=='markdown'
            else nbf.v4.new_code_cell(c.source) for c in cells]
    if smoke:
        for cell in copied:
            if cell.cell_type=='code' and 'SMOKE=False' in cell.source:
                cell.source=cell.source.replace('SMOKE=False','SMOKE=True',1)
                break
    notebook=nbf.v4.new_notebook(cells=copied,metadata=base.metadata)
    path.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,path)


write_notebook(ROOT/'beehive_retrained_probe_holdout.ipynb',False)
for smoke,stage in ((True,'kernel_smoke'),(False,'kernel_full')):
    folder=ROOT/'kaggle_deploy/retrained_holdout'/stage
    folder.mkdir(parents=True,exist_ok=True)
    write_notebook(folder/'beehive_retrained_probe_holdout.ipynb',smoke)
    suffix='-smoke' if smoke else ''
    metadata={
        'id':'faithfulchukwunwogor/beehive-retrained-probe-holdout'+suffix,
        'title':'Beehive Retrained Probe Holdout'+(' Smoke' if smoke else ''),
        'code_file':'beehive_retrained_probe_holdout.ipynb',
        'language':'python','kernel_type':'notebook','is_private':'true',
        'enable_gpu':'true','enable_internet':'true','machine_shape':'',
        'dataset_sources':[
            'faithfulchukwunwogor/beehive-qwen35-runtime',
            'faithfulchukwunwogor/beehive-qwen35-menu-adapter'],
        'competition_sources':[],
        'kernel_sources':[
            'faithfulchukwunwogor/beehive-mixed-probe-two-hour',
            'faithfulchukwunwogor/beehive-probe-retrained'],
        'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built frozen retrained-probe four-seed evaluation notebooks')
