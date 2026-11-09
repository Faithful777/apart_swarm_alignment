"""Build one Kaggle experiment that retrains, validates, and steers a bee probe."""
from pathlib import Path
import json
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
base = nbf.read(ROOT / 'beehive_direction_audit.ipynb', as_version=4)
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


md('''# Retrain the beehive vector probe with targeted counterfactual actions

This notebook **trains a new 14-output probe**. It first replays four completed
source trajectories from the previous two-hour experiment exactly, recovering
the local observations that led to each logged action. The trajectories cover
baseline and steered worlds in both hours. Cloned source states supply
additional feasible actions that the policy rarely chose, including feeding,
grooming, cleaning, heating, and comb repair. A small set of explicitly marked
synthetic but physically valid states separates pollen from brood nutrition
and hygiene from pathogen suppression. These counterfactual actions are
executed by the simulator and receive their actual immediate 14-vector effect.

New seeds 110–112 create untouched baseline and steered two-hour worlds for
layer selection and probe validation. The notebook compares new and old dense
probe errors on those worlds. It then screens the new probe's signed attribute
directions on identical frozen decision batches, calibrates nonnegative
combination weights from paired batch outcomes, and tests baseline, equal,
and calibrated combinations on new two-hour hive seeds 114–115. All bee
observations remain local; global vector labels are researcher-only.

The immediate action effect remains the training target. Complete two-hour
worlds are the downstream test of whether these new directions help hive
state over time. The old LoRA bee policy is unchanged throughout.
''')
cells.append(nbf.v4.new_code_cell(base.cells[1].source))

imports = base.cells[2].source
imports = imports.replace('from collections import Counter, defaultdict',
                          'from collections import Counter, defaultdict')
imports = imports.replace('from beehive_sim import Config, World, Observer, Simulation, VECTOR_NAMES, weighted_research_score, validate_action',
                          'from beehive_sim import Config, World, Observer, Brood, Simulation, VECTOR_NAMES, weighted_research_score, validate_action')
imports = imports.replace('from unsloth import FastLanguageModel',
                          'from unsloth import FastLanguageModel\nfrom sklearn.linear_model import Ridge, LogisticRegression')
imports = imports.replace("('bee_direction_audit_'+", "('bee_probe_retrained_'+")
code(imports)

setup = base.cells[3].source
setup = setup.replace('VALIDATION_SEEDS=(110,) if SMOKE else (110,111,112,113)',
                      'VALIDATION_SEEDS=(110,) if SMOKE else (110,111,112)')
setup = setup.replace('FINAL_SEEDS=(114,) if SMOKE else (114,115)',
                      'FINAL_SEEDS=(114,) if SMOKE else (114,115)')
setup += "\nOLD_W={layer:W[layer].copy() for layer in SELECTED_LAYERS}\n" \
         "OLD_BIAS={layer:BIAS[layer].copy() for layer in SELECTED_LAYERS}\n" \
         "BOOTSTRAP_DIR={layer:direction.copy() for layer,direction in EQUAL_DIR.items()}\n"
code(setup)

md('''## 1. Replay completed source worlds and collect targeted labels

Replaying logged actions costs no model generation. The notebook checks that
every replayed 14-value vector equals the saved vector exactly. Source seeds
90 and 91 provide the training trajectories. At one frozen batch per round,
feasible candidate actions are executed in cloned worlds to create additional
real simulator labels. Synthetic variants remain inside training only and are
counted separately. A newly created nonreference larva allows a pollen-reserve
change without changing the reference brood-nutrition metric. Removing local
contamination before cleaning allows hygiene to change without pathogen
suppression. Cool brood cells add informative heating examples.
''')

cells.append(nbf.v4.new_code_cell(base.cells[7].source))

code('''class ReplayCapturePolicy:
    label='saved_action_replay_for_probe_training'
    def __init__(self,world,logged,collect):
        self.world=world;self.logged=logged;self.collect=collect
        self.position=0;self.calls=0;self.observations=[];self.checkpoints=[]
    def decide_batch(self,observations):
        self.observations.extend(copy.deepcopy(observations))
        keep=self.calls%4==0 and (not SMOKE or self.calls==0)
        if keep:
            self.checkpoints.append({'seed':self.world.config.seed,
                'arm':self.collect,'hour':1+int((self.world.minutes-1e-8)//60),
                'world':copy.deepcopy(self.world),'observations':copy.deepcopy(observations)})
        count=len(observations)
        chosen=[row['action'] for row in self.logged[self.position:self.position+count]]
        assert len(chosen)==count
        self.position+=count;self.calls+=1
        return chosen

train_examples=[];training_checkpoints=[];replay_rows=[]
for seed in (90,91):
    for arm in ('baseline','bootstrap_equal'):
        folder=PROBE_ROOT/f'source_{seed}_{arm}'
        metadata=json.loads((folder/'metadata.json').read_text())
        logged=[json.loads(line) for line in (folder/'actions.jsonl').open()]
        saved=[json.loads(line)['vector'] for line in (folder/'vectors.jsonl').open()]
        world=World(Config(**metadata['calibration']['config']))
        policy=ReplayCapturePolicy(world,logged,arm)
        sim=Simulation(world,policy)
        replayed=sim.run(decision_batch_size=BATCH_SIZE)
        assert len(replayed)==len(saved)==len(logged)==400
        error=float(np.max(np.abs(np.asarray([row['vector'] for row in replayed])-saved)))
        assert error==0.0,(seed,arm,error)
        replay_rows.append({'seed':seed,'arm':arm,'actions':len(logged),
                            'max_vector_error':error,'checkpoints':len(policy.checkpoints)})
        training_checkpoints.extend(policy.checkpoints)
        for i,(observation,audit) in enumerate(zip(policy.observations,sim.audit)):
            if SMOKE and i>=24:continue
            train_examples.append({'seed':seed,'arm':arm,'hour':1+i//200,
                'kind':'natural','observation':observation,'action':audit['action'],
                'target':audit['immediate_action_delta']})
pd.DataFrame(replay_rows).to_csv(RUN_ROOT/'source_replay_audit.csv',index=False)
print('Exact source replays',len(replay_rows),'natural training examples',
      len(train_examples),'frozen training batches',len(training_checkpoints))
''')

code('''def labeled_action(world,bee_id,action):
    before=np.asarray(Observer(world).snapshot(),dtype=np.float64)
    outcome=world.execute(bee_id,action,world.config.round_minutes)
    if outcome['status']!='applied':return None
    after=np.asarray(Observer(world).snapshot(),dtype=np.float64)
    return (after-before).tolist()

def targeted_examples(checkpoints,training):
    examples=[]
    quotas={'feed_brood':8,'groom':4,'heat':4,'clean':4,'build_comb':4}
    for checkpoint in checkpoints:
        world=checkpoint['world'];observations=checkpoint['observations']
        for kind,limit in quotas.items():
            selected=[]
            for observation in observations:
                action=next((a for a in feasible_actions(observation)
                             if a['action']==kind),None)
                if action is not None:selected.append((observation,action))
            for observation,action in selected[:limit]:
                clone=copy.deepcopy(world)
                target=labeled_action(clone,observation['self']['id'],action)
                if target is None:continue
                if kind=='heat' and abs(target[8])<1e-10:continue
                examples.append({'seed':checkpoint['seed'],'arm':checkpoint.get('arm','baseline'),
                    'hour':checkpoint['hour'],'kind':'targeted_'+kind,
                    'observation':observation,'action':action,'target':target})
        if not training:continue
        # Controlled states are training augmentation only. Every label is
        # calculated by actually executing the candidate action in its clone.
        brood_bees=[o for o in observations if o['current_cell']['zone']=='brood'][:4]
        for index,original in enumerate(brood_bees):
            bee_id=original['self']['id'];cell_id=original['self']['cell']
            clone=copy.deepcopy(world)
            new_id=f'counterfactual-larva-{checkpoint["seed"]}-{index}'
            clone.brood[new_id]=Brood(new_id,cell_id,initial=False,hunger=0.8)
            observation=clone.observe(bee_id)
            action=next((a for a in feasible_actions(observation)
                         if a['action']=='feed_brood' and a['target']==new_id),None)
            if action is not None:
                target=labeled_action(clone,bee_id,action)
                if target is not None and target[4]<-1e-10 and abs(target[7])<1e-10:
                    examples.append({'seed':checkpoint['seed'],'arm':checkpoint.get('arm','baseline'),
                        'hour':checkpoint['hour'],'kind':'synthetic_pollen_only',
                        'observation':observation,'action':action,'target':target})
        for original in observations[:4]:
            bee_id=original['self']['id'];clone=copy.deepcopy(world)
            cell=clone.cells[clone.bees[bee_id].cell]
            if cell.zone=='outside' or cell.debris<=1e-5:continue
            cell.pathogen=0.0
            observation=clone.observe(bee_id)
            action=next((a for a in feasible_actions(observation)
                         if a['action']=='clean'),None)
            if action is None:continue
            target=labeled_action(clone,bee_id,action)
            if target is not None and target[10]>1e-10 and abs(target[12])<1e-10:
                examples.append({'seed':checkpoint['seed'],'arm':checkpoint.get('arm','baseline'),
                    'hour':checkpoint['hour'],'kind':'synthetic_hygiene_only',
                    'observation':observation,'action':action,'target':target})
        for original in brood_bees:
            bee_id=original['self']['id'];clone=copy.deepcopy(world)
            clone.cells[clone.bees[bee_id].cell].temperature_c=30.0
            observation=clone.observe(bee_id)
            action=next((a for a in feasible_actions(observation)
                         if a['action']=='heat'),None)
            if action is None:continue
            target=labeled_action(clone,bee_id,action)
            if target is not None and target[8]>1e-10:
                examples.append({'seed':checkpoint['seed'],'arm':checkpoint.get('arm','baseline'),
                    'hour':checkpoint['hour'],'kind':'synthetic_cool_brood_heat',
                    'observation':observation,'action':action,'target':target})
    return examples

augmented=targeted_examples(training_checkpoints,training=True)
train_examples.extend(augmented)
print('Training examples',len(train_examples),'augmentation',
      dict(Counter(row['kind'] for row in augmented)))
if not SMOKE:
    assert set(row['hour'] for row in train_examples)=={1,2}
    assert {'baseline','bootstrap_equal'}<=set(row['arm'] for row in train_examples)
    assert sum(row['kind']=='synthetic_pollen_only' for row in augmented)>=20
    assert sum(row['kind']=='synthetic_hygiene_only' for row in augmented)>=20
''')

md('''## 2. Create new held-out worlds before fitting the probe

Seeds 110–112 are not in probe training. Their baseline and previously
steered trajectories each cover two hours. Seed 110 is used only to choose
layers and diagnostic thresholds. Seeds 111–112 report prediction error.
The screen later uses frozen baseline states from seeds 110–111, while final
hive seeds 114–115 remain untouched until direction selection is complete.
''')

fresh = base.cells[8].source
fresh = fresh.replace('baseline_policy=CheckpointPolicy(world,seed in SCREEN_SEEDS)',
                      'baseline_policy=CheckpointPolicy(world,True)')
fresh = fresh.replace("EQUAL_DIR,capture=True", "BOOTSTRAP_DIR,capture=True")
fresh = fresh.replace("'equal_normalized'", "'bootstrap_equal'")
fresh = fresh.replace("f'validation_{seed}_equal'", "f'validation_{seed}_bootstrap'")
fresh = fresh.replace("'baseline','bootstrap_equal'", "'baseline','bootstrap_equal'")
fresh = fresh.replace("print('Frozen batches',len(checkpoints),'neutral probe examples',len(source_rows))",
    "fresh_checkpoints=list(checkpoints)\n"
    "eval_examples=[dict(row,kind='natural') for row in source_rows]\n"
    "eval_examples.extend(targeted_examples(fresh_checkpoints,training=False))\n"
    "checkpoints=[cp for cp in checkpoints if cp['seed'] in SCREEN_SEEDS]\n"
    "print('Frozen screen batches',len(checkpoints),'held-out examples',len(eval_examples))")
code(fresh)

md('''## 3. Fit and evaluate a new 14-output probe

Neutral teacher-forced response activations are extracted from all text
decoder layers for training and held-out examples. Per-attribute target scales
are computed from training data only. Rare changed attributes receive capped
row weights, and all fourteen outputs are fit together with ridge regression.
Layer selection uses seed 110. The independent report uses seeds 111–112 and
compares dense prediction error with the old probe and an always-zero
control, separately for natural and targeted examples and both hours.
''')

code('''all_examples=train_examples+eval_examples
N_TRAIN=len(train_examples)
FEATURE_LAYERS=tuple(range(len(DECODER_LAYERS)))
def encoded_chat(messages,add_generation_prompt):
    enc=processor.apply_chat_template(messages,tokenize=True,
        add_generation_prompt=add_generation_prompt,return_tensors='pt',return_dict=True)
    return {k:v for k,v in enc.items() if isinstance(v,torch.Tensor)}

pooled={};response_span=[0,0];handles=[]
for layer_i in FEATURE_LAYERS:
    def capture(_module,_inputs,output,layer_i=layer_i):
        hidden=output[0] if isinstance(output,(tuple,list)) else output
        start,end=response_span
        pooled[layer_i]=hidden[0,start:end].float().mean(dim=0).detach().cpu().numpy()
    handles.append(DECODER_LAYERS[layer_i].register_forward_hook(capture))
features=None;started=time.perf_counter()
try:
    for index,row in enumerate(all_examples):
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
        if features is None:
            features=np.empty((len(all_examples),len(FEATURE_LAYERS),len(pooled[0])),
                              dtype=np.float32)
        for layer_i in FEATURE_LAYERS:features[index,layer_i]=pooled[layer_i]
        if (index+1)%100==0:
            print('Neutral features',index+1,'/',len(all_examples),'minutes',
                  round((time.perf_counter()-started)/60,1),flush=True)
finally:
    for handle in handles:handle.remove()
assert np.isfinite(features).all()
targets=np.asarray([row['target'] for row in all_examples],dtype=np.float32)
assert targets.shape==(len(all_examples),14)
train_pos=np.arange(N_TRAIN)
if SMOKE:
    held=np.arange(N_TRAIN,len(all_examples))
    screen_pos,report_pos=held[::2],held[1::2]
else:
    screen_pos=np.asarray([i for i,row in enumerate(all_examples)
                           if i>=N_TRAIN and row['seed']==110])
    report_pos=np.asarray([i for i,row in enumerate(all_examples)
                           if i>=N_TRAIN and row['seed'] in (111,112)])
assert len(train_pos)>0 and len(screen_pos)>0 and len(report_pos)>0
print('Feature matrix',features.shape,'splits',len(train_pos),len(screen_pos),len(report_pos))
''')

code('''SCALES=[]
for j in range(14):
    nz=np.abs(targets[train_pos,j]);nz=nz[nz>1e-10]
    SCALES.append(max(float(np.percentile(nz,75)),1e-4) if len(nz) else 1.0)
SCALES=np.asarray(SCALES,dtype=np.float32)
train_counts=np.count_nonzero(np.abs(targets[train_pos])>1e-10,axis=0)
rarity=np.minimum(15.0,len(train_pos)/np.maximum(train_counts,1))
changed=np.abs(targets[train_pos])>1e-10
row_weights=np.maximum(1.0,(changed*rarity[None,:]).max(axis=1))

def weighted_scaled_mae(pred,truth):
    weight=np.where(np.abs(truth)>1e-10,4.0,1.0)
    return float(np.sum(weight*np.abs((pred-truth)/SCALES))/np.sum(weight))

def rare_changed_mae(pred,truth):
    scores=[]
    for j in ACTIVE:
        use=np.abs(truth[:,j])>1e-10
        if use.any():scores.append(float(np.mean(np.abs((pred[use,j]-truth[use,j])/SCALES[j]))))
    return float(np.mean(scores)) if scores else weighted_scaled_mae(pred,truth)

layer_models={};layer_rows=[]
for layer_i in FEATURE_LAYERS:
    x=features[train_pos,layer_i]
    mu=x.mean(axis=0);sd=np.maximum(x.std(axis=0),1e-3)
    ridge=Ridge(alpha=100.0,solver='lsqr',max_iter=1000)
    ridge.fit((x-mu)/sd,targets[train_pos]/SCALES,sample_weight=row_weights)
    pred=ridge.predict((features[screen_pos,layer_i]-mu)/sd)*SCALES
    score=0.5*weighted_scaled_mae(pred,targets[screen_pos])+\
          0.5*rare_changed_mae(pred,targets[screen_pos])
    layer_rows.append({'layer':layer_i,'selection_score':score,
                       'weighted_scaled_mae':weighted_scaled_mae(pred,targets[screen_pos]),
                       'rare_changed_mae':rare_changed_mae(pred,targets[screen_pos])})
    layer_models[layer_i]=(ridge,mu,sd)
layer_table=pd.DataFrame(layer_rows).sort_values('selection_score')
SELECTED_LAYERS=tuple(layer_table.head(2).layer.astype(int).tolist())
layer_table.to_csv(RUN_ROOT/'new_layer_screen.csv',index=False)
print('New selected layers',SELECTED_LAYERS,layer_table.head(5).to_dict('records'))
activation_norms.update({layer:float(np.median(np.linalg.norm(
    features[train_pos,layer],axis=1))) for layer in SELECTED_LAYERS})
assert all(np.isfinite(activation_norms[layer]) and activation_norms[layer]>0
           for layer in SELECTED_LAYERS)

W={};BIAS={};U={};new_predictions={}
for layer in SELECTED_LAYERS:
    ridge,mu,sd=layer_models[layer]
    W[layer]=(ridge.coef_.T/sd[:,None])*SCALES[None,:]
    BIAS[layer]=(ridge.intercept_-(mu/sd)@ridge.coef_.T)*SCALES
    norm=np.linalg.norm(W[layer][:,ACTIVE],axis=0)
    assert np.all(np.isfinite(norm)) and np.all(norm>1e-10)
    U[layer]=W[layer][:,ACTIVE]/norm
    new_predictions[layer]=features[:,layer]@W[layer]+BIAS[layer]
    np.savez(RUN_ROOT/f'retrained_probe_layer_{layer}.npz',W=W[layer],bias=BIAS[layer],
             active_indices=np.asarray(ACTIVE),train_change_counts=train_counts,
             target_scales=SCALES)
    xtrain=(features[train_pos,layer]-mu)/sd
    mask_coef=np.zeros((14,xtrain.shape[1]),dtype=np.float32)
    mask_bias=np.zeros(14,dtype=np.float32)
    for j in range(14):
        y=(np.abs(targets[train_pos,j])>1e-10).astype(int)
        if len(np.unique(y))<2:
            mask_bias[j]=10.0 if y[0] else -10.0
            continue
        clf=LogisticRegression(C=0.1,class_weight='balanced',max_iter=300,
                               solver='liblinear',random_state=271)
        clf.fit(xtrain,y)
        mask_coef[j]=clf.coef_[0];mask_bias[j]=clf.intercept_[0]
    np.savez(RUN_ROOT/f'retrained_mask_layer_{layer}.npz',
             coef=mask_coef,bias=mask_bias,feature_mean=mu,feature_sd=sd,
             threshold=0.5)

probe_rows=[]
for layer in SELECTED_LAYERS:
    for kind in ('natural','targeted'):
        for hour in (1,2):
            if SMOKE and hour==2:continue
            for arm in ('baseline','bootstrap_equal'):
                positions=[i for i in report_pos if all_examples[i]['hour']==hour and
                    all_examples[i]['arm']==arm and
                    (all_examples[i]['kind']=='natural' if kind=='natural' else
                     all_examples[i]['kind'].startswith('targeted_'))]
                if not positions:continue
                truth=targets[positions];pred=new_predictions[layer][positions]
                result={'layer':layer,'kind':kind,'hour':hour,'arm':arm,
                    'examples':len(positions),'nonzero_target_entries':int(
                        np.count_nonzero(np.abs(truth)>1e-10)),
                    'new_weighted_scaled_mae':weighted_scaled_mae(pred,truth),
                    'zero_control_mae':weighted_scaled_mae(np.zeros_like(truth),truth),
                    'new_rare_changed_mae':rare_changed_mae(pred,truth)}
                if layer in OLD_W:
                    old=features[positions,layer]@OLD_W[layer]+OLD_BIAS[layer]
                    result['old_weighted_scaled_mae']=weighted_scaled_mae(old,truth)
                    result['old_rare_changed_mae']=rare_changed_mae(old,truth)
                probe_rows.append(result)
pd.DataFrame(probe_rows).to_csv(RUN_ROOT/'retrained_probe_validation.csv',index=False)
print('Training changed labels',dict(zip(VECTOR_NAMES,train_counts.tolist())))
print(pd.DataFrame(probe_rows).to_string(index=False))

old_rows=[]
for layer in OLD_W:
    for kind in ('natural','targeted'):
        for hour in (1,2):
            if SMOKE and hour==2:continue
            positions=[i for i in report_pos if all_examples[i]['hour']==hour and
                (all_examples[i]['kind']=='natural' if kind=='natural' else
                 all_examples[i]['kind'].startswith('targeted_'))]
            if not positions:continue
            truth=targets[positions]
            prediction=features[positions,layer]@OLD_W[layer]+OLD_BIAS[layer]
            old_rows.append({'layer':layer,'kind':kind,'hour':hour,
                'examples':len(positions),
                'weighted_scaled_mae':weighted_scaled_mae(prediction,truth),
                'rare_changed_mae':rare_changed_mae(prediction,truth)})
pd.DataFrame(old_rows).to_csv(RUN_ROOT/'old_probe_validation.csv',index=False)
print('Old probe comparison',pd.DataFrame(old_rows).to_string(index=False))
''')

md('''## 4. Audit new column geometry and calibrate directions causally

The eight original active attributes remain eligible. Near-duplicate new
columns are collapsed only if their absolute cosine exceeds 0.98 at both
selected layers. The screen tests both signs of each remaining axis at
strength 0.03 on the same frozen decision batches. It measures exact changed
actions and paired post-batch hive-vector effects. The sign with the better
measured effect is selected for each axis. Nonnegative softmax weights are
learned from these paired effects and shrunk 25% toward equal weights. The
resulting equal and calibrated combinations are screened too. Final hive
seeds are never used to choose a sign, weight, layer, or strength.
''')

geometry = base.cells[5].source
code(geometry)

code('''def build_combo(weight_by_index):
    weights=np.asarray([weight_by_index.get(j,0.0) for j in ACTIVE],dtype=np.float32)
    assert np.any(weights!=0)
    return {layer:normalized(U[layer]@weights) for layer in SELECTED_LAYERS}

representatives=[]
for j in ACTIVE:
    duplicate=False
    for existing in representatives:
        if all(abs(float(U[layer][:,ACTIVE.index(j)]@
                         U[layer][:,ACTIVE.index(existing)]))>0.98
               for layer in SELECTED_LAYERS):
            duplicate=True;break
    if not duplicate:representatives.append(j)
print('Independent representative attributes',[(j,VECTOR_NAMES[j]) for j in representatives])

def single_direction(index,sign):
    column=ACTIVE.index(index)
    return {layer:U[layer][:,column].copy()*sign for layer in SELECTED_LAYERS}

def branch_batch(checkpoint,actions):
    clone=copy.deepcopy(checkpoint['world']);observer=Observer(clone)
    before=np.asarray(observer.snapshot(),dtype=np.float64)
    for observation,action in zip(checkpoint['observations'],actions):
        clone.execute(observation['self']['id'],action,
                      min(clone.config.round_minutes,HOURS*60))
    after=np.asarray(observer.snapshot(),dtype=np.float64)
    return after-before,after

baseline_branches={}
for ci,checkpoint in enumerate(checkpoints):
    delta,after=branch_batch(checkpoint,checkpoint['baseline_actions'])
    assert np.allclose(after,checkpoint['expected_after'],rtol=0,atol=1e-8)
    baseline_branches[ci]=delta

screen_rows=[];screen_decisions=[]
def screen_direction(name,directions,strength):
    global FRACTION
    FRACTION=float(strength)
    policy=ColumnSteeredPolicy(directions)
    effects=[];changed=0;positive=negative=0
    for ci,checkpoint in enumerate(checkpoints):
        actions=policy.decide_batch(checkpoint['observations'])
        delta,_=branch_batch(checkpoint,actions)
        effect=float(np.mean(delta-baseline_branches[ci]))
        effects.append(effect);positive+=effect>1e-10;negative+=effect< -1e-10
        for observation,original,action in zip(checkpoint['observations'],
                                                checkpoint['baseline_actions'],actions):
            if action==original:continue
            changed+=1
            screen_decisions.append({'candidate':name,'strength':strength,
                'seed':checkpoint['seed'],'hour':checkpoint['hour'],
                'bee_id':observation['self']['id'],
                'baseline_action':original['action'],'steered_action':action['action'],
                'batch_score_effect':effect})
    row={'candidate':name,'strength':float(strength),'batches':len(checkpoints),
         'decisions':sum(len(cp['observations']) for cp in checkpoints),
         'changed_actions':changed,'mean_batch_score_effect':float(np.mean(effects)),
         'positive_batches':int(positive),'negative_batches':int(negative)}
    screen_rows.append(row)
    print('Screen',name,strength,'changed',changed,'mean',
          round(row['mean_batch_score_effect'],8),flush=True)
    return row

axis_results={}
for j in representatives:
    axis_results[j]={}
    for sign,label in ((1,'plus'),(-1,'minus')):
        name=f'{VECTOR_NAMES[j]}_{label}'
        axis_results[j][sign]=screen_direction(name,single_direction(j,sign),0.03)

chosen_sign={j:max(axis_results[j],key=lambda s:axis_results[j][s]['mean_batch_score_effect'])
             for j in representatives}
effects=np.asarray([axis_results[j][chosen_sign[j]]['mean_batch_score_effect']
                    for j in representatives],dtype=np.float64)
scale=max(float(np.std(effects)),1e-5)
logits=2*(effects-effects.mean())/scale
soft=np.exp(logits-logits.max());soft/=soft.sum()
learned=0.25/len(representatives)+0.75*soft
equal_weights={j:chosen_sign[j] for j in representatives}
learned_weights={j:chosen_sign[j]*float(w) for j,w in zip(representatives,learned)}
EQUAL_DIR=build_combo(equal_weights)
LEARNED_DIR=build_combo(learned_weights)
screen_direction('new_equal_combination',EQUAL_DIR,0.03)
screen_direction('new_learned_combination',LEARNED_DIR,0.03)
for name,direction in (('new_equal_combination',EQUAL_DIR),
                       ('new_learned_combination',LEARNED_DIR)):
    for strength in (0.015,0.045):screen_direction(name,direction,strength)
pd.DataFrame(screen_rows).to_csv(RUN_ROOT/'new_causal_screen.csv',index=False)
pd.DataFrame(screen_decisions).to_csv(RUN_ROOT/'new_causal_changed_decisions.csv',index=False)
print('Chosen signs',dict((VECTOR_NAMES[j],chosen_sign[j]) for j in representatives))
print('Learned weights',dict((VECTOR_NAMES[j],float(w))
                             for j,w in zip(representatives,learned)))
''')

md('''## 5. Two-hour fresh-hive comparison and vector transcripts

For each combination, the strength with the highest paired screen score is
fixed before new final seeds 114–115 run. The final arms are baseline, new
equal-weight signed directions, and new outcome-calibrated signed directions.
Every action retains a timestamped 14-component post-action hive vector.
The notebook exports all final step changes and overlays all three score
trajectories on one plot.
''')

code('''final_strength={}
for name in ('new_equal_combination','new_learned_combination'):
    rows=[row for row in screen_rows if row['candidate']==name]
    winner=max(rows,key=lambda row:row['mean_batch_score_effect'])
    final_strength[name]=winner['strength']
print('Fixed final strengths',final_strength)

final_rows=[]
for seed in FINAL_SEEDS:
    baseline,_=run_arm(f'final_{seed}_baseline',seed,HOURS)
    baseline['arm']='baseline';baseline['changed_actions_vs_baseline']=0
    baseline['score_difference_vs_baseline']=0.0
    baseline['vector_difference_vs_baseline']=[0.0]*14
    final_rows.append(baseline)
    for name,direction in (('new_equal_combination',EQUAL_DIR),
                           ('new_learned_combination',LEARNED_DIR)):
        FRACTION=final_strength[name]
        row,_=run_arm(f'final_{seed}_{name}',seed,HOURS,direction)
        row['arm']=name
        final_rows.append(annotate_changes(row,baseline))
pd.DataFrame([{k:v for k,v in row.items() if k not in
              ('initial_vector','final_vector','vector_difference_vs_baseline')}
              for row in final_rows]).to_csv(RUN_ROOT/'final_hive_summary.csv',index=False)
pd.DataFrame([{'seed':row['seed'],'arm':row['arm'],'attribute':VECTOR_NAMES[j],
               'initial':row['initial_vector'][j],'final':row['final_vector'][j],
               'minus_baseline':row['vector_difference_vs_baseline'][j]}
              for row in final_rows for j in range(14)]).to_csv(
                  RUN_ROOT/'final_hive_vectors.csv',index=False)

step_rows=[]
for row in final_rows:
    actions=[json.loads(line) for line in (Path(row['output_dir'])/'actions.jsonl').open()]
    vectors=[json.loads(line) for line in (Path(row['output_dir'])/'vectors.jsonl').open()]
    start=pd.Timestamp(json.loads((Path(row['output_dir'])/'metadata.json').read_text())
                       ['initial_timestamp'])
    for action,vector in zip(actions,vectors):
        item={'seed':row['seed'],'arm':row['arm'],'event_id':action['event_id'],
              'timestamp':action['timestamp'],
              'sim_hour':(pd.Timestamp(action['timestamp'])-start).total_seconds()/3600,
              'action':action['action']['action'],'outcome':action['outcome']['status']}
        immediate=np.asarray(action['immediate_action_delta']);
        background=np.asarray(action['background_delta'])
        for j,name in enumerate(VECTOR_NAMES):
            item['value_'+name]=vector['vector'][j]
            item['immediate_'+name]=immediate[j]
            item['background_'+name]=background[j]
            item['total_'+name]=float(immediate[j]+background[j])
        item['hive_score']=float(np.mean(vector['vector']))
        step_rows.append(item)
steps=pd.DataFrame(step_rows)
assert len(steps)==sum(row['actions'] for row in final_rows)
steps.to_csv(RUN_ROOT/'final_step_changes.csv.gz',index=False,compression='gzip')

import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
colors={'baseline':'#555555','new_equal_combination':'#087e8b',
        'new_learned_combination':'#c17c00'}
for arm in colors:
    selected=steps[steps.arm==arm]
    mean=selected.groupby('event_id').agg({'sim_hour':'mean','hive_score':'mean'})
    ax.plot(mean.sim_hour,mean.hive_score,label=arm,color=colors[arm],linewidth=1.5)
ax.set(xlabel='Simulated hours',ylabel='Mean 14-attribute hive score',
       title='Fresh two-hour hive trajectories under retrained-probe steering')
ax.legend();ax.grid(alpha=0.2)
fig.savefig(RUN_ROOT/'mean_final_score_trajectories.png',dpi=150)
plt.close(fig)
print('Final score rows',[(r['seed'],r['arm'],round(r['final_score'],7),
                           round(r['score_difference_vs_baseline'],7)) for r in final_rows])
''')

code('''report={'smoke':SMOKE,'model':'continued Qwen3.5-0.8B bee menu LoRA',
        'training_source_seeds':[90,91],
        'fresh_validation_seeds':list(VALIDATION_SEEDS),
        'final_seeds':list(FINAL_SEEDS),'hours':HOURS,
        'source_replay':replay_rows,
        'training_examples':len(train_examples),
        'training_kind_counts':dict(Counter(row['kind'] for row in train_examples)),
        'training_change_counts':dict(zip(VECTOR_NAMES,train_counts.tolist())),
        'probe_splits':{'train':len(train_pos),'screen':len(screen_pos),
                        'report':len(report_pos)},
        'selected_layers':list(SELECTED_LAYERS),'layer_screen':layer_rows,
        'probe_validation':probe_rows,'old_probe_validation':old_rows,
        'direction_cosines':geometry,
        'representatives':representatives,
        'chosen_signs':dict((VECTOR_NAMES[j],chosen_sign[j]) for j in representatives),
        'learned_weights':dict((VECTOR_NAMES[j],float(w))
                               for j,w in zip(representatives,learned)),
        'causal_screen':screen_rows,'final_strength':final_strength,
        'final_runs':final_rows,'final_step_records':len(steps)}
(RUN_ROOT/'retrained_probe_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('Artifacts',RUN_ROOT)
''')


def write_notebook(path, smoke):
    copied = [nbf.v4.new_markdown_cell(cell.source) if cell.cell_type == 'markdown'
              else nbf.v4.new_code_cell(cell.source) for cell in cells]
    if smoke:
        for cell in copied:
            if cell.cell_type == 'code' and 'SMOKE=False' in cell.source:
                cell.source = cell.source.replace('SMOKE=False', 'SMOKE=True', 1)
                break
    notebook = nbf.v4.new_notebook(cells=copied, metadata=base.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)


write_notebook(ROOT / 'beehive_probe_retrained.ipynb', False)
for smoke, stage in ((True, 'kernel_smoke'), (False, 'kernel_full')):
    folder = ROOT / 'kaggle_deploy/probe_retrain' / stage
    folder.mkdir(parents=True, exist_ok=True)
    write_notebook(folder / 'beehive_probe_retrained.ipynb', smoke)
    suffix = '-smoke' if smoke else ''
    metadata = {
        'id': 'faithfulchukwunwogor/beehive-probe-retrained' + suffix,
        'title': 'Beehive Probe Retrained' + (' Smoke' if smoke else ''),
        'code_file': 'beehive_probe_retrained.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': 'true',
        'enable_gpu': 'true', 'enable_internet': 'true', 'machine_shape': '',
        'dataset_sources': [
            'faithfulchukwunwogor/beehive-qwen35-runtime',
            'faithfulchukwunwogor/beehive-qwen35-menu-adapter',
        ],
        'competition_sources': [],
        'kernel_sources': ['faithfulchukwunwogor/beehive-mixed-probe-two-hour'],
        'model_sources': [],
    }
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print('Built retrained vector probe and final steering notebooks')
