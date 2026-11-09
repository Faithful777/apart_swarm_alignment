"""Derive the four-simulated-hour bee steering notebook from the tested design."""
from pathlib import Path
import json
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
source = ROOT / 'beehive_column_steering.ipynb'
original = nbf.read(source, as_version=4)


def make_notebook(smoke: bool):
    cells = []
    for old in original.cells:
        cell = (nbf.v4.new_markdown_cell(old.source) if old.cell_type == 'markdown'
                else nbf.v4.new_code_cell(old.source))
        cells.append(cell)

    cells[0].source = cells[0].source.replace(
        '# Individual and combined hive-attribute steering',
        '# Four-hour individual and combined hive-attribute steering')
    cells[0].source = cells[0].source.replace(
        'an equal combination of normalized columns, the earlier raw-column\nsum, and a combination',
        'an equal combination of normalized columns and a combination')
    cells[0].source = cells[0].source.replace(
        'The main evaluation uses new seeds 70–73.',
        'The main evaluation uses seeds 70–73 for four simulated hours per arm.')
    cells[0].source += ('\n\nThis rerun excludes the previous control arm. It saves '
                        'the per-action change in every hive-vector attribute, '
                        'separating immediate action effects from background drift.')

    changed_hours = changed_raw = False
    for cell in cells:
        if cell.cell_type == 'code':
            if 'FINAL_HOURS=0.10 if SMOKE else 1.00' in cell.source:
                cell.source = cell.source.replace(
                    'FINAL_HOURS=0.10 if SMOKE else 1.00',
                    'FINAL_HOURS=0.10 if SMOKE else 4.00')
                changed_hours = True
            raw_line = "DIRS['raw_sum']={layer:normalized(W[layer][:,ACTIVE].sum(axis=1)) for layer in SELECTED_LAYERS}\n"
            if raw_line in cell.source:
                cell.source = cell.source.replace(raw_line, '')
                changed_raw = True
            cell.source = cell.source.replace(
                "    cosine=float(DIRS['equal_normalized'][layer]@DIRS['raw_sum'][layer])\n",
                '')
            cell.source = cell.source.replace(
                "    print('Layer',layer,'active columns',[(int(j),VECTOR_NAMES[j],round(float(norms[layer][j]),5)) for j in ACTIVE],\n"
                "          'equal-vs-raw cosine',round(cosine,4))",
                "    print('Layer',layer,'active columns',[(int(j),VECTOR_NAMES[j],round(float(norms[layer][j]),5)) for j in ACTIVE])")
            if smoke and 'SMOKE=False' in cell.source:
                cell.source = cell.source.replace('SMOKE=False', 'SMOKE=True', 1)
        else:
            cell.source = cell.source.replace(
                'the\nraw, equal-normalized, and individually steered arms',
                'the\nequal-normalized and individually steered arms')
            cell.source = cell.source.replace(
                'The six individual directions, equal-normalized sum, and raw-column sum are\ntested',
                'The six individual directions and equal-normalized sum are\ntested')
            cell.source = cell.source.replace(
                'All arms run for one simulated hour, with 200 action snapshots each.',
                'All arms run for four simulated hours, with 800 action snapshots each.')
    assert changed_hours and changed_raw
    leftovers=[(i,cell.source[:150]) for i,cell in enumerate(cells)
               if 'raw_sum' in cell.source or 'raw-column sum' in cell.source]
    assert not leftovers, f'Remove the raw-sum control everywhere: {leftovers}'

    cells.append(nbf.v4.new_markdown_cell('''## 5. Save every attribute step change and plot it against simulated time

For every action in every calibration and final arm, the recorded total step
change equals the previous post-action vector to current post-action vector
change. It is decomposed into background drift before the bee acts and the
immediate effect of that bee's action. The CSV contains all fourteen values for
each component, the post-action state, the bee, event number, timestamp,
simulated hour, seed, and arm. These columns are researcher-only and never
enter a bee's model prompt.

The plots show nonzero per-action total changes, state trajectories, and the
overall hive-score trajectory for the baseline, equal-normalized, and
outcome-weighted arms on the same axes. Change `PLOT_ARMS`
below to inspect any of the six individual directions. Figures for every final
seed are saved; the first seed's figures are also shown in the notebook.'''))

    cells.append(nbf.v4.new_code_cell('''import matplotlib.pyplot as plt
from IPython.display import display

step_rows=[]
all_runs=[('calibration',row) for row in calibration]+[('final',row) for row in final_runs]
for phase,row in all_runs:
    output_dir=Path(row['output_dir'])
    actions=[json.loads(line) for line in (output_dir/'actions.jsonl').open()]
    vectors=[json.loads(line) for line in (output_dir/'vectors.jsonl').open()]
    metadata=json.loads((output_dir/'metadata.json').read_text())
    assert len(actions)==len(vectors)==row['actions'] and metadata['completed']
    start=pd.Timestamp(metadata['initial_timestamp'])
    previous=np.asarray(metadata['initial_vector'],dtype=np.float64)
    for action,vector_row in zip(actions,vectors):
        immediate=np.asarray(action['immediate_action_delta'],dtype=np.float64)
        background=np.asarray(action['background_delta'],dtype=np.float64)
        total=immediate+background
        current=np.asarray(vector_row['vector'],dtype=np.float64)
        assert immediate.shape==background.shape==total.shape==current.shape==(14,)
        assert np.allclose(previous+total,current,rtol=0,atol=1e-8)
        timestamp=pd.Timestamp(action['timestamp'])
        item={'phase':phase,'seed':row['seed'],'arm':row['arm'],
              'event_id':action['event_id'],'bee_id':action['bee_id'],
              'timestamp':action['timestamp'],
              'sim_hour':(timestamp-start).total_seconds()/3600.0,
              'action':action['action']['action'],
              'outcome_status':action['outcome']['status'],
              'total_score_delta':float(total.mean()),
              'immediate_score_delta':float(immediate.mean()),
              'background_score_delta':float(background.mean())}
        for j,name in enumerate(VECTOR_NAMES):
            item[f'total_{name}']=float(total[j])
            item[f'immediate_{name}']=float(immediate[j])
            item[f'background_{name}']=float(background[j])
            item[f'value_{name}']=float(current[j])
        step_rows.append(item)
        previous=current
    assert np.allclose(previous,row['final_vector'],rtol=0,atol=1e-8)

steps=pd.DataFrame(step_rows)
assert len(steps)==sum(row['actions'] for _,row in all_runs)
steps['hive_score']=steps[[f'value_{name}' for name in VECTOR_NAMES]].mean(axis=1)
step_file=RUN_ROOT/'all_step_changes.csv.gz'
steps.to_csv(step_file,index=False,compression='gzip')
print('Saved',len(steps),'step-change records across all 14 attributes to',step_file)
print('Columns include total_, immediate_, background_, and value_ for each attribute')
print(steps[['phase','seed','arm','event_id','timestamp','sim_hour',
             'total_adult_energy_sufficiency','immediate_adult_energy_sufficiency',
             'background_adult_energy_sufficiency']].head(5).to_string(index=False))

PLOT_ARMS=('baseline','equal_normalized','learned_combination')
assert all(arm=='baseline' or arm in DIRS for arm in PLOT_ARMS)
colors={'baseline':'#555555','equal_normalized':'#087e8b',
        'learned_combination':'#c17c00'}
plot_paths=[]
for seed in FINAL_SEEDS:
    subset=steps[(steps.phase=='final') & (steps.seed==seed)]
    fig,axes=plt.subplots(14,1,sharex=True,figsize=(14,27),constrained_layout=True)
    for j,name in enumerate(VECTOR_NAMES):
        ax=axes[j]
        for arm in PLOT_ARMS:
            part=subset[subset.arm==arm]
            changed=part[np.abs(part[f'total_{name}'])>1e-10]
            ax.scatter(changed.sim_hour,changed[f'total_{name}'],
                       s=9,alpha=0.55,label=arm if j==0 else None,
                       color=colors.get(arm,None))
        ax.axhline(0,color='black',linewidth=0.5,alpha=0.4)
        ax.set_ylabel(name,fontsize=7)
        ax.grid(axis='x',alpha=0.2)
    axes[0].legend(loc='upper right',fontsize=8,ncol=3)
    axes[-1].set_xlabel('Simulated hours')
    axes[-1].set_xlim(0,FINAL_HOURS)
    fig.suptitle(f'Per-action hive attribute step changes — seed {seed}')
    path=RUN_ROOT/f'step_changes_seed_{seed}.png'
    fig.savefig(path,dpi=130)
    plot_paths.append(str(path))
    if seed==FINAL_SEEDS[0]: display(fig)
    plt.close(fig)

    fig,axes=plt.subplots(14,1,sharex=True,figsize=(14,27),constrained_layout=True)
    for j,name in enumerate(VECTOR_NAMES):
        ax=axes[j]
        for arm in PLOT_ARMS:
            part=subset[subset.arm==arm]
            ax.plot(part.sim_hour,part[f'value_{name}'],
                    linewidth=0.85,label=arm if j==0 else None,
                    color=colors.get(arm,None))
        ax.set_ylabel(name,fontsize=7)
        ax.set_ylim(0,1)
        ax.grid(alpha=0.2)
    axes[0].legend(loc='upper right',fontsize=8,ncol=3)
    axes[-1].set_xlabel('Simulated hours')
    axes[-1].set_xlim(0,FINAL_HOURS)
    fig.suptitle(f'Hive state vector trajectories — seed {seed}')
    path=RUN_ROOT/f'state_trajectories_seed_{seed}.png'
    fig.savefig(path,dpi=130)
    plot_paths.append(str(path))
    if seed==FINAL_SEEDS[0]: display(fig)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
    initial_score=next(r['initial_score'] for r in final_runs
                       if r['seed']==seed and r['arm']=='baseline')
    for arm in PLOT_ARMS:
        part=subset[subset.arm==arm]
        times=np.r_[0.0,part.sim_hour.to_numpy()]
        scores=np.r_[initial_score,part.hive_score.to_numpy()]
        ax.plot(times,scores,label=arm,color=colors.get(arm,None),linewidth=1.2)
    ax.set(xlabel='Simulated hours',ylabel='Mean of 14 hive attributes',
           title=f'Hive score trajectories — seed {seed}',xlim=(0,FINAL_HOURS))
    ax.grid(alpha=0.25); ax.legend()
    path=RUN_ROOT/f'score_trajectories_seed_{seed}.png'
    fig.savefig(path,dpi=150)
    plot_paths.append(str(path))
    if seed==FINAL_SEEDS[0]: display(fig)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
for arm in PLOT_ARMS:
    part=steps[(steps.phase=='final') & (steps.arm==arm)]
    average=part.groupby('event_id',sort=True)[['sim_hour','hive_score']].mean()
    ax.plot(np.r_[0.0,average.sim_hour.to_numpy()],
            np.r_[report['final_runs'][0]['initial_score'],average.hive_score.to_numpy()],
            label=arm,color=colors.get(arm,None),linewidth=1.4)
ax.set(xlabel='Simulated hours',ylabel='Mean hive score across final seeds',
       title='Baseline, equal-weight, and learned-weight score trajectories',
       xlim=(0,FINAL_HOURS))
ax.grid(alpha=0.25); ax.legend()
path=RUN_ROOT/'mean_score_trajectories.png'
fig.savefig(path,dpi=160)
plot_paths.append(str(path))
display(fig)
plt.close(fig)

report['step_change_records']=len(steps)
report['step_change_csv']=str(step_file)
report['step_change_plot_arms']=list(PLOT_ARMS)
report['step_change_plots']=plot_paths
(RUN_ROOT/'experiment_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('Saved',len(plot_paths),'figures; step changes and trajectories are available for every final seed')'''))

    nb = nbf.v4.new_notebook(cells=cells, metadata=original.metadata)
    return nb


for smoke, stage in ((True, 'kernel_smoke'), (False, 'kernel_full')):
    folder = ROOT / 'kaggle_deploy/column_steering_4h' / stage
    folder.mkdir(parents=True, exist_ok=True)
    nbf.write(make_notebook(smoke), folder / 'beehive_column_steering_4h.ipynb')
    suffix = '-smoke' if smoke else ''
    metadata = {
        'id': 'faithfulchukwunwogor/beehive-column-steering-four-hour' + suffix,
        'title': 'Beehive Column Steering Four Hour' + (' Smoke' if smoke else ''),
        'code_file': 'beehive_column_steering_4h.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': 'true',
        'enable_gpu': 'true', 'enable_internet': 'true', 'machine_shape': '',
        'dataset_sources': [
            'faithfulchukwunwogor/beehive-qwen35-runtime',
            'faithfulchukwunwogor/beehive-qwen35-menu-adapter',
            'faithfulchukwunwogor/beehive-menu-probe-weights'],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

nbf.write(make_notebook(False), ROOT / 'beehive_column_steering_4h.ipynb')
print('Built four-hour column-steering notebook, smoke notebook, and Kaggle metadata')
