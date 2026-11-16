# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "matplotlib>=3.9,<4"]
# ///
"""Analyze the prespecified full-budget learning-rate and conditional replay study."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from learning_rate_confirmation import SEEDS,CHECKPOINTS,checkpoint
from regression_ablation import read,write
from opponent_study import parameter_payload_digest
from checkpoint_gate import load_scores,promote


def interval(values):
    rng=np.random.default_rng(2026111701);values=np.asarray(values);draws=[]
    for _ in range(20):
        seeds=rng.integers(values.shape[0],size=(500,values.shape[0]))
        games=rng.integers(values.shape[1],size=(500,values.shape[1]))
        draws.extend(values[seeds[:,:,None],games[:,None,:]].mean(axis=(1,2)))
    return np.quantile(draws,[.025,.975]).tolist()


def read_evaluation(path,names):
    report=read(path);keys,matchups=load_scores(report)
    if not set(names)<=matchups.keys():raise ValueError('evaluation panel incomplete')
    return keys,np.mean([matchups[name][1]for name in names],axis=0),matchups


def results(root,plan,arm,directory,seed):
    data=[];keys=None;opponents={}
    for training_seed in SEEDS:
        path=root/directory/f'lr-confirm-{arm}-s{training_seed}-i200-{seed}.json'
        k,score,matchups=read_evaluation(path,plan['release_primary'])
        if keys is None:keys=k
        if k!=keys:raise ValueError('evaluation seed pairing differs')
        data.append(score)
        for name,(_,values)in matchups.items():opponents.setdefault(name,[]).append(values)
    return keys,np.array(data),{name:np.array(v)for name,v in opponents.items()}


def curve(root,plan,arm,iteration):
    arrays=[];keys=None
    for seed in SEEDS:
        path=root/'curves'/f'lr-confirm-{arm}-s{seed}-i{iteration}-{plan["curve_seed"]}.json'
        k,score,_=read_evaluation(path,plan['curve_primary'])
        if keys is None:keys=k
        if k!=keys:raise ValueError('curve seed pairing differs')
        arrays.append(score)
    return np.array(arrays)


def analyze(root,phase):
    plan=read(root/'plan.json');models=read(root/'models.json')
    arm='low_lr'if phase=='lr'else'low_lr_replay'
    control='control'if phase=='lr'else'low_lr'
    directory='release-lr'if phase=='lr'else'release-replay'
    seed=plan['lr_release_seed']if phase=='lr'else plan['replay_release_seed']
    ka,a,oa=results(root,plan,arm,directory,seed);kb,b,ob=results(root,plan,control,directory,seed)
    if ka!=kb:raise ValueError('candidate/control schedules differ')
    delta=a-b;bounds=interval(delta);per_seed=delta.mean(axis=1)
    per_opponent={name:dict(candidate=float(oa[name].mean()),control=float(ob[name].mean()),
        difference=float((oa[name]-ob[name]).mean()),interval95=interval(oa[name]-ob[name]))for name in oa}
    retention_a=curve(root,plan,arm,200)-curve(root,plan,arm,50)
    retention_b=curve(root,plan,control,200)-curve(root,plan,control,50)
    retention_difference=retention_a-retention_b
    confirmed=bool(bounds[0]>0 and sum(per_seed>0)>=2)
    if phase=='lr':confirmed=confirmed and bool(min(per_seed)>-.05 and retention_difference.mean()>=0)
    decision=dict(confirmed=confirmed,candidate=arm,control=control,
        candidate_score=float(a.mean()),control_score=float(b.mean()),difference=float(delta.mean()),interval95=bounds,
        per_seed={str(seed):float(v)for seed,v in zip(SEEDS,per_seed)},opponents=per_opponent,
        candidate_retention=float(retention_a.mean()),control_retention=float(retention_b.mean()),
        retention_difference=float(retention_difference.mean()),retention_difference_interval95=interval(retention_difference),
        rule=plan['confirmation_rule']if phase=='lr'else plan['replay_rule'],
        caveat='Crossed bootstrap over three independent training seeds and shared paired-seat game seed blocks. Three seeds limit population inference.')
    decision_path=root/f'{phase}-decision.json'
    if decision_path.exists():assert read(decision_path)==decision,'Immutable analysis decision changed'
    else:write(decision_path,decision)
    # Candidate seed 42 and final checkpoint 200 are preselected; no searching
    # across trained seeds using held-out scores. A failed gate keeps the incumbent.
    incumbent_spec=Path(plan['incumbent_spec']if phase=='lr'else read(root/'replay-incumbent.json')['agent_spec'])
    gate=promote(root/directory/f'incumbent-{seed}.json',
        root/directory/f'lr-confirm-{arm}-s42-i200-{seed}.json',
        incumbent_spec,Path(models[f'lr-confirm-{arm}-s42'])/'agent.json',Path(plan['registry']),
        root/f'promotion-{phase}.json',family_alpha=.025,eligible=confirmed)
    initialization={}
    for training_seed in SEEDS:
        arms=['control','low_lr']+(['low_lr_replay']if phase=='replay'else[])
        hashes={}
        for candidate in arms:
            identity=f'lr-confirm-{candidate}-s{training_seed}'
            initial=checkpoint(root,models,identity,50).parent/'opponents/iteration-000000.bpk'
            hashes[candidate]=parameter_payload_digest(initial)
        if len(set(hashes.values()))!=1:raise ValueError('matched initial tensor payloads differ')
        initialization[str(training_seed)]=hashes
    write(root/'initialization-sha256.json',initialization)
    csv_rows=[]
    for stage in ['curves','release-lr','release-replay']:
        for path in sorted((root/stage).glob('*.json')):
            report=read(path)
            for evaluation in report['baselines']:
                summary=evaluation['summary'];wins,losses,draws=summary['outcomes']
                csv_rows.append(dict(stage=stage,checkpoint=report['id'],opponent=summary['opponent'],
                    games=summary['games'],wins=wins,losses=losses,draws=draws,score=summary['score']))
    with(root/'scores.csv').open('w')as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    available=['control','low_lr']+(['low_lr_replay']if phase=='replay'else[])
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    colors=['#64748b','#2563eb','#ea580c']
    for candidate,color in zip(available,colors):
        values=[curve(root,plan,candidate,i)for i in CHECKPOINTS]
        means=np.array([v.mean()for v in values]);ci=np.array([interval(v)for v in values])
        ax.errorbar(CHECKPOINTS,means,yerr=[means-ci[:,0],ci[:,1]-means],label=candidate,marker='o',capsize=4,color=color)
    ax.axhline(.5,color='#94a3b8',linestyle='--');ax.set(xlabel='Checkpoint iteration',ylabel='Mean score against three fixed opponents',
        ylim=(.3,.85),xticks=CHECKPOINTS,title='Full-budget training: 128 games per iteration, three independent seeds\nPointwise 95% intervals; final release results use a separate schedule and an additional opponent')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15);ax.legend(frameon=False)
    for ext in ['png','svg']:fig.savefig(root/f'confirmation-curves.{ext}',dpi=180)
    lines=['# Full-budget learning-rate confirmation','',
        'Six fresh runs compare learning rates 0.0003 and 0.0001 with independent initialization/training seeds 42–44. '+
        'All use 128 games per iteration, 200 iterations, a batch size of 128, natural fresh-sample updates, uniform historical sampling, '+
        'the same 80/20 self-play/history game mixture and the original exploration schedule. Full snapshots are saved every 50 iterations.','',
        'Checkpoint curves use a shared 1,000-game-per-opponent schedule against greedy, uniform-50 and adaptive-50; random is secondary. '+
        'Final release comparisons use a different schedule and add the frozen external mixed-opponent agent. '+
        'Final checkpoint 200 and promotion candidate seed 42 were specified before results.','',
        '![Training curves](confirmation-curves.png)','']
    for part in ['lr','replay']:
        path=root/f'{part}-decision.json'
        if not path.exists():continue
        d=read(path);lo,hi=d['interval95'];g=read(root/f'promotion-{part}.json')
        lines+=[f'## {"Learning rate" if part=="lr"else"Replay combination"} result','',
            f"Confirmation rule passed: **{d['confirmed']}**. Candidate mean release score {d['candidate_score']:.2%}; control {d['control_score']:.2%}. "+
            f"Difference **{100*d['difference']:+.2f} percentage points**, paired 95% interval [{100*lo:+.2f}, {100*hi:+.2f}].",'',
            f"Continuation from checkpoint 50 to 200: candidate {100*d['candidate_retention']:+.2f} points; control {100*d['control_retention']:+.2f} points. "+
            'This is measured on the curve panel; the release panel includes an extra reference.','',
            '| Opponent | Candidate | Control | Difference |','|---|---:|---:|---:|']
        for name,result in d['opponents'].items():
            lines.append(f"| {name} | {result['candidate']:.2%} | {result['control']:.2%} | {100*result['difference']:+.2f} pp |")
        lines+=['',f"Preselected seed-42 champion promotion gate passed: **{g['passed']}**. "+
            f"Primary lower bound {100*g['primary_lower_bound']:+.2f} points; every opponent must have a lower bound above -5 points. "+
            'Each of two possible gates uses family alpha 0.025, split across primary and opponent checks.','']
    lines+=['## Limits and reproduction','',
        '- Three independent training seeds provide a stronger check than the earlier continuations, but still limit precision.',
        '- Equal interaction budgets; natural gradient sample counts vary with trajectory length. Replay replaces 25% of the existing selected slots, preserving each iteration\'s gradient budget.',
        '- Replay starts empty, retains the latest 50,000 shuffled fresh samples with original Monte Carlo outcomes, and saves its state for exact resumption.',
        '- Frozen reference policies and their hashes are recorded before evaluations. Original artifacts are immutable.',
        '- A model agent.json publishes an immutable model identity. It does not make that model the champion. The champion is the separately gated data/champions/sequence.json pointer.',
        '- Plan, initialization/binary/reference hashes, manifests, logs, model paths, scores.csv, per-game evaluations and promotion evidence are retained alongside this report.','',
        '`python tools/learning_rate_confirmation.py run`','',
        '`uv run tools/analyze_learning_rate_confirmation.py data/studies/learning-rate-confirmation --phase '+phase+'`']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decision,indent=2));print('Promotion passed:',gate['passed'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--phase',choices=['lr','replay'],required=True)
    a=parser.parse_args();analyze(a.root.resolve(),a.phase)
