# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "matplotlib>=3.9,<4"]
# ///
"""Analyze the matched checkpoint-50 continuation experiment.

uv run tools/analyze_regression_ablation.py data/studies/regression-ablation
Resamples continuation seeds and evaluation seed blocks, preserving all arms,
reference opponents, and both seats together for paired comparisons.
"""
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
from analysis_common import read
ARMS = ['control', 'low_learning_rate', 'replay', 'historical_weight', 'fixed_exploration']
from analyze_history_diagnosis import blocks

SEEDS=[4201,4202,4203]
ORIGINS=['uniform','adaptive']
PRIMARY=['greedy','uniform-50','adaptive-50']
LABELS={'control':'Control','low_learning_rate':'Lower learning rate','replay':'25% retained samples',
        'historical_weight':'20% historical samples','fixed_exploration':'Fixed exploration'}


def bootstrap(values):
    """values: fixed origins x continuation seeds x paired evaluation seeds.

    Use the same seed draws across origins. Origins are two fixed trained models,
    not a sample of independent training seeds. Evaluation seed blocks are common
    across all origins, seeds, arms and opponents.
    """
    rng=np.random.default_rng(2026111602)
    estimates=[]
    for _ in range(20):
        seed_indices=rng.integers(values.shape[1],size=(500,values.shape[1]))
        block_indices=rng.integers(values.shape[2],size=(500,values.shape[2]))
        estimates.extend(values[:,seed_indices[:,:,None],block_indices[:,None,:]].mean(axis=(0,2,3)))
    return np.quantile(estimates,[.025,.975,.00625,.99375]).tolist()


def analyze(root):
    plan=read(root/'plan.json');models=read(root/'models.json')
    reports={}
    for path in sorted((root/'evaluations').glob('*.json')):
        report=read(path);identity=report['id']
        assert report['config']['seed']==plan['evaluation_seed']
        assert report['config']['games']==plan['evaluation_games']
        reports[identity]=report
    expected=[c['id']for c in plan['cases']]+[f'{origin}-initial'for origin in ORIGINS]
    missing=set(expected)-reports.keys()
    if missing:raise ValueError(f'{len(missing)} evaluations missing; analysis requires the whole prespecified matrix')
    arrays={}; seed_keys=None;row_results=[]
    for identity,report in reports.items():
        matchup={}
        for evaluation in report['baselines']:
            name=evaluation['summary']['opponent'];keys,scores=blocks(evaluation)
            if seed_keys is None:seed_keys=keys
            assert keys==seed_keys
            matchup[name]=scores
        assert set(matchup)=={'random',*PRIMARY}
        matchup['primary']=np.mean([matchup[name]for name in PRIMARY],axis=0)
        arrays[identity]=matchup
        row_results.append(dict(id=identity,**{name:float(values.mean())for name,values in matchup.items()}))
    comparisons=[];summary=[]
    for arm in ARMS:
        scores=np.array([[arrays[f'{origin}-{arm}-s{seed}']['primary']for seed in SEEDS]for origin in ORIGINS])
        control=np.array([[arrays[f'{origin}-control-s{seed}']['primary']for seed in SEEDS]for origin in ORIGINS])
        initial=np.array([[arrays[f'{origin}-initial']['primary']for seed in SEEDS]for origin in ORIGINS])
        ci=bootstrap(scores)
        summary.append(dict(arm=arm,score=float(scores.mean()),interval95=ci[:2],
                            mean_change_from_initial=float((scores-initial).mean()),
                            per_origin={o:float(scores[i].mean())for i,o in enumerate(ORIGINS)},
                            per_continuation_seed={str(s):float(scores[:,i].mean())for i,s in enumerate(SEEDS)}))
        if arm=='control':continue
        delta=scores-control;ci=bootstrap(delta)
        comparisons.append(dict(arm=arm,difference=float(delta.mean()),interval95=ci[:2],
            interval_familywise95_four_interventions=ci[2:],
            per_origin={o:dict(difference=float(delta[i].mean()),interval95=bootstrap(delta[i:i+1])[:2])for i,o in enumerate(ORIGINS)},
            per_continuation_seed={str(s):float(delta[:,i].mean())for i,s in enumerate(SEEDS)}))
    write_summary=dict(plan=plan,results=row_results,summary=summary,comparisons=comparisons,
        interval_note='Crossed bootstrap over three continuation seeds and paired game-seed blocks; two initial models fixed. Familywise interval uses Bonferroni for four pooled primary intervention comparisons. Small seed count limits precision.')
    (root/'summary.json').write_text(json.dumps(write_summary,indent=2)+'\n')
    with(root/'scores.csv').open('w')as file:
        writer=csv.DictWriter(file,fieldnames=list(row_results[0]));writer.writeheader();writer.writerows(row_results)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for i,origin in enumerate(ORIGINS):
        ax=axes[i]
        for j,arm in enumerate(ARMS):
            scores=np.array([[arrays[f'{origin}-{arm}-s{seed}']['primary']for seed in SEEDS]])
            ci=bootstrap(scores);mean=float(scores.mean())
            ax.errorbar(j,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',capsize=4,color='#2563eb')
            ax.scatter([j-.12,j,j+.12],scores.mean(axis=2).flatten(),s=22,color='#ea580c',alpha=.8,zorder=3)
        initial=float(arrays[f'{origin}-initial']['primary'].mean())
        ax.axhline(initial,color='#64748b',linestyle='--',label='Initial checkpoint 50')
        ax.set(title=f'{origin.capitalize()} starting model',ylabel='Mean score against three fixed opponents',
               xticks=range(5),xticklabels=[LABELS[a].replace(' ','\n')for a in ARMS],ylim=(.25,.8))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper right')
    fig.suptitle('Controlled continuations: checkpoint 50 → 100\nThree continuation seeds per condition; orange dots = individual seeds; pointwise 95% intervals')
    for ext in ['png','svg']:fig.savefig(root/f'intervention-comparison.{ext}',dpi=180)
    lines=['# Regression intervention experiment','',
           'All conditions branch from the original checkpoint-50 model and Adam state. '+
           'Each starting model uses three continuation seeds and five conditions, for 30 total runs. '+
           'Each run adds 50 iterations × 64 games and 27 × 128 gradient samples per iteration. '+
           'The final checkpoint 100 is evaluated without exploration on 400 games per opponent, '+
           'with both seats for each of 200 shared seed blocks.','',
           'The primary score equally averages greedy, uniform-50, and adaptive-50. Random is secondary. '+
           'These references and the final checkpoint were selected before results.','',
           '| Condition | Primary score | Change from control | 95% interval | Adjusted interval* |',
           '|---|---:|---:|---:|---:|']
    byarm={c['arm']:c for c in comparisons}
    for row in summary:
        arm=row['arm'];c=byarm.get(arm)
        if c:
            lo,hi=c['interval95'];alo,ahi=c['interval_familywise95_four_interventions']
            lines.append(f"| {LABELS[arm]} | {row['score']:.1%} | {100*c['difference']:+.1f} pp | [{100*lo:+.1f}, {100*hi:+.1f}] | [{100*alo:+.1f}, {100*ahi:+.1f}] |")
        else:lines.append(f"| {LABELS[arm]} | {row['score']:.1%} | — | — | — |")
    lines+=['', '## What the experiment shows', '',
            'Reducing the learning rate from 0.0003 to 0.0001 is the strongest intervention in this screening study. '+
            'It improves the pooled primary score by about 12 percentage points, with improvement in all six '+
            'matched starting-model/continuation-seed pairs. The familywise-adjusted interval stays above zero. '+
            'The architecture, optimizer initialization, game mixture, interaction count and gradient batch budget '+
            'are held fixed. This is direct evidence that update size contributes to the observed weakness under '+
            'this continuation protocol; it does not prove a specific internal failure mechanism.', '',
            'Retained samples improve the pooled primary score by about 6.6 percentage points, with five of six '+
            'matched pairs improving. This supports retaining experience as a useful intervention here, though '+
            'the original Monte Carlo targets become off-policy and the uniform-origin results vary considerably.', '',
            'Increasing historical sample weight and holding exploration fixed show no reliable benefit after '+
            'adjusting for four primary comparisons. These results do not rule out other pool settings or '+
            'exploration schedules.', '',
            'The next confirmation should use the original 128-game budget, independent initial training seeds '+
            'and a fresh reserved evaluation schedule. Combining lower learning rate with replay has not been tested.', '',
            '## Scores by opponent', '',
            '| Condition | Random | Greedy | Uniform-50 | Adaptive-50 |',
            '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        means=[np.mean([arrays[f'{origin}-{arm}-s{seed}'][name].mean() for origin in ORIGINS for seed in SEEDS])
               for name in ['random',*PRIMARY]]
        lines.append('| '+LABELS[arm]+' | '+' | '.join(f'{v:.1%}' for v in means)+' |')
    lines+=['','*Bonferroni adjustment across four pooled primary intervention comparisons. '+
            'Intervals resample continuation seeds and evaluation seed blocks, preserving pairing. '+
            'Only three continuation seeds are available; these are not independent initial training runs.','',
            '![Comparison](intervention-comparison.png)','','## Interpretation limits','',
            '- Both starting models were trained with seed 42. Results apply to these two fixed initial policies.',
            '- Reduced screening budget differs from the original 128-game, natural-update training. Use the control here for causal comparisons.',
            '- Retained samples are drawn from earlier continuation iterations with their original Monte Carlo outcomes. The buffer starts empty, keeps the latest 50,000 samples, and fills gradually. Replay changes coverage and introduces stale/off-policy targets together.',
            '- The historical-weight condition changes category proportions in gradient samples to 80/20, while retaining the 80/20 game mixture.',
            '- Fixed exploration holds epsilon at the saved checkpoint-50 value; all other arms follow the original 200-iteration decay.',
            '- Every condition has the same interaction and optimization budgets and per-iteration game/policy seed schedule. Trajectories and opponent qualities can diverge after the intervention.',
            '- This study tests these parameter choices. No benefit does not rule out the broader mechanism.',
            '- Archived continuation artifacts are inputs to this analysis. The historical continuation executor has been removed.','',
            '## Analyze saved results','',
            '`uv run tools/analyze_regression_ablation.py data/studies/regression-ablation`','',
            'See plan.json, provenance.json, models.json, logs/, evaluations/, scores.csv and summary.json for exact inputs and results.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    for row in summary:print(row)
    for row in comparisons:print(row)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
