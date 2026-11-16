# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "matplotlib>=3.9,<4"]
# ///
"""Summarize checkpoint evaluations with paired, seed-block bootstrap intervals.

uv run tools/analyze_history_diagnosis.py data/studies/history-diagnosis
Intervals describe game-seed uncertainty for these trained models, not uncertainty
across independent training runs. Seat-swapped games remain in the same block.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def blocks(evaluation):
    games = evaluation['games']
    assert len(games) > 0 and len(games) % 2 == 0
    keys, scores = [], []
    for a, b in zip(games[::2], games[1::2]):
        assert a['learner_seat'] == 0 and b['learner_seat'] == 1
        assert a['config']['game_seed'] == b['config']['game_seed']
        assert a['config']['agent_seeds'] == list(reversed(b['config']['agent_seeds']))
        keys.append((a['config']['game_seed'], tuple(a['config']['agent_seeds']), a['config']['max_actions']))
        scores.append((a['score'] + b['score']) / 2)
    return keys, np.array(scores)


def interval(values):
    rng = np.random.default_rng(901)
    estimates = np.concatenate([values[rng.integers(len(values), size=(500, len(values)))].mean(axis=1) for _ in range(20)])
    return [float(v) for v in np.quantile(estimates, [.025, .975])]


def analyze(root):
    evaluations = {}
    rows = []
    for directory in ['baselines', 'panel']:
        for path in sorted((root / directory).glob('*.json')):
            report = json.loads(path.read_text())
            mode, iteration = report['id'].rsplit('-', 1)
            for evaluation in report['baselines']:
                opponent = evaluation['summary']['opponent']
                key = (mode, int(iteration), opponent)
                assert key not in evaluations
                keys, scores = blocks(evaluation)
                evaluations[key] = (keys, scores)
                lower, upper = interval(scores)
                rows.append(dict(agent=mode, iteration=int(iteration), opponent=opponent,
                    games=len(scores)*2, score=float(scores.mean()), lower=lower, upper=upper))
    for mode in ['uniform', 'adaptive']:
        for iteration in [50, 100, 150, 200]:
            a = evaluations.get((mode, iteration, 'uniform-50'))
            b = evaluations.get((mode, iteration, 'adaptive-50'))
            if a is None or b is None: continue
            assert a[0] == b[0]
            scores = (a[1] + b[1]) / 2
            evaluations[(mode, iteration, 'fixed-neural-panel')] = (a[0], scores)
            lower, upper = interval(scores)
            rows.append(dict(agent=mode, iteration=iteration, opponent='fixed-neural-panel',
                games=len(scores)*4, score=float(scores.mean()), lower=lower, upper=upper))
    comparisons = []
    def compare(a, b, label):
        if a not in evaluations or b not in evaluations:
            return
        ka, va = evaluations[a]; kb, vb = evaluations[b]
        assert ka == kb, 'paired comparisons require identical game and policy seed schedules'
        delta = va-vb
        comparisons.append(dict(comparison=label, opponent=a[2], difference=float(delta.mean()), interval=interval(delta)))
    opponents = sorted({r['opponent'] for r in rows})
    for opponent in opponents:
        for mode in ['uniform','adaptive']:
            compare((mode,200,opponent),(mode,50,opponent), f'{mode}: 200 minus 50')
        for iteration in [50,100,150,200]:
            compare(('adaptive',iteration,opponent),('uniform',iteration,opponent), f'adaptive minus uniform at {iteration}')
    (root/'summary.json').write_text(json.dumps(dict(results=rows,comparisons=comparisons,
        caveat='One training seed. Pointwise 95% bootstrap intervals over 500 paired-seat seed blocks; not adjusted for multiple comparisons.'),indent=2))
    with (root/'scores.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,opponent in zip(axes.flat,['random','greedy','uniform-50','adaptive-50']):
        for mode,color in [('uniform','#2563eb'),('adaptive','#ea580c')]:
            series=sorted([r for r in rows if r['agent']==mode and r['opponent']==opponent],key=lambda r:r['iteration'])
            if not series:continue
            x=np.array([r['iteration'] for r in series]); y=np.array([r['score'] for r in series])
            ax.errorbar(x,y,yerr=[y-[r['lower'] for r in series],[r['upper'] for r in series]-y],label=mode.capitalize(),marker='o',capsize=3,color=color)
        ax.axhline(.5,color='#64748b',linestyle='--',linewidth=1)
        ax.set(title=f'Against {opponent}',xlabel='Checkpoint iteration',ylabel='Score rate',ylim=(.3,1),xticks=[50,100,150,200])
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Checkpoint investigation: fixed opponents, shared evaluation seeds\n1,000 games per matchup; pointwise 95% seed-block bootstrap intervals; one training seed')
    for ext in ['png','svg']:fig.savefig(root/f'checkpoint-comparison.{ext}',dpi=180)
    diagnostics = {}
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    for mode, color in [('uniform', '#2563eb'), ('adaptive', '#ea580c')]:
        paths = sorted((root/'diagnostics').glob(f'{mode}-200-*.json'))
        if not paths: continue
        report = json.loads(paths[0].read_text())
        diagnostic = report['baselines'][0]['diagnostics']
        diagnostics[mode] = diagnostic
        bins = [b for b in diagnostic['value_bins'] if b['decisions'] >= 30]
        ax.plot([(b['prediction_sum']/b['decisions']+1)/2 for b in bins],
                [(b['outcome_sum']/b['decisions']+1)/2 for b in bins],
                marker='o', color=color, label=mode.capitalize())
    ax.plot([0,1],[0,1], '--', color='#64748b', label='Predicted = observed')
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='Predicted score of chosen move: (value + 1) / 2',
           ylabel='Observed final score', title='Final checkpoints vs greedy: value reliability\nFirst 200 paired-seed games; bins with at least 30 decisions')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.15); ax.legend(frameon=False)
    fig.savefig(root/'value-reliability.png',dpi=180)
    (root/'diagnostic-summary.json').write_text(json.dumps(diagnostics,indent=2))
    for row in rows:print(f"{row['agent']:9} {row['iteration']:3} vs {row['opponent']:12}: {row['score']:.1%} [{row['lower']:.1%}, {row['upper']:.1%}]")
    print('\nChanges from checkpoint 50 to 200:')
    for c in comparisons:
        if 'minus 50' in c['comparison']:print(c)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
