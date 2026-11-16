"""Prepare/run controlled continuations from the two seed-42 iteration-50 models.

Build examples first, then: python tools/regression_ablation.py run
Partial runs are preserved; rerunning uses a new attempt directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ARMS = ['control', 'low_learning_rate', 'replay', 'historical_weight', 'fixed_exploration']


def read(path): return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    cases = []
    order = random.Random(2026111602)
    for seed in [4201, 4202, 4203]:
        group = [dict(id=f'{origin}-{arm}-s{seed}', origin=origin, arm=arm, seed=seed)
                 for origin in ['uniform', 'adaptive'] for arm in ARMS]
        order.shuffle(group); cases.extend(group)
    plan = dict(cases=cases, steps=50, games=64, updates=27, batch_size=128,
                replay_capacity=50000, replay_fraction=.25, learning_rate_divisor=3,
                evaluation_games=400, evaluation_seed=2026111602,
                primary='Mean score against greedy, uniform-50, adaptive-50, equally weighted',
                secondary='Random score, per-opponent scores, fresh/historical/replay sample counts',
                caveats=['Three continuation seeds from each of two fixed seed-42 models; not independent initial training runs.',
                         'Screening uses 64 games/27 updates per iteration, versus original 128 games/natural fresh-sample updates.',
                         'Replay keeps prior continuation samples, initially empty; old Monte Carlo targets are off-policy.',
                         'Historical sample weighting targets 20% gradient samples without changing game mix.',
                         'Fixed exploration holds the checkpoint-50 epsilon; control keeps original decay horizon 200.',
                         'Final checkpoint 100 is preselected; no evaluation-based checkpoint selection.'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'plan.json').exists():
        assert read(root/'plan.json') == plan, 'Use a new output directory for a different plan'
    else:
        write(root/'plan.json',plan)
        sources = {}
        for origin in ['uniform','adaptive']:
            source=REPO/f'data/models/mc-history-{origin}-v1/checkpoints/iteration-000050'
            sources[origin]={name:digest(source/name) for name in ['model.bpk','optimizer.bpk','state.json']}
        write(root/'provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
             binaries={name:digest(REPO/f'target/release/examples/{name}')for name in ['ablation','evaluate']},
             sources=sources))
    for case in cases:
        config=dict(source=str(REPO/f"data/models/mc-history-{case['origin']}-v1/checkpoints/iteration-000050"),
                    intervention=case['arm'],seed=case['seed'],
                    **{key:plan[key] for key in ['steps','games','updates','batch_size','replay_capacity']})
        write(root/'configs'/f"{case['id']}.json",config)
    return plan


def validate(rows, plan, case):
    assert len(rows)==plan['steps']
    for offset,row in enumerate(rows):
        assert row['iteration']==51+offset
        assert len(row['game_seed_schedule'])==plan['games']
        assert row['optimization_samples']==plan['updates']*plan['batch_size']
        assert abs(row['learning_rate']-(0.0001 if case['arm']=='low_learning_rate' else 0.0003))<1e-15
        expected=.3+(49 if case['arm']=='fixed_exploration' else 50+offset)/199*(.05-.3)
        assert abs(row['epsilon']-expected)<1e-12
        assert row['retained_samples']==(864 if case['arm']=='replay' and offset>0 else 0)
        assert row['replay_size']<=plan['replay_capacity']
        if case['arm']=='historical_weight': assert 691<=row['historical_samples']<=692


def run(root,plan):
    resolved=read(root/'models.json')if(root/'models.json').exists()else{}
    logs=root/'logs';logs.mkdir(exist_ok=True)
    for case in plan['cases']:
        identity=case['id']
        existing=Path(resolved[identity])if identity in resolved else None
        if existing is None or not(existing/'complete.json').exists():
            attempt=0
            while True:
                destination=root/'models'/f'{identity}-attempt-{attempt}'
                if not destination.exists():break
                if(destination/'complete.json').exists():break
                attempt+=1
            destination.parent.mkdir(exist_ok=True)
            if not(destination/'complete.json').exists():
                command=[str(REPO/'target/release/examples/ablation'),str(root/'configs'/f'{identity}.json'),str(destination)]
                write(root/'active.json',dict(phase='training',case=case,command=command))
                print(f'Training {identity} ({len(resolved)}/{len(plan["cases"])} completed)',flush=True)
                with(logs/f'{destination.name}.log').open('w')as log:
                    subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
            resolved[identity]=str(destination);write(root/'models.json',resolved)
        validate(read(Path(resolved[identity])/'report.json'),plan,case)
    # Verify game and policy seed schedules across every arm (even though games diverge).
    for origin in ['uniform','adaptive']:
        for seed in [4201,4202,4203]:
            schedules=[]
            for arm in ARMS:
                rows=read(Path(resolved[f'{origin}-{arm}-s{seed}'])/'report.json')
                schedules.append([r['game_seed_schedule']for r in rows])
            assert all(s==schedules[0]for s in schedules)
    checkpoints=[dict(id=case['id'],path=str(Path(resolved[case['id']])/'model.bpk'))for case in plan['cases']]
    checkpoints.extend(dict(id=f'{origin}-initial',path=str(REPO/f'data/models/mc-history-{origin}-v1/checkpoints/iteration-000050/model.bpk'))for origin in ['uniform','adaptive'])
    opponents=[['random',dict(kind='random')],['greedy',dict(kind='greedy',weights=[0,0,1,5,10,100])]]
    opponents.extend([f'{origin}-50',dict(kind='monte_carlo',checkpoint=str(REPO/f'data/models/mc-history-{origin}-v1/checkpoints/iteration-000050/model.bpk'))]for origin in ['uniform','adaptive'])
    for checkpoint in checkpoints:
        expected=root/'evaluations'/f"{checkpoint['id']}-{plan['evaluation_seed']}.json"
        if expected.exists():continue
        manifest=dict(device='cuda:0',games=plan['evaluation_games'],inference_batch_games=128,
                      seeds=[plan['evaluation_seed']],checkpoints=[checkpoint],opponents=opponents)
        path=root/'evaluation-manifest.json';write(path,manifest)
        write(root/'active.json',dict(phase='evaluation',checkpoint=checkpoint))
        print('Evaluating '+checkpoint['id'],flush=True)
        with(logs/'evaluation.log').open('a')as log:
            subprocess.run([str(REPO/'target/release/examples/evaluate'),str(path),str(root/'evaluations')],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'evaluation-checkpoints.json',{c['id']:dict(path=c['path'],sha256=digest(Path(c['path'])))for c in checkpoints})
    write(root/'active.json',dict(phase='complete',completed_utc=datetime.now(timezone.utc).isoformat()))
    print('Study complete: '+str(root),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=REPO/'data/studies/regression-ablation')
    args=parser.parse_args();root=args.output.resolve();plan=prepare(root)
    if args.command=='run':run(root,plan)
