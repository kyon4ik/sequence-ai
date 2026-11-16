"""Full-budget LR confirmation, conditional replay experiment and champion gates.

Build train_study/evaluate examples first. Run on host GPU:
    python tools/learning_rate_confirmation.py run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import subprocess
from datetime import datetime,timezone
from regression_ablation import REPO,read,write,digest
from opponent_study import parameter_payload_digest

ARMS=['control','low_lr'];SEEDS=[42,43,44];CHECKPOINTS=[50,100,150,200]


def prepare(root):
    base=read(REPO/'data/models/mc-history-uniform-v1/experiment.json')['training']
    base.update(iterations=200,games_per_iteration=128,batch_size=128,inference_batch_games=128,
                epsilon_decay_iterations=200,replay=None)
    cases=[]
    for seed in SEEDS:
        arms=ARMS if seed%2==0 else list(reversed(ARMS))
        cases.extend(dict(id=f'lr-confirm-{arm}-s{seed}',arm=arm,seed=seed)for arm in arms)
    registry=REPO/'data/champions/sequence.json'
    if(root/'plan.json').exists():return read(root/'plan.json')
    incumbent=Path(read(registry)['agent_spec'])if registry.exists()else REPO/'data/models/mc-history-uniform-v1/agent.json'
    refs={f'{mode}-50':str(REPO/f'data/models/mc-history-{mode}-v1/checkpoints/iteration-000050/model.bpk')for mode in ['uniform','adaptive']}
    refs['external-mixed']=str(REPO/'data/models/mc-mixed-v1/model.bpk')
    plan=dict(cases=cases,training=base,checkpoints=CHECKPOINTS,
              seeds=SEEDS,evaluation_games=1000,curve_seed=2026111701,lr_release_seed=2026111702,
              replay_release_seed=2026111704,reference_paths=refs,
              curve_primary=['greedy','uniform-50','adaptive-50'],release_primary=['greedy',*refs],
              incumbent_spec=str(incumbent.resolve()),registry=str(registry),
              confirmation_rule='Final release primary paired 95% lower bound > 0; at least 2/3 seeds positive; no seed decline > 5 points; lower-rate mean 200-minus-50 retention >= control.',
              replay_rule='Conditional on LR confirmation; compare full 200-iteration 1e-4+25% replay runs with 1e-4 alone; release primary 95% lower bound > 0 and at least 2/3 seeds improve.',
              promotion_rule='Preselected seed-42 final candidate only; strict primary superiority and per-opponent lower bounds > -5 points; one-sided Bonferroni bootstrap with family alpha .025 for each of two possible gates.',
              limitations=['Three independent training seeds, small seed count.',
                           'Equal environment interactions; natural fresh-sample update counts can vary with trajectory length.',
                           'Reference opponents fixed before results; external-mixed used only in final release comparisons.',
                           'No checkpoint selection using the final release schedule. Final checkpoint 200 is preselected.'])
    root.mkdir(parents=True,exist_ok=True);write(root/'plan.json',plan)
    write(root/'provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
         git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
         binaries={name:digest(REPO/f'target/release/examples/{name}')for name in ['train_study','evaluate']},
         references={name:digest(Path(path))for name,path in refs.items()},
         incumbent_spec_sha256=digest(incumbent)))
    return plan


def train(root,plan,cases):
    models=read(root/'models.json')if(root/'models.json').exists()else{}
    logs=root/'logs';logs.mkdir(exist_ok=True)
    for case in cases:
        identity=case['id']
        if identity in models and(Path(models[identity])/'agent.json').exists():continue
        config=dict(plan['training']);config.update(seed=case['seed'],learning_rate=.0003 if case['arm']=='control'else .0001)
        if case['arm']=='low_lr_replay':config['replay']=dict(capacity=50000,fraction=.25)
        candidates=sorted((root/'models').glob(identity+'-attempt-*'))if(root/'models').exists()else[]
        completed=[p for p in candidates if(p/'agent.json').exists()]
        if completed:
            models[identity]=str(completed[0]);write(root/'models.json',models);continue
        snapshots=[p for d in candidates for p in(d/'checkpoints').glob('iteration-*')if(p/'state.json').exists()]
        resume=max(snapshots,key=lambda p:int(p.name.split('-')[-1]))if snapshots else None
        attempt=0
        while(root/'models'/f'{identity}-attempt-{attempt}').exists():attempt+=1
        destination=root/'models'/f'{identity}-attempt-{attempt}';destination.parent.mkdir(exist_ok=True)
        manifest=dict(id=identity,training=config,checkpoint_interval=50,resume=str(resume)if resume else None)
        path=root/'configs'/f'{identity}-attempt-{attempt}.json';write(path,manifest)
        command=[str(REPO/'target/release/examples/train_study'),str(path),str(destination)]
        write(root/'active.json',dict(phase='training',case=case,destination=str(destination)))
        print('Training '+identity,flush=True)
        with(logs/f'{destination.name}.log').open('w')as log:subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        # A resumed run owns only newer published snapshots. Preserve older
        # selected checkpoints via source references in the model ledger.
        models[identity]=str(destination);write(root/'models.json',models)
        rows=read(destination/'report.json')['iterations'];assert len(rows)==200
        assert all(r['episodes']==128 for r in rows)
    return models


def checkpoint(root,models,identity,iteration):
    direct=Path(models[identity])/'checkpoints'/f'iteration-{iteration:06}'/'model.bpk'
    if direct.exists():return direct
    candidates=sorted((root/'models').glob(identity+'-attempt-*'))
    for path in candidates:
        found=path/'checkpoints'/f'iteration-{iteration:06}'/'model.bpk'
        if found.exists():return found
    raise FileNotFoundError(direct)


def panel(plan,release):
    names=plan['release_primary']if release else plan['curve_primary']
    result=[]
    for name in names:
        config=dict(kind='greedy',weights=[0,0,1,5,10,100])if name=='greedy'else dict(kind='monte_carlo',checkpoint=plan['reference_paths'][name])
        result.append([name,config])
    if not release:result.append(['random',dict(kind='random')])
    return result


def evaluate(root,plan,models,cases,directory,seed,release=False,incumbent_spec=None):
    entries=[]
    for case in cases:
        for iteration in([200]if release else CHECKPOINTS):
            entries.append(dict(id=f"{case['id']}-i{iteration}",path=str(Path(models[case['id']])/"model.bpk" if release else checkpoint(root,models,case['id'],iteration))))
    if incumbent_spec:
        spec=read(Path(incumbent_spec));path=(Path(incumbent_spec).parent/spec['config']['checkpoint']).resolve()
        entries.append(dict(id='incumbent',path=str(path)))
    opponents=panel(plan,release)
    opponent_hashes={name:digest(Path(config['checkpoint']))for name,config in opponents if config['kind']=='monte_carlo'}
    for entry in entries:
        output=root/directory/f"{entry['id']}-{seed}.json"
        checkpoint_hash=digest(Path(entry['path']))
        evidence=root/'evaluation-inputs'/f'{directory}-{entry["id"]}.json'
        identity=dict(checkpoint=entry,checkpoint_sha256=checkpoint_hash,opponent_checkpoint_sha256=opponent_hashes,seed=seed)
        if evidence.exists():assert read(evidence)==identity,'Evaluation inputs changed'
        else:write(evidence,identity)
        if not output.exists():
            manifest=dict(device='cuda:0',games=plan['evaluation_games'],inference_batch_games=128,seeds=[seed],checkpoints=[entry],opponents=opponents)
            path=root/'evaluation-manifest.json';write(path,manifest)
            write(root/'active.json',dict(phase='evaluation',directory=directory,checkpoint=entry))
            print(f"Evaluating {directory}: {entry['id']}",flush=True)
            with(root/'logs'/f'{directory}.log').open('a')as log:
                subprocess.run([str(REPO/'target/release/examples/evaluate'),str(path),str(root/directory)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert digest(Path(entry['path']))==checkpoint_hash
        assert {name:digest(Path(config['checkpoint']))for name,config in opponents if config['kind']=='monte_carlo'}==opponent_hashes
        report=read(output);assert report['config']['seed']==seed
        report.update(checkpoint_sha256=checkpoint_hash,opponent_checkpoint_sha256=opponent_hashes)
        write(output,report)


def run(root,plan):
    if(root/'active.json').exists() and read(root/'active.json')['phase']=='complete':
        print('Study already complete: '+str(root));return
    cases=plan['cases'];models=train(root,plan,cases)
    for seed in SEEDS:
        payloads=[]
        for arm in ARMS:
            path=checkpoint(root,models,f'lr-confirm-{arm}-s{seed}',50).parent/'opponents/iteration-000000.bpk'
            payloads.append(parameter_payload_digest(path))
        assert len(set(payloads))==1,'Matched runs did not share initialization'
    evaluate(root,plan,models,cases,'curves',plan['curve_seed'])
    evaluate(root,plan,models,cases,'release-lr',plan['lr_release_seed'],True,plan['incumbent_spec'])
    command=['uv','run',str(REPO/'tools/analyze_learning_rate_confirmation.py'),str(root),'--phase','lr']
    subprocess.run(command,cwd=REPO,check=True)
    decision=read(root/'lr-decision.json')
    if decision['confirmed']:
        replay_cases=[dict(id=f'lr-confirm-low_lr_replay-s{s}',arm='low_lr_replay',seed=s)for s in SEEDS]
        models=train(root,plan,replay_cases)
        evaluate(root,plan,models,replay_cases,'curves',plan['curve_seed'])
        low_cases=[c for c in cases if c['arm']=='low_lr']
        registry=Path(plan['registry'])
        if(root/'replay-incumbent.json').exists():incumbent=read(root/'replay-incumbent.json')['agent_spec']
        else:
            incumbent=read(registry)['agent_spec']if registry.exists()else plan['incumbent_spec']
            write(root/'replay-incumbent.json',dict(agent_spec=incumbent))
        evaluate(root,plan,models,low_cases+replay_cases,'release-replay',plan['replay_release_seed'],True,incumbent)
        subprocess.run(['uv','run',str(REPO/'tools/analyze_learning_rate_confirmation.py'),str(root),'--phase','replay'],cwd=REPO,check=True)
    write(root/'active.json',dict(phase='complete',completed_utc=datetime.now(timezone.utc).isoformat()))
    print('Full study complete: '+str(root),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=REPO/'data/studies/learning-rate-confirmation')
    a=parser.parse_args();root=a.output.resolve();plan=prepare(root)
    if a.command=='run':run(root,plan)
