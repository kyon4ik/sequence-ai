"""Promote an explicitly evaluated checkpoint only after a paired fixed-panel gate.

The evaluator reports must be bound to checkpoint hashes by the study runner.
Requires a positive primary difference and adjusted lower bounds above -5 points
for every opponent. Failed gates preserve the incumbent registry byte-for-byte.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from regression_ablation import read, write, digest


def load_scores(report):
    result={};schedule=None
    for evaluation in report['baselines']:
        name=evaluation['summary']['opponent']
        if name in result:raise ValueError('duplicate opponent')
        games=evaluation['games']
        if len(games)!=report['config']['games']:raise ValueError('evaluation config game count mismatch')
        if len(games)<400 or len(games)%2:raise ValueError('gate needs at least 400 completed, paired-seat games per opponent')
        keys=[];scores=[]
        for a,b in zip(games[::2],games[1::2]):
            if a['learner_seat']!=0 or b['learner_seat']!=1 or a['config']['game_seed']!=b['config']['game_seed'] or a['config']['agent_seeds']!=list(reversed(b['config']['agent_seeds'])):
                raise ValueError('invalid seat-swapped seed pairing')
            if any(g['score'] not in [0.,0.5,1.]for g in [a,b]):raise ValueError('invalid score')
            keys.append((a['config']['game_seed'],tuple(a['config']['agent_seeds']),a['config']['max_actions'],b['config']['max_actions']))
            scores.append((a['score']+b['score'])/2)
        if schedule is None:schedule=keys
        if keys!=schedule:raise ValueError('opponents have different seed schedules')
        if len(games)!=evaluation['summary']['games']:raise ValueError('game count mismatch')
        if abs(sum(scores)/len(scores)-evaluation['summary']['score'])>1e-12:raise ValueError('score summary mismatch')
        result[name]=(evaluation['baseline'],scores)
    if not result:raise ValueError('empty opponent panel')
    return schedule,result


def bound(values,alpha,repetitions=5000):
    rng=random.Random(2026111703);n=len(values)
    means=sorted(sum(rng.choices(values,k=n))/n for _ in range(repetitions))
    return means[max(0,int(alpha*repetitions)-1)]


def compare(incumbent,candidate,family_alpha=.05):
    if not 0<family_alpha<1:raise ValueError("invalid family alpha")
    if incumbent['config']!=candidate['config'] or incumbent['device']!=candidate['device']:
        raise ValueError('evaluation protocol differs')
    keys,a=load_scores(incumbent);other,b=load_scores(candidate)
    if len({key[0]for key in keys})!=len(keys):raise ValueError('duplicate game seeds')
    hashes=incumbent.get('opponent_checkpoint_sha256',{})
    if hashes!=candidate.get('opponent_checkpoint_sha256',{}):raise ValueError('opponent checkpoint hashes differ')
    for name,(policy,_) in a.items():
        if policy['kind']=='monte_carlo':
            if name not in hashes or digest(Path(policy['checkpoint']))!=hashes[name]:
                raise ValueError('opponent checkpoint is unbound or changed')
    if keys!=other or a.keys()!=b.keys():raise ValueError('candidate/incumbent seed schedules or panels differ')
    differences={}
    for name in a:
        if a[name][0]!=b[name][0]:raise ValueError('opponent policy differs')
        differences[name]=[y-x for x,y in zip(a[name][1],b[name][1])]
    primary=[sum(v)/len(differences)for v in zip(*differences.values())]
    alpha=family_alpha/(len(differences)+1)
    details={name:dict(difference=sum(v)/len(v),lower_bound=bound(v,alpha))for name,v in differences.items()}
    primary_lower=bound(primary,alpha);delta=sum(primary)/len(primary)
    passed=primary_lower>0 and all(d['lower_bound']>-.05 for d in details.values())
    return dict(passed=passed,primary_difference=delta,primary_lower_bound=primary_lower,
                opponents=details,seed_blocks=len(keys),one_sided_alpha_per_check=alpha,
                protocol='Paired seed-block percentile bootstrap; 5000 resamples; Bonferroni across primary plus opponents; strict primary superiority; per-opponent non-regression margin 5 points.')


def binding(report,spec_path):
    spec_path=spec_path.resolve();spec=read(spec_path)
    if spec['config']['kind']!='monte_carlo':raise ValueError('gate requires a neural checkpoint spec')
    model=(spec_path.parent/spec['config']['checkpoint']).resolve()
    if model!=Path(report['checkpoint']).resolve():raise ValueError('agent spec and evaluation checkpoint differ')
    expected=report.get('checkpoint_sha256')
    if not expected or digest(model)!=expected:raise ValueError('evaluation is missing a matching checkpoint hash')
    return dict(agent_spec=str(spec_path),agent_spec_sha256=digest(spec_path),checkpoint=str(model),checkpoint_sha256=expected)


def registry_record(binding,decision_path):
    spec=read(Path(binding['agent_spec']))
    spec['descriptor']=dict(id='champion',name='Sequence champion',metadata=dict(source_agent_id=spec['descriptor']['id']))
    spec['config']['checkpoint']=binding['checkpoint']
    return dict(**binding,**spec,decision=str(decision_path.resolve()))


def promote(incumbent_path,candidate_path,incumbent_spec,candidate_spec,registry,decision_path,family_alpha=.05,eligible=True):
    incumbent=read(incumbent_path);candidate=read(candidate_path)
    old=binding(incumbent,incumbent_spec);new=binding(candidate,candidate_spec)
    if decision_path.exists():
        prior=read(decision_path)
        if prior['incumbent_report_sha256']!=digest(incumbent_path) or prior['candidate_report_sha256']!=digest(candidate_path):
            raise ValueError('immutable gate decision inputs changed')
        if prior['aggregate_confirmation_passed']!=eligible:raise ValueError('immutable gate eligibility changed')
        if not registry.exists():
            write(registry,registry_record(new if prior['passed']else old,decision_path))
        elif prior['passed'] and read(registry)['checkpoint_sha256']==old['checkpoint_sha256']:
            write(registry,registry_record(new,decision_path))
        return prior
    if registry.exists():
        registered=read(registry)
        if registered['checkpoint_sha256']!=old['checkpoint_sha256']:
            raise ValueError('evaluated incumbent is not the registered champion')
        if registered['agent_spec_sha256']!=old['agent_spec_sha256']:
            raise ValueError('registered incumbent spec changed')
    # An absent registry is initialized with the explicit incumbent, never with an
    # unevaluated candidate. Existing registries are untouched when a gate fails.
    result=compare(incumbent,candidate,family_alpha)
    result["aggregate_confirmation_passed"]=eligible
    result["passed"]=result["passed"] and eligible
    result.update(incumbent=old,candidate=new,incumbent_report=str(incumbent_path.resolve()),
                  candidate_report=str(candidate_path.resolve()),
                  incumbent_report_sha256=digest(incumbent_path),candidate_report_sha256=digest(candidate_path))
    write(decision_path,result)
    if result['passed']:write(registry,registry_record(new,decision_path))
    elif not registry.exists():write(registry,registry_record(old,decision_path))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['incumbent','candidate','incumbent-spec','candidate-spec','registry','decision']:
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--family-alpha',type=float,default=.05)
    a=parser.parse_args();r=promote(a.incumbent,a.candidate,a.incumbent_spec,a.candidate_spec,a.registry,a.decision,a.family_alpha)
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['passed']else 2)
