import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checkpoint_gate import compare,promote
from regression_ablation import write,digest,read


def report(score,checkpoint):
    games=[]
    for seed in range(200):
        for seat in range(2):
            games.append(dict(config=dict(game_seed=seed,agent_seeds=[1,2]if seat==0 else[2,1],max_actions=1000),learner_seat=seat,score=score))
    return dict(config=dict(seed=7,games=400,inference_batch_games=128),device='cuda:0',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        baselines=[dict(baseline=dict(kind='greedy',weights=[0,0,1,5,10,100]),summary=dict(opponent='greedy',games=400,score=score),games=games)])


class CheckpointGateTests(unittest.TestCase):
    def test_promotion_binding_failure_preserves_incumbent_and_is_idempotent(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            models=[];specs=[];reports=[]
            for name,score in [('old',.5),('better',1.),('worse',0.)]:
                model=root/f'{name}.bpk';model.write_bytes(name.encode());models.append(model)
                spec=root/f'{name}-agent.json';write(spec,dict(descriptor=dict(id=name,name=name,metadata={}),config=dict(kind='monte_carlo',checkpoint=model.name)));specs.append(spec)
                result=root/f'{name}-eval.json';write(result,report(score,model));reports.append(result)
            registry=root/'champion.json'
            decision=root/'promoted.json'
            passed=promote(reports[0],reports[1],specs[0],specs[1],registry,decision)
            self.assertTrue(passed['passed']);original=registry.read_bytes()
            self.assertEqual(read(registry)['config']['checkpoint'],str(models[1]))
            self.assertEqual(passed,promote(reports[0],reports[1],specs[0],specs[1],registry,decision))
            failed=promote(reports[1],reports[2],specs[1],specs[2],registry,root/'failed.json')
            self.assertFalse(failed['passed']);self.assertEqual(registry.read_bytes(),original)
            models[2].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint hash'):
                promote(reports[1],reports[2],specs[1],specs[2],registry,root/'changed.json')

    def test_pairing_protocol_and_duplicate_seeds_are_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'model';p.write_bytes(b'model')
            a=report(.5,p);b=copy.deepcopy(a)
            self.assertFalse(compare(a,b)['passed'])
            b['config']['seed']=8
            with self.assertRaisesRegex(ValueError,'protocol'):compare(a,b)
            b=copy.deepcopy(a);b['baselines'][0]['games'][0]['config']['game_seed']=999
            with self.assertRaisesRegex(ValueError,'pairing'):compare(a,b)
            a=report(.5,p);a['baselines'][0]['games'][2]['config']['game_seed']=0;a['baselines'][0]['games'][3]['config']['game_seed']=0
            with self.assertRaisesRegex(ValueError,'duplicate'):compare(a,a)

    def test_primary_gain_cannot_hide_an_opponent_regression(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'model';p.write_bytes(b'model')
            a=report(.5,p);b=report(.5,p)
            a['baselines']=[];b['baselines']=[]
            for i,score in enumerate([1.,1.,0.]):
                old=report(.5,p)['baselines'][0];new=report(score,p)['baselines'][0]
                for baseline in [old,new]:
                    baseline['summary']['opponent']=str(i)
                    baseline['baseline']['weights'][-1]+=i
                a['baselines'].append(old);b['baselines'].append(new)
            result=compare(a,b)
            self.assertGreater(result['primary_lower_bound'],0)
            self.assertFalse(result['passed'])

    def test_aggregate_failure_prevents_individual_promotion(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);model=root/'model';model.write_bytes(b'model')
            spec=root/'agent.json';write(spec,dict(descriptor=dict(id='model',name='model',metadata={}),config=dict(kind='monte_carlo',checkpoint='model')))
            old=root/'old.json';new=root/'new.json';write(old,report(0.,model));write(new,report(1.,model))
            result=promote(old,new,spec,spec,root/'champion.json',root/'decision.json',eligible=False)
            self.assertFalse(result['passed']);self.assertFalse(result['aggregate_confirmation_passed'])

if __name__=='__main__':unittest.main()
