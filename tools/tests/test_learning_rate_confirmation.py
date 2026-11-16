import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from learning_rate_confirmation import evaluate
from regression_ablation import read,write,digest

class ConfirmationDriverTests(unittest.TestCase):
    def test_release_evaluation_accepts_serialized_incumbent_and_binds_final_model(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);model_dir=root/'model';model_dir.mkdir();(root/'logs').mkdir()
            model=model_dir/'model.bpk';model.write_bytes(b'model')
            spec=root/'agent.json';write(spec,dict(config=dict(kind='monte_carlo',checkpoint='model/model.bpk')))
            plan=dict(release_primary=['greedy'],reference_paths={},evaluation_games=1000)
            def fake_evaluator(command,**kwargs):
                manifest=read(Path(command[1]));checkpoint=manifest['checkpoints'][0];seed=manifest['seeds'][0]
                write(Path(command[2])/f"{checkpoint['id']}-{seed}.json",dict(config=dict(seed=seed),checkpoint=checkpoint['path']))
            with patch('learning_rate_confirmation.subprocess.run',side_effect=fake_evaluator):
                evaluate(root,plan,{'case':str(model_dir)},[dict(id='case')],'release',77,release=True,incumbent_spec=str(spec))
            for identity in ['case-i200','incumbent']:
                report=read(root/'release'/f'{identity}-77.json')
                self.assertEqual(Path(report['checkpoint']),model)
                self.assertEqual(report['checkpoint_sha256'],digest(model))

if __name__=='__main__':unittest.main()
