"""The retained-gradient oracle must reject a corrupted collective result."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.validate_resnet_steps import check_resnet_steps
from MyFlows.distributed.launcher import run_training
from MyFlows.distributed.measurement import write_json


class ResNetCollectiveAuditTest(unittest.TestCase):
    def test_detects_corruption_before_optimizer_reference(self):
        rng=np.random.default_rng(2)
        split={name:(rng.random((24,3,32,32),dtype=np.float32),rng.random((24,2),dtype=np.float32))
               for name in ('train','val','test')}
        base=dict(task='donkey_resnet18',device='cpu',base_width=2,image_h=32,image_w=32,
                  optimizer='adam',learning_rate=.0003,seed=0,global_batch=8,epochs=1,steps=3,
                  timeout=30,profile=False,collect_history=False,numerical_snapshots=True,
                  record_local_gradients=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reference=root/'single';actual=root/'ps'
            result=run_training(**base,split_arrays=split,mode='single',transport='none',train_workers=1,artifact_dir=str(reference))
            self.assertEqual(result['status'],'passed')
            config=dict(base,mode='ps',transport='grpc_proto',train_workers=2,shard_sizes=[5,3],artifact_dir=str(actual))
            result=run_training(**config,split_arrays=split)
            self.assertEqual(result['status'],'passed',result.get('error'))
            write_json(actual/'config.json',config)
            check_resnet_steps(actual,reference,steps=3)
            path=actual/'rank-0/step-0001.npz'
            with np.load(path) as f:
                state={k:f[k] for k in f.files}
            state['gradient::stem.conv.kernel'].flat[0]+=.2
            np.savez(path,**state)
            with self.assertRaisesRegex(AssertionError,'collective'):
                check_resnet_steps(actual,reference,steps=3)


if __name__=='__main__':unittest.main()
