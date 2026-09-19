import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from MyFlows.distributed.measurement import save_state, load_state
from MyFlows.distributed.session import TrainSession
from MyFlows.distributed.engine import PSEngine
from MyFlows.distributed.transport_grpc import RingInbox, dict_to_ps_response, ps_response_to_dict


class ExperimentIntegrityTest(unittest.TestCase):
    def test_ring_control_and_gradients_use_protobuf(self):
        import json
        from MyFlows.distributed.launcher import run_training
        with tempfile.TemporaryDirectory() as directory:
            result = run_training(task='synthetic', device='cpu', optimizer='adam', mode='ring',
                                  transport='grpc_proto', train_workers=2, global_batch=8, steps=2,
                                  epochs=1, artifact_dir=directory, collect_history=False, timeout=30)
            self.assertEqual(result['status'], 'passed', result.get('error'))
            for rank in range(2):
                final = json.loads((Path(directory)/f'rank-{rank}/final.json').read_text(encoding='utf-8'))
                traffic = final['traffic']
                self.assertNotIn('socket_framing_bytes', traffic)
                self.assertGreater(traffic['join_init_request_bytes'], 0)
                self.assertGreater(traffic['confirm_update_request_bytes'], 0)
                self.assertGreater(traffic['SCATTER_request_bytes'], 0)

    def test_heartbeat_expiry_aborts_whole_synchronous_run(self):
        e = PSEngine(run_id='hb', n_workers=2, init_parameters={'w':np.zeros(1,np.float32)},
                     optimizer_meta={}, shard_sizes=[1,1], heartbeat_timeout_s=3)
        e.initialized.update((0,1))
        e.heartbeats={0:10.,1:12.}
        self.assertEqual(e.check_liveness(now=14.), [0])
        self.assertIn('heartbeat timeout',e.aborted)
        self.assertEqual(e.n_workers,2)

    def test_grpc_preserves_optimizer_hyperparameters_exactly(self):
        meta={'impl':'Adam','learning_rate':.001,'beta_1':.9,'beta_2':.999,'eps':1e-8,'t':20}
        restored=ps_response_to_dict(dict_to_ps_response({'optimizer':meta}))['optimizer']
        for key,value in meta.items():
            self.assertEqual(restored[key],value)

    def test_ring_rejects_wrong_neighbor_and_future_step(self):
        inbox=RingInbox();inbox.configure('r',1,4,'schema',2)
        data=np.ones(2,dtype='<f4').tobytes()
        msg=dict(run_id='r',schema_hash='schema',receiver_rank=1,sender_rank=0,
                 stage='SCATTER',round=0,chunk_id=0,valid_length=2,data=data,
                 payload_hash=hashlib.sha256(data).hexdigest(),global_step=0,collective_id='r:0')
        self.assertEqual(inbox.put(dict(msg,sender_rank=2))['status'],'INVALID_MESSAGE')
        self.assertEqual(inbox.put(dict(msg,global_step=7,collective_id='r:7'))['status'],'INVALID_MESSAGE')
        self.assertEqual(inbox.put(msg)['status'],'OK')
        self.assertTrue(inbox.put(msg)['duplicate'])

    def test_checkpoint_restores_adam_and_next_update(self):
        config=dict(task='synthetic',device='cpu',optimizer='adam',seed=0,global_batch=8,steps=3)
        a=TrainSession(config)
        x,y=a.task.make_data(config)
        for step in range(2):
            c=a.forward_backward(x[step],y[step],materialize_cpu_gradients=False)
            a.apply_global_gradients(c['device_gradients'],str(step))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.npz';save_state(a,path)
            b=TrainSession(config);load_state(b,path)
        self.assertEqual(a.digest(),b.digest())
        for s in (a,b):
            c=s.forward_backward(x[2],y[2],materialize_cpu_gradients=False)
            s.apply_global_gradients(c['device_gradients'],'next')
        self.assertEqual(a.digest(),b.digest())
        self.assertEqual(b.optimizer.t,3)


if __name__=='__main__':
    unittest.main()
