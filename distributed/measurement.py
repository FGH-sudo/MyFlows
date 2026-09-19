"""Optional, durable experiment measurements; never transports model updates."""

import json
from pathlib import Path
import time

import numpy as np

from ..core.device import asnumpy, xp


def account(client, kind, request_bytes, response_bytes, rpc_s, encode_s=0., decode_s=0., payload=0):
    fields = {'request_bytes': request_bytes, 'response_bytes': response_bytes,
              'rpc_observed_s': rpc_s, 'encode_s': encode_s, 'decode_s': decode_s,
              'outgoing_gradient_bytes': payload}
    for key, value in fields.items():
        client.traffic[key] = client.traffic.get(key, 0) + value
        typed = str(kind) + '_' + key
        client.traffic[typed] = client.traffic.get(typed, 0) + value


def timed(session, label, function):
    start = time.perf_counter()
    if session.config.get('profile') and session.device == 'cuda':
        first, last = xp.cuda.Event(), xp.cuda.Event()
        first.record()
        result = function()
        last.record()
        last.synchronize()
        gpu_s = float(xp.cuda.get_elapsed_time(first, last)) / 1000
    else:
        result = function()
        gpu_s = None
    return result, {label + '_wall_s': time.perf_counter() - start, label + '_gpu_s': gpu_s}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_state(session, path, extra=None, gradients=None, local_gradients=None):
    opt = session.optimizer_state_cpu()
    arrays = {'parameter::' + k: v for k, v in session.named_parameters_cpu().items()}
    arrays.update({'buffer::' + k: asnumpy(v, copy=True) for k, v in session.buffers.items()})
    for kind in ('v', 's'):
        arrays.update({kind + '::' + k: v for k, v in opt.pop(kind).items()})
    if gradients is not None:
        arrays.update({'gradient::' + k: asnumpy(v, copy=True) for k, v in gradients.items()})
    if local_gradients is not None:
        arrays.update({'local_gradient::' + k: asnumpy(v, copy=True) for k, v in local_gradients.items()})
    arrays['metadata'] = np.array(json.dumps({'optimizer': opt, 'version': session.local_version,
                                             **(extra or {})}))
    np.savez(path, **arrays)


def load_state(session, path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f['metadata']))
        saved_buffers = {k.split('::', 1)[1]: f[k] for k in f.files if k.startswith('buffer::')}
        if set(saved_buffers) != set(session.buffers):
            raise ValueError('checkpoint model buffers differ')
        for name, value in saved_buffers.items():
            session.buffers[name][...] = xp.asarray(value)
        session.load_parameters({k.split('::', 1)[1]: f[k] for k in f.files if k.startswith('parameter::')})
        opt = meta['optimizer']
        for kind in ('v', 's'):
            opt[kind] = {k.split('::', 1)[1]: f[k] for k in f.files if k.startswith(kind + '::')}
        session.load_optimizer_state(opt)
        session.local_version = int(meta['version'])
    return meta


def evaluate(session, data, split, batch=128):
    x, y = data[split]
    stats = {}
    started = time.perf_counter()
    for start in range(0, len(x), batch):
        xb, yb = x[start:start+batch], y[start:start+batch]
        session.x.value, session.y.value = xp.asarray(xb), xp.asarray(yb)
        session.graph.forward()
        part = session.task.metric_stats(session, xb, yb)
        loss = float(asnumpy(session.loss_node.value))
        part.setdefault('loss', {'sum': loss * len(xb), 'count': len(xb), 'kind': 'mean'})
        for name, item in part.items():
            total = stats.setdefault(name, {'sum': 0., 'count': 0, 'kind': item['kind']})
            total['sum'] += item['sum']
            total['count'] += item['count']
    from .metrics_reduce import values_from_stats
    return {'split': split, 'samples': len(x), **values_from_stats(stats),
            'evaluation_s': time.perf_counter()-started}


class Recorder:
    """Per-rank records plus metadata-only readiness barriers outside train time."""
    def __init__(self, session, data, config, rank, stop=None):
        self.session, self.data, self.config, self.rank, self.stop = session, data, config, rank, stop
        self.root = Path(config['artifact_dir']) if config.get('artifact_dir') else None
        self.rows = []
        self.clients = []
        self.started = time.perf_counter()
        if self.root:
            self.directory = self.root / f'rank-{rank}'
            self.directory.mkdir(parents=True, exist_ok=True)
            self.steps_file = (self.directory / 'steps.jsonl').open('w', encoding='utf-8', buffering=1)
            self.epochs_file = (self.directory / 'epochs.jsonl').open('w', encoding='utf-8', buffering=1)

    def barrier(self, name):
        if not self.root:
            return
        directory = self.root / 'barriers'
        write_json(directory / f'{name}-{self.rank}.json', {'ready_s': time.perf_counter()})
        deadline = time.perf_counter() + float(self.config.get('timeout', 120))
        while not all((directory / f'{name}-{r}.json').exists() for r in range(self.config['train_workers'])):
            if self.stop is not None and self.stop.is_set():
                raise RuntimeError('measurement barrier aborted')
            if time.perf_counter() > deadline:
                raise TimeoutError(f'measurement barrier {name}')
            time.sleep(.002)

    def prepare(self, cursor, shard=None):
        if not self.root:
            return
        for r in range(self.config['train_workers']):
            if r == self.rank:
                x, y = cursor.get(0, 0, shard)
                self.session.forward_backward(x, y, materialize_cpu_gradients=False)
            self.barrier(f'warmup-{r}')
        initial = {'digest': self.session.digest(), 'schema': self.session.schema(),
                   'optimizer': self.session.optimizer_meta(), 'device': self.session.device,
                   'backends': self.session.actual_conv_backends(),
                   'parameters': {k: {'dtype': str(v.value.dtype), 'module': type(v.value).__module__}
                                  for k, v in self.session.params.items()}}
        if getattr(self.session.task, 'model_metadata', None):
            initial['model'] = self.session.task.model_metadata
        if self.session.device == 'cuda':
            from cupy_backends.cuda.libs import nvrtc
            initial['cuda'] = {'runtime_version': xp.cuda.runtime.runtimeGetVersion(),
                               'driver_version': xp.cuda.runtime.driverGetVersion(),
                               'nvrtc_version': nvrtc.getVersion(), 'cupy': xp.__version__}
        if self.rank == 0:
            save_state(self.session, self.root / 'initial.npz')
            if self.config.get('evaluate') and isinstance(self.data, dict):
                initial['train'] = evaluate(self.session, self.data, 'train', self.config.get('eval_batch', self.config['global_batch']))
                initial['val'] = evaluate(self.session, self.data, 'val', self.config.get('eval_batch', self.config['global_batch']))
        write_json(self.directory / 'initial.json', initial)
        self.barrier('prepared')
        write_json(self.directory / 'prepare.json', {'prepare_s': time.perf_counter()-self.started})

    def start_epoch(self, epoch):
        if self.root:
            self.rows = []
            self.barrier(f'epoch-{epoch}-ready')
            self.epoch_start = time.perf_counter()

    def traffic(self):
        totals = {}
        for client in self.clients:
            for key, value in getattr(client, 'traffic', {}).copy().items():
                totals[key] = totals.get(key, 0) + value
        return totals

    def record(self, epoch, step, step_start, data_s, computed, sync_s=0., confirm_s=0.,
               digest_s=0., traffic_before=None, gradients=None):
        if not self.root:
            return
        ended = time.perf_counter()
        row = {'rank': self.rank, 'epoch': epoch, 'step': step, 'start_s': step_start, 'end_s': ended,
               'step_wall_s': ended-step_start, 'data_s': data_s, 'gradient_sync_s': sync_s,
               'sync_update_s': ended-computed['backward_end_s'],
               'update_confirm_s': confirm_s, 'digest_s': digest_s,
               'n_samples': computed['n_samples'], 'loss': computed['loss'],
               **computed.get('timings', {}), **getattr(self.session, 'update_timings', {})}
        for key, value in self.traffic().items():
            row[key] = value - (traffic_before or {}).get(key, 0)
        row['gradient_payload_bytes'] = sum(v.nbytes for v in computed['device_gradients'].values())
        row['backends'] = computed.get('backends')
        self.steps_file.write(json.dumps(row) + '\n')
        self.rows.append(row)
        if self.config.get('numerical_snapshots'):
            save_state(self.session, self.directory / f'step-{step:04d}.npz', gradients=gradients,
                       local_gradients=computed['device_gradients'] if self.config.get('record_local_gradients') else None)

    def end_epoch(self, epoch, step):
        if not self.root:
            return
        ended = self.rows[-1]['end_s']
        row = {'rank': self.rank, 'epoch': epoch, 'steps': len(self.rows),
               'start_s': self.epoch_start, 'end_s': ended, 'epoch_train_wall_s': ended-self.epoch_start,
               'samples': sum(r['n_samples'] for r in self.rows)}
        for key in self.rows[0]:
            if key.endswith('_s') and key not in ('start_s','end_s') or key.endswith('_bytes'):
                values = [r.get(key) for r in self.rows if r.get(key) is not None]
                row[key] = sum(values) if values else None
        self.barrier(f'epoch-{epoch}-updated')
        if self.rank == 0 and self.config.get('save_epoch_checkpoints'):
            save_state(self.session, self.root/f'epoch-{epoch:03d}.npz', {'next_data_epoch': epoch+1+int(self.config.get('data_epoch', 0))})
        if self.rank == 0 and self.config.get('evaluate') and isinstance(self.data, dict):
            row['val'] = evaluate(self.session, self.data, 'val', self.config.get('eval_batch', self.config['global_batch']))
        self.epochs_file.write(json.dumps(row) + '\n')
        self.barrier(f'epoch-{epoch}-evaluated')

    def finish(self):
        if not self.root:
            return
        self.barrier('final-committed')
        save_state(self.session, self.directory / 'final.npz',
                   {'epochs': self.config.get('epochs'), 'seed': self.config.get('seed')})
        state = {'digest': self.session.digest(), 'optimizer': self.session.optimizer_meta(),
                 'version': self.session.local_version, 'device': self.session.device,
                 'backends': self.session.actual_conv_backends(), 'traffic': self.traffic(),
                 'optimizer_arrays': {kind: {name: {'dtype':str(value.dtype), 'module':type(value).__module__}
                     for name, node in self.session.params.items()
                     if (value := getattr(self.session.optimizer, kind, {}).get(node)) is not None}
                     for kind in ('v', 's')}}
        if self.session.device == 'cuda':
            state['allocator_reserved_bytes_at_finish'] = xp.get_default_memory_pool().total_bytes()
        if self.rank == 0 and self.config.get('evaluate') and isinstance(self.data, dict):
            state['train'] = evaluate(self.session, self.data, 'train', self.config.get('eval_batch', self.config['global_batch']))
            state['val'] = evaluate(self.session, self.data, 'val', self.config.get('eval_batch', self.config['global_batch']))
            if self.config.get('final_test', True):
                state['test'] = evaluate(self.session, self.data, 'test', self.config.get('eval_batch', self.config['global_batch']))
        write_json(self.directory / 'final.json', state)
        self.barrier('final-saved')
        self.steps_file.close()
        self.epochs_file.close()
