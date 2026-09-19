"""Existing ResNet18 adapted to controlled distributed road regression."""

import numpy as np

from ...core.device import asnumpy, get_device, xp
from ...core.graph import Graph
from ...core.node import Variable
from ...layers.layer import Conv2D
from ...layers.resnet import BatchNorm2d, ResNet18
from ...ops.convolution import MaxPool2d_Op
from ...ops.loss import MSELoss
from ...utils.initializers import make_initializer
from .donkey_cnn import DonkeyCnnTask


class DonkeyResNet18Task(DonkeyCnnTask):
    name = 'donkey_resnet18'

    def build_model(self, config, device='cpu'):
        if config.get('bn_mode', 'frozen') != 'frozen':
            raise ValueError('Distributed comparison requires common frozen BN statistics')
        backend = 'cuda_native_cublas' if get_device() == 'cuda' else 'numpy'
        backend = config.get('conv_backend') or backend
        model = ResNet18(in_channels=3, output_dim=2, stem=config.get('resnet_stem', 'imagenet'),
                         base_width=int(config.get('base_width', 64)), dropout=0,
                         initializer=make_initializer('kaiming_normal', seed=int(config.get('seed', 0))),
                         name='resnet18_donkey')
        layers = [('stem.conv', model.stem_conv), ('stem.bn', model.stem_bn)]
        for stage_id, stage in enumerate((model.layer1, model.layer2, model.layer3, model.layer4), 1):
            for block_id, block in enumerate(stage):
                prefix = f'layer{stage_id}.{block_id}'
                layers += [(f'{prefix}.{name}', getattr(block, name)) for name in ('conv1', 'bn1', 'conv2', 'bn2')]
                if block.downsample is not None:
                    layers += [(prefix+'.downsample.conv', block.downsample_conv),
                               (prefix+'.downsample.bn', block.downsample_bn)]
        params, buffers = {}, {}
        for name, layer in layers:
            if isinstance(layer, Conv2D):
                layer.backend = backend
                params[name+'.kernel'], params[name+'.bias'] = layer.kernel, layer.b
            elif isinstance(layer, BatchNorm2d):
                params[name+'.gamma'], params[name+'.beta'] = layer.gamma, layer.beta
                layer.running_mean = xp.asarray(layer.running_mean, dtype=xp.float32)
                layer.running_var = xp.asarray(layer.running_var, dtype=xp.float32)
                buffers[name+'.running_mean'] = layer.running_mean
                buffers[name+'.running_var'] = layer.running_var
        params.update({'fc.weight': model.fc_W, 'fc.bias': model.fc_b})
        for name, node in params.items():
            node.name = name
            node.value = xp.asarray(node.value, dtype=xp.float32)
        if config.get('bn_statistics'):
            with np.load(config['bn_statistics'], allow_pickle=False) as saved:
                if set(saved.files) != set(buffers):
                    raise ValueError('BN statistics names differ')
                for name, target in buffers.items():
                    if saved[name].shape != target.shape or not np.isfinite(saved[name]).all():
                        raise ValueError('invalid BN statistics '+name)
                    target[...] = xp.asarray(saved[name], dtype=xp.float32)
        model.eval()  # Fixed moments; gamma/beta and all convolution/FC weights remain trainable.
        x = Variable(xp.zeros((1, 3, int(config.get('image_h', 120)), int(config.get('image_w', 160))), xp.float32), name='input')
        y = Variable(xp.zeros((1, 2), xp.float32), name='target')
        logits = model(x)
        loss = MSELoss(logits, y)
        graph = Graph(loss, optimize=False)
        for node in graph.nodes:
            if isinstance(node, MaxPool2d_Op):
                node.backend = backend
        self.model, self.buffers = model, buffers
        self.model_metadata = {'architecture': 'MyFlows.layers.resnet.ResNet18', 'base_width': model.widths[0],
                               'stage_widths': model.widths, 'stem': model.stem_type, 'residual_blocks': 8,
                               'parameters': sum(int(node.value.size) for node in params.values()),
                               'bn_mode': 'frozen moments; trainable gamma/beta', 'dropout': 0,
                               'bn_statistics': config.get('bn_statistics'), 'dtype': 'float32'}
        return {'x': x, 'y': y, 'logits': logits, 'loss': loss, 'graph': graph, 'params': params,
                'kind': 'regression', 'backend': backend, 'buffers': buffers}
