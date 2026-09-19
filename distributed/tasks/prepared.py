"""Read immutable prepared datasets without duplicating images per worker."""

import json
from pathlib import Path

import numpy as np


class IndexedImages:
    def __init__(self, images, indices):
        self.images = images
        self.indices = indices
        self.shape = (len(indices), *images.shape[1:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return np.asarray(self.images[self.indices[item]], dtype=np.float32) / np.float32(255)


def load_prepared(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    images = np.load(directory / 'images.npy', mmap_mode='r')
    labels = np.load(directory / 'labels.npy', mmap_mode='r')
    split = {'meta': meta}
    for name in ('train', 'val', 'test'):
        indices = np.load(directory / f'{name}_indices.npy')
        split[name] = (IndexedImages(images, indices), labels[indices])
    return split
