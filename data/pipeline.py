# -*- coding: utf-8 -*-
"""
生产者-消费者数据流水线（拓展 4）。

Worker 进程负责 IO + 解码 + 可选增强；主进程从 Queue 取 batch。
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Callable, Iterator, Sequence


def _worker_loop(
    index_queue: mp.Queue,
    out_queue: mp.Queue,
    dataset: Sequence,
    batch_size: int,
    load_fn: Callable,
    transform_fn: Callable | None,
    stop_event: mp.Event,
):
  while not stop_event.is_set():
    try:
      idx = index_queue.get(timeout=0.5)
    except queue.Empty:
      continue
    if idx is None:
      break
    sample = load_fn(dataset[idx])
    if transform_fn is not None:
      sample = transform_fn(sample)
    out_queue.put(sample)


class MultiprocessDataLoader:
  """简易多进程 DataLoader。"""

  def __init__(
      self,
      dataset: Sequence,
      batch_size: int,
      *,
      num_workers: int = 0,
      shuffle: bool = False,
      seed: int = 0,
      load_fn: Callable | None = None,
      transform_fn: Callable | None = None,
  ):
    self.dataset = dataset
    self.batch_size = max(1, int(batch_size))
    self.num_workers = max(0, int(num_workers))
    self.shuffle = bool(shuffle)
    self.seed = int(seed)
    self.load_fn = load_fn
    self.transform_fn = transform_fn

  def __len__(self) -> int:
    n = len(self.dataset)
    return (n + self.batch_size - 1) // self.batch_size

  def _iter_inprocess(self) -> Iterator:
    if self.load_fn is None:
      raise ValueError("MultiprocessDataLoader 需要非空的 load_fn")

    import random

    indices = list(range(len(self.dataset)))
    if self.shuffle:
      rng = random.Random(self.seed)
      rng.shuffle(indices)
    batch_x, batch_meta = [], []
    for idx in indices:
      sample = self.load_fn(self.dataset[idx])
      if self.transform_fn:
        sample = self.transform_fn(sample)
      batch_x.append(sample[0])
      batch_meta.append(sample[1] if len(sample) > 1 else None)
      if len(batch_x) >= self.batch_size:
        yield batch_x, batch_meta
        batch_x, batch_meta = [], []
    if batch_x:
      yield batch_x, batch_meta

  def __iter__(self) -> Iterator:
    if self.load_fn is None:
      raise ValueError("MultiprocessDataLoader 需要非空的 load_fn")
    if self.num_workers == 0:
      yield from self._iter_inprocess()
      return

    ctx = mp.get_context("spawn")
    index_q: mp.Queue = ctx.Queue(maxsize=self.num_workers * 4)
    out_q: mp.Queue = ctx.Queue(maxsize=self.num_workers * 2)
    stop_event = ctx.Event()

    import random

    indices = list(range(len(self.dataset)))
    if self.shuffle:
      rng = random.Random(self.seed)
      rng.shuffle(indices)
    for idx in indices:
      index_q.put(idx)
    for _ in range(self.num_workers):
      index_q.put(None)

    workers = []
    for _ in range(self.num_workers):
      p = ctx.Process(
          target=_worker_loop,
          args=(
              index_q,
              out_q,
              self.dataset,
              self.batch_size,
              self.load_fn,
              self.transform_fn,
              stop_event,
          ),
      )
      p.daemon = True
      p.start()
      workers.append(p)

    received = 0
    total_samples = len(indices)
    batch_x, batch_meta = [], []
    try:
      while received < total_samples:
        try:
          sample = out_q.get(timeout=30.0)
        except queue.Empty:
          break
        received += 1
        batch_x.append(sample[0])
        batch_meta.append(sample[1] if len(sample) > 1 else None)
        if len(batch_x) >= self.batch_size:
          yield batch_x, batch_meta
          batch_x, batch_meta = [], []
      if batch_x:
        yield batch_x, batch_meta
    finally:
      stop_event.set()
      for p in workers:
        p.join(timeout=2.0)
        if p.is_alive():
          p.terminate()
