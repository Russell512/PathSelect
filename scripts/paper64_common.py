"""Independent FINAL selection comparison; no historical pipeline context."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from selector.cil_ops import four_round, one_shot, mean_norm, ORDERS
from selector.flat_selector import text_nav_feats
from selector.i6_expert import I6Expert, zscore
from selector.incremental_ridge import aug

TASKS = list(ORDERS['reverse'])
VARIANTS = ['A_four16', 'A_top64', 'B_four16', 'B_top64']
SPLITS = ['train', 'val', 'test']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(map(str, ids))).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def finite_tree(x):
    if isinstance(x, torch.Tensor):
        if not bool(torch.isfinite(x).all()):
            raise ValueError('nonfinite artifact tensor')
    elif isinstance(x, dict):
        for v in x.values(): finite_tree(v)
    elif isinstance(x, (list, tuple)):
        for v in x: finite_tree(v)
    elif isinstance(x, float) and not np.isfinite(x):
        raise ValueError('nonfinite artifact number')


def save_artifact(path, payload, metadata):
    path = Path(path)
    finite_tree(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save({'metadata': metadata, 'payload': payload}, temp)
        with open(temp, 'rb') as f: os.fsync(f.fileno())
        os.replace(temp, path)
        marker = {'sha256': file_hash(path), 'metadata': metadata, 'completed': now()}
        write_json(path.with_suffix(path.suffix + '.done.json'), marker)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return marker


def load_artifact(path, metadata):
    path = Path(path)
    marker = path.with_suffix(path.suffix + '.done.json')
    if not marker.exists(): return None
    m = read_json(marker)
    if m['metadata'] != metadata or not path.exists() or file_hash(path) != m['sha256']:
        raise ValueError(f'invalid artifact provenance/checksum: {path}')
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if obj['metadata'] != metadata: raise ValueError(f'metadata mismatch: {path}')
    finite_tree(obj['payload'])
    return obj['payload']


def config():
    from selector.text_encoder import load_config
    cfg = load_config()
    exp = yaml.safe_load((ROOT / 'configs/experiments/paper64.yaml').read_text())
    if cfg.get('machine') is None: raise ValueError('NAVCIL_MACHINE is required')
    if cfg['threads'] != exp['threads']: raise ValueError('thread setting mismatch')
    return cfg, exp


def environment():
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions())
    return {'python': sys.version, 'packages': packages, 'torch': str(torch.__version__),
            'device': 'cpu', 'feature_dtype': 'float32', 'ridge_dtype': 'float64',
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'os': platform.platform(), 'cpu_arch': platform.machine()}


def setup(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def source_identity():
    manifest = ROOT / 'SOURCE.json'
    if manifest.exists():
        identity = read_json(manifest)
        for name, expected in identity['files'].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f'Runtime source snapshot changed: {name}')
        return identity
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    entries = {p: file_hash(ROOT / p) for p in files if p and not p.startswith('outputs/')}
    return {'implementation_commit': revision, 'source_fingerprint': digest(entries), 'files': entries}


def freeze_protocol(out, cfg, exp):
    p = {**exp, 'study_kind': 'development-followup after prior test exploration',
         'primary': 'stage4 equal-task CIL accuracy; paper(original) / reverse reported separately',
         'analysis': 'A/random: five seeds averaged within fold; ten-fold sample SD ddof=1; deterministic rows once',
         'masked_primary': 'true task selects evidence AND final class pair (oracle WP)',
         'masked_secondary': 'predicted-task evidence; final pair restricted to true task',
         'gamma_selection': 'oracle val WP: tasks within stage, 8 order/stage cells, folds, A seeds equally weighted; exact tie larger gamma; freeze before formal test',
         'fixed_comparisons': 'router/readout 1e-3; top64 minus four16; lambda=1.5; shared A heads; own train evidence and W',
         'random': 'SHA256 canonical [seed,fold,slide_id] first 8 bytes little endian; numpy default_rng without replacement; task/order/stage invariant',
         'CI_scope': 'test sampling conditional on fitted models; overlapping CV training sets, seeds and orders are not independent',
         'global_classes': ['ESAD','ESCC','CCRCC','PRCC','IDC','ILC','LUAD','LUSC'],
         'tolerances': {'float32_atol':1e-6,'ridge_SR_atol':1e-8,'ridge_W_atol':1e-6,'ridge_logits_atol':1e-8,'ridge_residual_atol':1e-8,'ridge_rtol':1e-7,'metric_atol':1e-12,'tie':1e-12,'indices':'exact','predictions':'exact; investigate near ties'},
         'data_protection': 'Dataset and historical experiments read-only; new runs use a separate output namespace.',
         'retry_resume': 'completed checksummed heads/tasks only; incomplete heads restart full five epochs; max 2 automatic scheduler retries per unit',
         'timing': 'same physical node, CPU 4 threads; rotating A/B top64/four16 on same slides; discard warmup; real online compute; shared peak labelled process-wide',
         'all_patches_read_and_scored': True}
    path = out / 'protocol/protocol.json'
    with lock(out / 'protocol/.lock'):
        if path.exists() and read_json(path) != p: raise ValueError('protocol already frozen with other settings')
        if not path.exists():
            write_json(path, p)
            atomic_bytes(out / 'protocol/protocol.md', ('# PathSelect 論文 top64 補實驗\n\n本批為既有探索後的正式比較；不宣稱全新確認性研究。\n\n' + json.dumps(p, ensure_ascii=False, indent=2) + '\n').encode())
        atomic_bytes(out / 'protocol/SHA256', (file_hash(path) + '\n').encode())
    return p


def feature_check(Z, sid):
    if Z.ndim != 2 or Z.shape[1] != 512 or Z.shape[0] == 0:
        raise ValueError(f'invalid feature shape {sid}: {tuple(Z.shape)}')
    if Z.dtype != torch.float32 or not bool(torch.isfinite(Z).all()):
        raise ValueError(f'invalid feature dtype/values {sid}: {Z.dtype}')


def select_indices(Z, score, mode, budget=64):
    if score.shape != (Z.shape[0],) or not bool(torch.isfinite(score).all()):
        raise ValueError('invalid patch score')
    if mode == 'top64': idx = one_shot(score, budget=budget)
    elif mode == 'four16': idx = four_round(Z, score, lam=1.5, budget=64, step=16)
    else: raise ValueError(mode)
    if len(idx) != min(len(Z), budget) or len(idx.unique()) != len(idx) or idx.min() < 0 or idx.max() >= len(Z):
        raise ValueError('selection cardinality/index failure')
    return idx


def evidence(Z, ft, head, mode):
    score = zscore(text_nav_feats(Z, ft)[:, 0]) if head is None else head(Z, ft)
    t0 = time.perf_counter()
    idx = select_indices(Z, score, mode)
    selection_s = time.perf_counter() - t0
    return mean_norm(Z, idx), idx, selection_s


class Readout:
    """One growing subtype ridge; only current task data enter add_task."""
    def __init__(self, gamma=1e-3):
        self.gamma = gamma
        self.A = torch.zeros(513, 513, dtype=torch.float64)
        self.cols = {}
        self.tasks = []

    def add_task(self, task, V, y):
        if task in self.tasks or set(y.tolist()) != {2*task, 2*task+1}:
            raise ValueError('duplicate task or incorrect labels')
        X = aug(V)
        self.A += X.T @ X
        for c in (2*task, 2*task+1): self.cols[c] = X[y == c].sum(0)
        self.tasks.append(task)

    def state(self):
        classes = sorted(self.cols)
        B = torch.stack([self.cols[c] for c in classes], 1)
        W = torch.linalg.solve(self.A + self.gamma * torch.eye(513, dtype=torch.float64), B)
        return {'A': self.A.clone(), 'B': B, 'W': W, 'classes': classes, 'tasks': list(self.tasks), 'gamma': self.gamma}


def route(Z, router):
    scores = aug(mean_norm(Z).reshape(1, -1))[0] @ router['W']
    positions = sorted(router['tasks'])
    ordered = scores[[router['tasks'].index(p) for p in positions]]
    return positions[int(ordered.argmax())], ordered


def classify(V, readout, task):
    scores = aug(V.reshape(1, -1))[0] @ readout['W']
    pair = scores[[readout['classes'].index(2*task), readout['classes'].index(2*task+1)]]
    return 2*task + int(pair.argmax()), scores, float(abs(pair[0]-pair[1]))


def random_indices(n, seed, fold, slide_id, budget=64):
    key = json.dumps([int(seed), int(fold), str(slide_id)], separators=(',', ':')).encode()
    derived = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    return torch.tensor(np.random.default_rng(derived).choice(n, min(n, budget), replace=False), dtype=torch.long)
