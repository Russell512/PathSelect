"""Enumerate required physical prediction cells before any formal test evaluation."""
from pathlib import Path
from scripts.paper64_common import ROOT, atomic_bytes, file_hash, read_json, write_json, now, source_identity
from scripts.paper64_run import GRID, rep_names, manifest
import json


def expected_cells():
    rows = []
    for fold in range(1, 11):
        for seed in [0, 42, 43, 44, 45, 46]:
            for split in ['val', 'test']:
                for rep in rep_names(seed):
                    if split == 'val' and rep not in ['MEAN', 'B64', 'A64']:
                        continue
                    modes = ['RDG', 'LIN8'] if rep == 'MEAN' else ['RDG', 'COS'] if rep in ['B64', 'A64'] else ['RDG']
                    for mode in modes:
                        gammas = [None] if mode == 'COS' else GRID if rep in ['MEAN', 'B64', 'A64'] else [.001]
                        for gamma in gammas:
                            config = f'{rep}_{mode}_g{gamma if gamma is not None else "none"}'
                            for order in ['paper', 'reverse']:
                                for stage in range(1, 5):
                                    rows.append({'fold': fold, 'worker_seed': seed, 'representation': rep,
                                                 'split': split, 'readout': mode, 'gamma_wp': gamma, 'gamma_tp': .001,
                                                 'order': order, 'stage': stage,
                                                 'prediction_path': f'predictions/{split}/fold{fold:02d}_seed{seed}/{config}/{order}/stage{stage}.pt'})
    assert len(rows) == 8560 and sum(r['split'] == 'test' for r in rows) == 4880
    return rows


def initialize(out):
    manifest(out)
    source = source_identity()
    rows = [{**r, 'source_fingerprint': source['source_fingerprint'],
             'implementation_commit': source['implementation_commit']} for r in expected_cells()]
    atomic_bytes(out / 'protocol/expected_physical_cells.jsonl',
                 ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows).encode())
    target = out / 'manifests/literature_rows.json'
    atomic_bytes(target, (ROOT / 'results/paper_top64/literature_rows.json').read_bytes())


def freeze(out):
    path = out / 'protocol/formal_freeze.json'
    selected = out / 'protocol/selection.json'
    selection = read_json(selected)
    assert len(selection['validation_files']) == 60
    if path.exists():
        assert read_json(path)['selection_sha256'] == file_hash(selected)
        return
    assert not list((out / 'predictions/test').rglob('*.pt')), 'freeze must precede formal test'
    write_json(path, {'frozen_at': now(), 'selection_sha256': file_hash(selected),
                     'selection': selection['selected'], 'physical_cells': 8560,
                     'validation_cells': 3680, 'test_cells': 4880,
                     'physical_manifest_sha256': file_hash(out / 'protocol/expected_physical_cells.jsonl'),
                     'expanded_manifest_sha256': file_hash(out / 'protocol/expanded_manifest.json'),
                     'aliases_are_not_independent_experiments': True,
                     'test_gamma_sweep': 'descriptive/post hoc; cannot change selection'})
