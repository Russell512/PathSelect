"""Publication parity, portable API, and result-export integrity checks."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pathselect import PathSelect
from scripts.paper64_common import evidence
from scripts.paper64_manifest import expected_cells

ROOT = Path(__file__).resolve().parents[1]


def test_scientific_source_matches_accepted_study():
    ref = json.loads((ROOT / 'docs/scientific_core.json').read_text())
    for path, expected in ref['unchanged_file_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    source = (ROOT / 'scripts/paper64_common.py').read_text()
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name, expected in ref['common_source_sha256'].items():
        assert hashlib.sha256(ast.get_source_segment(source, nodes[name]).encode()).hexdigest() == expected


def test_exported_results_are_original_bytes():
    root = ROOT / 'results/paper_top64'
    ref = json.loads((root / 'PROVENANCE.json').read_text())
    for item in ref['files']:
        assert hashlib.sha256((root / item['file']).read_bytes()).hexdigest() == item['sha256']


def test_physical_manifest_counts_and_unique_paths():
    cells = expected_cells()
    assert len(cells) == len({r['prediction_path'] for r in cells}) == 8560
    assert sum(r['split'] == 'val' for r in cells) == 3680
    assert sum(r['split'] == 'test' for r in cells) == 4880


def test_public_api_matches_original_evidence_and_resumes(tmp_path):
    g = torch.Generator().manual_seed(23)
    texts = F.normalize(torch.randn(2, 2, 512, generator=g), dim=-1)
    model = PathSelect(texts)
    slides = [torch.randn(80, 512, generator=g) for _ in range(6)]
    labels = torch.arange(6) % 2
    model.fit_task(0, slides, labels)
    for z in slides:
        out = model.predict_slide(z)
        reference, indices, _ = evidence(z, texts[0], None, 'top64')
        assert torch.equal(out['indices'], indices)
        assert torch.equal(out['evidence'], reference)
        assert out['predicted_class'] in [0, 1]
    path = tmp_path / 'model.pt'
    model.save(path)
    restored = PathSelect.load(path)
    assert torch.equal(model.predict_slide(slides[0])['scores'], restored.predict_slide(slides[0])['scores'])
    for obj in [model, restored]:
        obj.fit_task(1, slides, labels + 2)
    assert torch.equal(model._subtype_state['W'], restored._subtype_state['W'])
    assert torch.equal(model._router_state['W'], restored._router_state['W'])
    for n in [1, 37, 64, 65]:
        out = model.predict_slide(torch.randn(n, 512, generator=g))
        assert len(out['indices']) == min(n, 64)
        assert torch.isfinite(out['evidence']).all()
    with pytest.raises(ValueError):
        model.predict_slide(slides[0], 'invalid')
    with pytest.raises(ValueError):
        model.fit_task(0, slides, labels)


def test_frozen_texts_match_benchmark():
    ref = json.loads((ROOT / 'configs/benchmark_hashes.json').read_text())
    for task, info in ref['text'].items():
        path = ROOT / 'cache/text' / f'f_txt_{task}.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        d = torch.load(path, weights_only=True)
        assert d['class_names'] == info['class_names']
        assert list(d['f_txt'].shape) == [2, 512]
