import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import nbformat
import pytest
import torch

from model.architectures import Forecaster
from model.experiment import file_digest, load_protocol

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('architecture', ['A', 'B', 'C'])
def test_notebook_full_model_export(tmp_path, monkeypatch, architecture):
    notebook = nbformat.read(ROOT / 'training/forecasting_colab.ipynb', as_version=4)
    nbformat.validate(notebook)
    source = next(c.source for c in notebook.cells if c.id == 'complete-model-export')
    protocol_path = ROOT / 'experiments/forecasting-v3-approved.json'
    protocol = load_protocol(protocol_path)
    suite = tmp_path / 'suite'
    suite.mkdir()
    content = tmp_path / 'content'
    content.mkdir()
    downloads = []
    colab = types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(download=downloads.append)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original = Forecaster(architecture, protocol_path).eval()
        for seed in protocol['training']['seeds']:
            run = suite / f'{architecture}-{seed}'
            run.mkdir()
            checkpoint = run / 'best.pt'
            torch.save({'state_dict': original.state_dict(),
                        'protocol_sha256': protocol['protocol_sha256']}, checkpoint)
            (run / 'result.json').write_text(json.dumps({
                'architecture': architecture, 'seed': seed, 'checkpoint': 'best.pt',
                'checkpoint_sha256': file_digest(checkpoint), 'parameter_count': 1,
                'validation': {'overall': {'mae': 2, 'smape_percent': 3}},
            }))
        def notebook_path(value):
            return content if str(value) == '/content' else Path(value)
        namespace = dict(Path=notebook_path, suite=suite, protocol=protocol, torch=torch,
                         PROJECT_ROOT=ROOT, PROTOCOL_PATH=protocol_path)
        exec(compile(source, '<export cell>', 'exec'), namespace)
        assert len(downloads) == 1
        archive = Path(downloads[0])
        original_bytes = archive.read_bytes()
        unpacked = tmp_path / 'extracted'
        with zipfile.ZipFile(archive) as bundle:
            assert 'model.pt' in bundle.namelist()
            assert 'test.json' not in bundle.namelist()
            assert not any(name.endswith('.npz') or '.env' in name for name in bundle.namelist())
            bundle.extractall(unpacked)
        # A separate process must load without the original repo on its import path.
        check = subprocess.run([sys.executable, '-I', '-c',
            'import sys; sys.path.insert(0, sys.argv[1]); '
            'from load_model import load_model; import torch; '
            'torch.set_num_threads(1); m=load_model(); '
            'assert m(torch.zeros(1,52,3),torch.zeros(1,dtype=torch.int64),'
            'torch.zeros(1,dtype=torch.int64)).shape == (1,12)', str(unpacked)],
            cwd=unpacked, capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
        exec(compile(source, '<export cell>', 'exec'), namespace)
        assert len(downloads) == 1
        assert archive.read_bytes() == original_bytes
        # A final status exports saved test scores without evaluating any arrays.
        (suite / 'selection.json').write_text(json.dumps({'architecture': architecture}))
        (suite / 'test.json').write_text(json.dumps({'overall': {'mae': 999}}))
        exec(compile(source, '<export cell>', 'exec'), namespace)
        assert len(downloads) == 2
        with zipfile.ZipFile(downloads[-1]) as bundle:
            assert 'test.json' in bundle.namelist()
        (suite / 'selection.json').write_text(json.dumps({'architecture': None}))
        exec(compile(source, '<export cell>', 'exec'), namespace)
        with zipfile.ZipFile(downloads[-1]) as bundle:
            assert 'test.json' not in bundle.namelist()
            assert json.loads(bundle.read('metrics.json'))['status'] == 'no_qualifying_champion'
    finally:
        torch.set_num_threads(old_threads)


def test_export_comparison_uses_three_seed_validation_means(tmp_path):
    notebook = nbformat.read(ROOT / 'training/forecasting_colab.ipynb', as_version=4)
    source = next(c.source for c in notebook.cells if c.id == 'complete-model-export')
    comparison = source[source.index('completed = []'):source.index('selection_path =')]
    protocol = load_protocol(ROOT / 'experiments/forecasting-v3-approved.json')
    def add(architecture, errors):
        for seed, error in zip(protocol['training']['seeds'], errors):
            run = tmp_path / f'{architecture}-{seed}'
            run.mkdir(exist_ok=True)
            (run / 'result.json').write_text(json.dumps({
                'parameter_count': 1, 'test': {'mae': -error},
                'validation': {'overall': {'mae': error, 'smape_percent': 3}},
            }))
    scope = dict(suite=tmp_path, protocol=protocol, torch=torch,
                 read_json=lambda path: json.loads(path.read_text()))
    add('A', [4, 4, 4])
    add('B', [1, 9, 9])
    add('C', [0])  # Incomplete experiments cannot win.
    exec(comparison, scope)
    assert scope['winner']['architecture'] == 'A'
    add('B', [3, 3, 3])
    exec(comparison, scope)
    assert scope['winner']['architecture'] == 'B'
