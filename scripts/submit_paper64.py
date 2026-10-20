"""Create an immutable source snapshot and submit a configurable Slurm phase."""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper64_ops import ROOT, append, canonical, command, locked, now, read, sha, write

PHASES = ['tests', 'preflight', 'smoke', 'pilot', 'prepare', 'select', 'test', 'timing', 'report']


def snapshot(out, testing=False):
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    rev = head.stdout.strip() if head.returncode == 0 else 'UNCOMMITTED_TEST_ONLY'
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    if (dirty or head.returncode) and not testing:
        raise RuntimeError('Formal snapshots require a clean, tested commit')
    files = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT, text=True).split('\0')) - {''}
    machine = os.environ.get('NAVCIL_MACHINE')
    if machine:
        files.add(f'configs/machine_{machine}.yaml')
    entries = {p: sha(ROOT / p) for p in sorted(files)}
    identity = {'implementation_commit': rev, 'source_fingerprint': canonical(entries),
                'files': entries, 'testing_uncommitted': bool(dirty)}
    dest = out / 'source' / (('tests-' if testing else '') + identity['source_fingerprint'][:16])
    if not dest.exists():
        temp = dest.with_name(dest.name + '.tmp-' + str(os.getpid()))
        temp.mkdir(parents=True)
        for name in entries:
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write(temp / 'SOURCE.json', identity)
        os.replace(temp, dest)
    for name, checksum in entries.items():
        assert sha(dest / name) == checksum
    write(out / 'environment' / ('test-source.json' if testing else 'source.json'), identity)
    return dest, identity


def submit(args):
    out = args.output.resolve()
    with locked(out / 'slurm/submission.lock'):
        if args.source:
            source = args.source.resolve()
            identity = read(source / 'SOURCE.json')
            for name, checksum in identity['files'].items():
                assert sha(source / name) == checksum
        else:
            source, identity = snapshot(out, args.phase == 'tests')
        if args.phase == 'prepare':
            pilot = read(out / 'manifests/pilot_receipt.json')
            assert pilot['pass'] and pilot['spec']['identity']['source_fingerprint'] == identity['source_fingerprint']
            for name, checksum in pilot['artifacts'].items():
                assert sha(name) == checksum
        if args.phase == 'test':
            assert (out / 'protocol/formal_freeze.json').is_file()
        actual = out.with_name(out.name + '_' + args.phase + '_' + identity['source_fingerprint'][:8]) if args.phase in ['smoke', 'pilot'] else out
        for name in ['logs', 'protocol', 'manifests', 'environment']:
            (actual / name).mkdir(parents=True, exist_ok=True)
        if args.phase in ['smoke', 'pilot']:
            for name in ['manifests/data.json', 'manifests/preflight.json', 'protocol/protocol.json']:
                shutil.copy2(out / name, actual / name)
        stamp = str(time.time_ns())
        receipt = out / 'manifests' / f'{args.phase}_{stamp}_receipt.json'
        spec = {'phase': args.phase, 'output': str(actual), 'study_output': str(out),
                'source': str(source), 'identity': identity, 'folds': args.folds,
                'receipt': str(receipt), 'created': now()}
        specpath = out / 'slurm' / f'{args.phase}-{stamp}.json'
        write(specpath, spec)
        batch = out / 'slurm' / f'{args.phase}-{stamp}.sbatch'
        python = str(Path(args.python).absolute())
        batch.write_text('#!/bin/bash\nset -euo pipefail\n'
                         'export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'
                         + 'cd ' + shlex.quote(str(source)) + '\nexec srun --ntasks=1 '
                         + shlex.join([python, 'scripts/paper64_worker.py', str(specpath)]) + '\n')
        cmd = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
               f'--mem={args.memory_gb}G', f'--time={args.minutes}',
               '--job-name=pathselect-' + args.phase, '--output=' + str(out / 'logs' / f'{args.phase}-%j.log')]
        for key in ['partition', 'account', 'qos']:
            if getattr(args, key):
                cmd.append('--' + key + '=' + getattr(args, key))
        if args.gpus_per_node:
            cmd.append('--gpus-per-node=' + str(args.gpus_per_node))
        host = actual / 'environment/compute.json'
        node = args.node or (read(host)['hostname'] if host.exists() and args.phase != 'tests' else None)
        if node:
            cmd.append('--nodelist=' + node)
        if args.dependency:
            cmd.append('--dependency=afterok:' + args.dependency)
        cmd.append(str(batch))
        result = command(cmd)
        if result['returncode']:
            append(out / 'slurm/submission_errors.jsonl', result)
            raise RuntimeError(result['stderr'])
        record = {'job_id': result['stdout'].strip().split(';')[0], 'phase': args.phase,
                  'source': str(source), 'output': str(actual), 'command': cmd,
                  'spec': str(specpath), 'receipt': str(receipt), 'submitted': now(),
                  'folds': args.folds, 'device': 'cpu', 'gpus_allocated': args.gpus_per_node,
                  'gpus_used': 0}
        append(out / 'slurm/jobs.jsonl', record)
        print(record, flush=True)
        return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=PHASES, required=True)
    p.add_argument('--folds', type=lambda s: list(map(int, s.split(','))), default=[1])
    p.add_argument('--dependency')
    p.add_argument('--source', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--minutes', type=int, default=240)
    p.add_argument('--memory-gb', type=int, default=32)
    p.add_argument('--gpus-per-node', type=int, default=0, help='Only if required by site admission; models still use CPU')
    p.add_argument('--partition', default=os.environ.get('PATHSELECT_SLURM_PARTITION'))
    p.add_argument('--account', default=os.environ.get('PATHSELECT_SLURM_ACCOUNT'))
    p.add_argument('--qos')
    p.add_argument('--node')
    args = p.parse_args()
    if not args.folds or any(f not in range(1, 11) for f in args.folds):
        p.error('folds must be IDs 1–10')
    submit(args)


if __name__ == '__main__':
    main()
