#!/usr/bin/env python3
"""Calibrate official random-init FastGS 30k on an idle GPU and prepared work dir."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from reconstruction.resources import BASELINE, SPEC, verify_vendor
from reconstruction.protocol import split_protocol
from reconstruction.pipeline import gpu_environment, protocol_digest, apply_reference, save_report
from reconstruction.runtime import write_json


def active_processes(uuid):
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).splitlines()
    return {int(line.split(',')[1]) for line in lines if line.split(',')[0].strip() == uuid}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True, help='work/ from a pipeline run')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--report', type=Path, help='Fill scores in an existing report.json without rerunning reconstruction')
    p.add_argument('--reuse', type=Path, help='Reuse a matching calibration.json instead of running calibration')
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    work, out = a.work.resolve(), a.output.resolve()
    split_protocol(json.loads((work/'split.json').read_text()))
    env, uuid = gpu_environment(a.gpu)
    before = verify_vendor()
    if a.reuse:
        from reconstruction.pipeline import validate_reference
        reference = json.loads(a.reuse.read_text())
        validate_reference(reference, uuid, protocol_digest(work))
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'calibration.json', reference)
        if a.report:
            report = json.loads(a.report.read_text())
            save_report(a.report.resolve().parent, apply_reference(report, reference))
        return
    if active_processes(uuid):
        raise RuntimeError('Calibration requires an idle GPU; retry when it is free')
    # Some containerized drivers cannot expose foreign PIDs; also check utilization.
    for _ in range(3):
        usage = int(subprocess.check_output(['nvidia-smi', '-i', uuid, '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
        if usage > 5:
            raise RuntimeError('GPU is computing; calibration must wait for an idle GPU')
        time.sleep(1)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(a.repeats):
        if active_processes(uuid):
            raise RuntimeError('GPU became busy before calibration')
        dest = out/f'repeat-{repeat:02d}'
        command = [sys.executable, str(BASELINE/'module_b/train_from_ply_fastgs.py'), '-s', str(work), '-m', str(dest),
                   '--eval', '-r', '1', '--sh_degree', '3', '--iterations', '30000', '--position_lr_max_steps', '30000',
                   '--densify_until_iter', '15000', '--init_random', '--mult', '0.5', '--quiet']
        foreign = set()
        with (out/f'reference-{repeat:02d}.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                foreign |= active_processes(uuid)-{process.pid}
                time.sleep(1)
        if process.returncode or foreign:
            write_json(out/'calibration.json', dict(status='invalid', foreign_pids=sorted(foreign), returncode=process.returncode))
            raise RuntimeError('Calibration failed or shared GPU activity was detected')
        row = json.loads((dest/'module_b.json').read_text())
        if row['iterations'] != 30000 or not row['init_mode'].startswith('random:') or row['mult'] != .5 or row['trainer'] != 'fastgs':
            raise RuntimeError('Unexpected reference protocol')
        rows.append(row)
    if before != verify_vendor():
        raise RuntimeError('Frozen source changed')
    seconds = statistics.median(r['train_seconds'] for r in rows)
    write_json(out/'calibration.json', dict(status='complete', gpu_uuid=uuid, commit=SPEC['commit'],
        reference_trainer='fastgs', iterations=30000, mult=.5, repeats=a.repeats,
        protocol_sha256=protocol_digest(work), reference_training_seconds_median=seconds,
        T_iter_seconds=seconds/30000, runs=rows, foreign_processes_observed=False, formal_acceptance=False))
    if a.report:
        report = json.loads(a.report.read_text())
        save_report(a.report.resolve().parent, apply_reference(report, json.loads((out/'calibration.json').read_text())))
    print(f'FastGS 30k: {seconds:.3f}s; T_iter: {seconds/30000:.8f}s')


if __name__ == '__main__':
    main()
