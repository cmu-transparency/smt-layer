"""Matched CROWN verification with durable per-example results and resume."""
import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import sys
import time

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class PairModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.extractor = model.extractor
        self.fc1, self.fc2, self.fc3 = model.fc1, model.fc2, model.fc3

    def forward(self, images):
        features = torch.cat([self.extractor(images[:, :1]),
                              self.extractor(images[:, 1:2])], dim=1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(features)))))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='One or more checkpoints; pass both for a matched comparison.')
    parser.add_argument('--examples', type=int, default=20)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--eps', type=float, nargs='+', default=[0, 1/255, 2/255, 4/255, 8/255])
    parser.add_argument('--output', default='verification_results.csv')
    parser.add_argument('--bound-method', '--smt-bound-method', dest='bound_method',
                        choices=['CROWN', 'CROWN-Optimized'], default='CROWN-Optimized',
                        help='The same method is used for BOTH models.')
    parser.add_argument('--alpha-iterations', type=int, default=20)
    parser.add_argument('--alpha-max-time', type=float, default=30,
                        help='Soft limit per bound direction, in seconds.')
    parser.add_argument('--example-timeout', type=float, default=180,
                        help='Wall-clock worker limit per example, including imports and model loading.')
    parser.add_argument('--threads', type=int, default=2)
    return parser.parse_args()


def load_test_examples(count, seed):
    dataset = torchvision.datasets.MNIST('data/mnist', train=False, download=True)
    rng = random.Random(seed)
    by_class = {d: (dataset.targets == d).nonzero().flatten().tolist() for d in range(10)}
    examples = []
    while len(examples) < count:
        pairs = list(itertools.product(range(10), repeat=2))
        rng.shuffle(pairs)
        for left, right in pairs:
            indices = [rng.choice(by_class[left]), rng.choice(by_class[right])]
            images = dataset.data[indices].float().div(255)
            examples.append((images, left + right, indices))
            if len(examples) == count:
                break
    return examples


def bits(value):
    return [int(c) for c in format(value, '05b')]


def possible_bit_assignments(lower, upper):
    return itertools.product(*[[True] if lo >= 0 else [False] if hi < 0 else [False, True]
                               for lo, hi in zip(lower, upper)])


def sum_bits(assignment):
    # Exactly the theory in MNISTAdder: two unsigned four-bit operands.
    left = sum(int(b) * 2**(3-i) for i, b in enumerate(assignment[:4]))
    right = sum(int(b) * 2**(3-i) for i, b in enumerate(assignment[4:]))
    return bits(left + right)


def verify_example(checkpoint, image, target, eps, settings):
    from scripts.mnist_addition import MNISTAdder
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    torch.set_num_threads(settings['threads'])
    model_type = checkpoint['model_type']
    model = MNISTAdder(use_maxsmt=checkpoint.get('maxsat_backward', False)).cpu().eval()
    model.load_state_dict(checkpoint['model_state'])
    images = image.unsqueeze(0)
    with torch.no_grad():
        clean = (PairModel(model)(images) if model_type == 'baseline' else
                 model((images[:, :1], images[:, 1:2]), return_sat=True))
    clean_bits = (clean.flatten() >= 0).int().tolist()
    correct = clean_bits == bits(target)
    if eps == 0:
        return dict(clean_correct=correct, certified_stable=True, certified_correct=correct,
                    ambiguous_bits=0, status='certified', lower_bounds='', upper_bounds='',
                    bound_seconds=0.0)

    # Each independent image is a batch item for SMT. The baseline needs its
    # entire learned head in the graph; disable sparse intermediate bounds in
    # BOTH paths to avoid the BoundSlice sparse-patch issue.
    bound_input = images if model_type == 'baseline' else images.reshape(2, 1, 28, 28)
    prefix = PairModel(model) if model_type == 'baseline' else model.extractor
    start = time.monotonic()
    bounded = BoundedModule(prefix, bound_input, device='cpu', bound_opts={
        'sparse_intermediate_bounds': False,
        'optimize_bound_args': {'iteration': settings['alpha_iterations'],
                                'max_time': settings['alpha_max_time']},
    })
    perturbation = PerturbationLpNorm(norm=float('inf'), eps=eps,
                                    x_L=(bound_input-eps).clamp(0, 1),
                                    x_U=(bound_input+eps).clamp(0, 1))
    lower, upper = bounded.compute_bounds(
        x=(BoundedTensor(bound_input, perturbation),), method=settings['bound_method'])
    bound_seconds = time.monotonic() - start
    if (lower is None or upper is None or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all() or not (lower <= upper).all()):
        raise ValueError('Invalid or missing bounds')
    lo, hi = lower.detach().flatten().tolist(), upper.detach().flatten().tolist()
    ambiguous = sum(l < 0 <= u for l, u in zip(lo, hi))
    if model_type == 'baseline':
        stable = all(l >= 0 if b else u < 0 for b, l, u in zip(clean_bits, lo, hi))
    else:
        stable = all(sum_bits(a) == clean_bits for a in possible_bit_assignments(lo, hi))
    return dict(clean_correct=correct, certified_stable=stable,
                certified_correct=correct and stable, ambiguous_bits=ambiguous,
                status='certified' if stable else 'inconclusive',
                lower_bounds=json.dumps(lo), upper_bounds=json.dumps(hi), bound_seconds=bound_seconds)


def worker(result_queue, checkpoint_path, image, target, eps, settings, log_path):
    # Library diagnostics stay in a per-task log; the parent prints progress.
    with open(log_path, 'w', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            result_queue.put(verify_example(checkpoint, image, target, eps, settings))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_queue.put({'status': 'error', 'error': str(exc)})


def run_task(checkpoint_path, image, target, eps, settings, log_path):
    context = mp.get_context('spawn')
    result_queue = context.Queue()
    process = context.Process(target=worker,
                              args=(result_queue, checkpoint_path, image, target, eps, settings, log_path))
    start = time.monotonic()
    process.start()
    try:
        process.join(settings['example_timeout'])
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            result = {'status': 'timeout'}
        else:
            try:
                result = result_queue.get(timeout=2)
            except queue.Empty:
                result = {'status': 'error', 'error': f'Worker exit code {process.exitcode}'}
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
        result_queue.close()
    return {'clean_correct': '', 'certified_stable': False, 'certified_correct': False,
            'ambiguous_bits': '', 'lower_bounds': '', 'upper_bounds': '',
            'bound_seconds': '', 'error': '', **result, 'seconds': time.monotonic()-start}


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'w') as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_summary(path, records):
    groups = {}
    for result in records.values():
        groups.setdefault((result['checkpoint'], result['eps']), []).append(result)
    rows = []
    for (checkpoint, eps), results in groups.items():
        n = len(results)
        clean_evaluated = sum(r['clean_correct'] != '' for r in results)
        rows.append(dict(checkpoint=checkpoint, model_type=results[0]['model_type'], eps=eps,
                         completed_examples=n,
                         clean_evaluated=clean_evaluated,
                         clean_correct=sum(r['clean_correct'] is True for r in results),
                         clean_accuracy=(sum(r['clean_correct'] is True for r in results)/clean_evaluated
                                         if clean_evaluated else ''),
                         certified_stability=sum(r['certified_stable'] for r in results)/n,
                         certified_accuracy=sum(r['certified_correct'] for r in results)/n,
                         certified=sum(r['status']=='certified' for r in results),
                         inconclusive=sum(r['status']=='inconclusive' for r in results),
                         timeouts=sum(r['status']=='timeout' for r in results),
                         errors=sum(r['status']=='error' for r in results),
                         total_seconds=sum(r['seconds'] for r in results)))
    if rows:
        temp = path.with_suffix(path.suffix+'.tmp')
        with open(temp, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)


def main():
    args = parse_args()
    if args.examples < 1 or args.alpha_iterations < 1 or min(args.alpha_max_time, args.example_timeout, args.threads) <= 0:
        raise ValueError('Counts and runtime limits must be positive.')
    if any(not 0 <= eps <= 1 for eps in args.eps):
        raise ValueError('Use finite epsilon values between zero and one.')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix('.resume.json')
    details_path = output.with_name(output.stem+'_examples.csv')
    logs = output.with_name(output.stem+'_logs')
    logs.mkdir(exist_ok=True)
    settings = {k: getattr(args, k) for k in
                ['bound_method', 'alpha_iterations', 'alpha_max_time', 'example_timeout', 'threads']}
    checkpoints = []
    for path in args.checkpoint:
        path = str(Path(path).resolve())
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        kind = checkpoint['model_type']
        if kind not in ['baseline', 'smt'] or checkpoint.get('maxsat_forward', False):
            raise ValueError('Only baseline and ordinary SMT checkpoints are supported.')
        checkpoints.append({'path': path, 'sha256': digest(path), 'model_type': kind})
    examples = load_test_examples(args.examples, args.sample_seed)
    manifest = {'version': 1, 'script_sha256': digest(__file__), 'settings': settings,
                'checkpoints': checkpoints, 'eps': args.eps,
                'sample_indices': [indices for _, _, indices in examples],
                'sample_seed': args.sample_seed,
                'image_sha256': hashlib.sha256(torch.stack([x for x, _, _ in examples]).numpy().tobytes()).hexdigest(),
                'packages': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'auto-lirpa']}}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['manifest'] != manifest:
            raise ValueError('Resume settings, data, code, or checkpoints differ. Use a new --output filename.')
        print(f"Resuming: {len(state['records'])} tasks already saved.", flush=True)
    else:
        if output.exists() or details_path.exists():
            raise ValueError('Output already exists without matching resume state. Use a new --output filename.')
        state = {'manifest': manifest, 'records': {}}
        atomic_json(state_path, state)
    try:
        for eps in args.eps:
            for index, (image, target, indices) in enumerate(examples):
                for number, checkpoint in enumerate(checkpoints):
                    key = f'{number}:{index}:{eps!r}'
                    if key in state['records']:
                        continue
                    print(f"{checkpoint['model_type']} eps={eps:.8g} example {index+1}/{len(examples)} starting (CPU)", flush=True)
                    log_path = logs / f'model{number}_example{index}_eps{eps:.8g}.log'
                    result = run_task(checkpoint['path'], image, target, eps, settings, str(log_path))
                    result = {'checkpoint': checkpoint['path'], 'model_type': checkpoint['model_type'],
                              'example_index': index, 'left_mnist_index': indices[0],
                              'right_mnist_index': indices[1], 'target': target, 'eps': eps,
                              **settings, **result, 'log': str(log_path)}
                    state['records'][key] = result
                    atomic_json(state_path, state)
                    save_csv(details_path, list(state['records'].values()))
                    write_summary(output, state['records'])
                    print(f"  {result['status']} in {result['seconds']:.1f}s; saved", flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted; rerun the same command to resume completed tasks.', flush=True)
    save_csv(details_path, list(state['records'].values()))
    write_summary(output, state['records'])
    print(f'Results: {output}; examples: {details_path}', flush=True)


def save_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(path.suffix+'.tmp')
    with open(temp, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


if __name__ == '__main__':
    main()
