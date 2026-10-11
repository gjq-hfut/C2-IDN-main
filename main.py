"""Neighbor-sampled GNN training/evaluation using the existing dataset.py.
"""
import argparse
import csv
import json
import os
import random
import pickle
import resource
import sys
import time
import tempfile
from pathlib import Path
from contextlib import nullcontext

# Must precede CUDA context creation.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('MKL_SERVICE_FORCE_INTEL', '0')
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from dataset import get_data, get_pretain_user_data, get_pretain_item_data, add_dataset_arguments
from model import twoD_MIR, MacridVAE
from deployment_profile import (ResourceProfiler, benchmark_model, load_bundle,
                                run_scaling, synchronize, write_json)

PROFILE = None


def phase(name, **metadata):
    return PROFILE.phase(name, **metadata) if PROFILE is not None else nullcontext()


def start_memory_measurement(device):
    """Reset GPU peaks; Linux process-lifetime peak RSS cannot be reset here."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def save_memory_report(args, device, stage):
    """Save selected-device allocator peaks and Linux main-process peak RSS."""
    gpu_allocated_gb = None
    gpu_reserved_gb = None
    gpu_name = None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        gpu_allocated_gb = max(torch.cuda.max_memory_allocated(device),
                               PROFILE.global_allocated if PROFILE else 0) / 1e9
        gpu_reserved_gb = max(torch.cuda.max_memory_reserved(device),
                              PROFILE.global_reserved if PROFILE else 0) / 1e9
        gpu_name = torch.cuda.get_device_name(device)

    host_peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9
    report = {
        'stage': stage,
        'model_name': args.model_name,
        'dataset': args.dataset,
        'device': str(device),
        'gpu_name': gpu_name,
        'unit': 'GB',
        'bytes_per_GB': 1_000_000_000,
        'peak_gpu_allocated_gb': gpu_allocated_gb,
        'peak_gpu_reserved_gb': gpu_reserved_gb,
        'peak_host_rss_gb': host_peak_rss_gb,
        'gpu_scope': (
            'Selected-device PyTorch allocator from before data/model '
            'initialization to this report; not total CUDA process memory'
        ),
        'host_scope': (
            'Current Python process lifetime peak RSS through this report; '
            'excludes child-process memory totals'
        ),
        'includes_training': not args.eval_only,
        'includes_pretraining': bool(not args.eval_only and args.pretrain_epochs),
        'includes_validation': True,
        'includes_test': bool(args.eval_only and args.eval_on_test),
        'config': {
            'train_batch_size': args.batch_size,
            'pretrain_batch_size': args.pretrain_batch_size,
            'eval_graph_batch_size': args.eval_graph_batch_size,
            'eval_text_batch_size': args.eval_text_batch_size,
            'eval_user_batch_size': args.eval_user_batch_size,
            'eval_item_batch_size': args.eval_item_batch_size,
            'eval_pair_batch_size': args.eval_pair_batch_size,
            'num_clusters': args.num_clusters,
            'emb_dim': args.emb_dim,
            'graph_layers': args.graph_layers,
            'neighbor_sizes': args.neighbor_sizes,
            'max_subgraph_nodes': args.max_subgraph_nodes,
            'max_subgraph_edges': args.max_subgraph_edges,
            'num_workers': args.num_workers,
            'topk': args.topk,
            'random_seed': args.random_seed,
            'eval_seed': args.eval_seed,
        },
        'software': {
            'python': sys.version.split()[0],
            'pytorch': str(torch.__version__),
            'pytorch_cuda': torch.version.cuda,
        },
    }
    output_path = Path(args.output_dir) / f'{stage}_memory.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)

    print(f'\n[{stage}] Memory report', flush=True)
    if gpu_allocated_gb is not None:
        print(f'  Peak GPU allocated: {gpu_allocated_gb:.4f} GB', flush=True)
        print(f'  Peak GPU reserved:  {gpu_reserved_gb:.4f} GB', flush=True)
    else:
        print('  GPU peaks: not applicable (CPU run)', flush=True)
    print(f'  Peak host RSS:      {host_peak_rss_gb:.4f} GB', flush=True)
    if args.num_workers > 0:
        print('  Note: host RSS excludes worker memory. Use --num-workers 0 '
              'for a single-process comparison.', flush=True)
    print(f'  Saved to: {output_path}\n', flush=True)
    return report


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Sparse CUDA reductions may be nondeterministic; do not make them fatal.
    torch.use_deterministic_algorithms(True, warn_only=True)


def binary_csr(matrix):
    result = sp.csr_matrix(matrix, dtype=np.float32, copy=True)
    if not np.isfinite(result.data).all():
        raise ValueError('Interaction matrix contains non-finite values.')
    result.sum_duplicates()
    result.data = (result.data > 0).astype(np.float32)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def decode_batch(data, num_users, num_items, item_offset):
    user_ids, item_ids, labels = (torch.as_tensor(x).detach().cpu().reshape(-1) for x in data)
    if not (user_ids.numel() == item_ids.numel() == labels.numel()):
        raise ValueError('Batch user IDs, item IDs and labels must have equal lengths.')
    if ((user_ids != user_ids.long()).any() or (item_ids != item_ids.long()).any()):
        raise ValueError('Node IDs must be integers.')
    user_ids, item_ids = user_ids.long(), item_ids.long() - item_offset
    labels = labels.float()
    if ((user_ids < 0) | (user_ids >= num_users)).any():
        raise ValueError('User IDs must be zero-based and smaller than num_users.')
    if ((item_ids < 0) | (item_ids >= num_items)).any():
        raise ValueError('Item IDs are out of range; check --item-id-offset (default: num_users).')
    if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Expected binary 0/1 labels; get_data must use binary=True.')
    return user_ids, item_ids, labels


def sample_unseen_items(user_ids, train_csr, rng):
    """Uniform sampling in the complement; no rejection loop or dense U x I mask.

Returns retained row positions and one negative for each retained occurrence.
Users who have consumed every item are omitted because no BPR negative exists.
"""
    users = np.asarray(user_ids, dtype=np.int64)
    negative = np.empty(users.size, dtype=np.int64)
    keep = np.ones(users.size, dtype=bool)
    n_items = train_csr.shape[1]
    for user in np.unique(users):
        positions = np.flatnonzero(users == user)
        positives = train_csr.indices[train_csr.indptr[user]:train_csr.indptr[user + 1]]
        missing = n_items - len(positives)
        if missing == 0:
            keep[positions] = False
            continue
        ranks = rng.integers(missing, size=len(positions))
        # Convert a rank in the complement into its actual item ID.
        offsets = np.searchsorted(positives - np.arange(len(positives)), ranks, side='right')
        negative[positions] = ranks + offsets
    positions = np.flatnonzero(keep)
    return positions, negative[positions]


def make_optimizer(params, args, lr=None, weight_decay=0.):
    kwargs = dict(lr=args.lr if lr is None else lr, weight_decay=weight_decay)
    if args.optimizer.lower() == 'adam':
        return optim.Adam(params, **kwargs)
    if args.optimizer.lower() == 'sgd':
        return optim.SGD(params, **kwargs)
    raise ValueError('--optimizer must be Adam or SGD.')


def plain_metadata(value):
    """Keep checkpoint metadata compatible with weights_only=True."""
    if isinstance(value, np.generic):
        return plain_metadata(value.item())
    if isinstance(value, dict):
        return {str(key): plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Unsupported checkpoint metadata type: {type(value).__name__}')


def save_checkpoint(path, model, **metadata):
    torch.save({'model_state': model.state_dict(), **plain_metadata(metadata)}, path)


def read_checkpoint(path, device):
    """Read checkpoints created by your own runs of this script only.

    Older saves contain NumPy validation scalars. They need ordinary pickle
    loading on older PyTorch versions. Never use this compatibility loader
    for checkpoints from unknown/untrusted sources.
    """
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        message = str(exc).lower()
        if 'numpy' not in message or not any(word in message for word in ('scalar', 'dtype')):
            raise
        print(f'Loading legacy NumPy metadata from your training checkpoint: {path}', flush=True)
        state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or 'model_state' not in state:
        raise ValueError('Expected a checkpoint containing model_state from this training script.')
    return state


def load_checkpoint(path, model, device):
    state = read_checkpoint(path, device)
    model.load_state_dict(state['model_state'])
    return model


def pretrain(args, model, loader, device, kind):
    if args.pretrain_epochs == 0:
        return model
    optimizer = make_optimizer(model.parameters(), args, lr=args.pretrain_lr,
                               weight_decay=.1 if kind == 'user' else .99)
    path = Path(args.output_dir) / f'{kind}_vae_best.pt'
    best_loss, patience = float('inf'), 0
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total, count = 0., 0
        for index, texts in tqdm(loader, desc=f'Pretrain {kind} {epoch}/{args.pretrain_epochs}'):
            index = torch.as_tensor(index, device=device, dtype=torch.long).reshape(-1)
            if ((index < 0) | (index >= model.num_user)).any():
                raise ValueError(f'{kind} pretraining loader must return local entity IDs.')
            texts = torch.as_tensor(texts, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = model.calculate_loss(texts, index, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {kind} pretraining loss.')
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.detach().item() * index.numel()
            count += index.numel()
        if not count:
            raise ValueError(f'Empty {kind} pretraining loader.')
        current = total / count
        print(f'{kind} VAE epoch {epoch}: loss={current:.6f}', flush=True)
        if current < best_loss:
            best_loss, patience = current, 0
            save_checkpoint(path, model, epoch=epoch)
        else:
            patience += 1
            if patience >= args.early_stopping:
                break
    return load_checkpoint(path, model, device)


def merge_topk(best_scores, best_ids, block_scores, item_start, k):
    """Merge one candidate block; exact streaming Top-K with item-ID tie breaks."""
    block_ids = np.broadcast_to(np.arange(item_start, item_start + block_scores.shape[1]),
                                 block_scores.shape)
    scores = np.concatenate((best_scores, block_scores), axis=1)
    ids = np.concatenate((best_ids, block_ids), axis=1)
    # Ascending -score, then ascending item ID; only K + block_size columns exist.
    order = np.lexsort((ids, -scores), axis=1)[:, :min(k, scores.shape[1])]
    return np.take_along_axis(scores, order, axis=1), np.take_along_axis(ids, order, axis=1)


@torch.no_grad()
def test(model, heldout_csr, train_csr, user_tfidf, item_tfidf, args, topk):
    """Sampled embeddings + full candidate ranking with bounded score memory.
"""
    model.eval()
    started = time.time()
    cache_parent = Path(args.eval_cache_dir or args.output_dir)
    cache_parent.mkdir(parents=True, exist_ok=True)
    eval_seed = args.random_seed if args.eval_seed is None else args.eval_seed
    cache_bytes = 2 * (model.num_users + model.num_items) * model.num_prototypes * model.emb_size * 4
    print(f'Eval sampled encoding; temporary disk cache: {cache_bytes / (1024 ** 3):.3f} GiB', flush=True)
    with tempfile.TemporaryDirectory(prefix='sampled_eval_', dir=cache_parent) as directory:
        cache = model.build_eval_cache(
            user_tfidf, item_tfidf, directory,
            graph_batch_size=args.eval_graph_batch_size,
            text_batch_size=args.eval_text_batch_size, seed=eval_seed, profiler=PROFILE)
        try:
            with phase('evaluation_ranking_and_metrics'):
                results = evaluate_cached(model, heldout_csr, train_csr, cache, args, topk)
        finally:
            model.close_eval_cache(cache)
    print(f'Evaluation elapsed={time.time() - started:.2f}s; {results}', flush=True)
    return results


def evaluate_cached(model, heldout_csr, train_csr, cache, args, topk):
    totals = {f'{name}@{k}': 0. for k in topk for name in ('recall', 'ndcg')}
    eligible = np.flatnonzero(np.diff(heldout_csr.indptr) > 0)
    if not len(eligible):
        raise ValueError('Evaluation split has no positive interactions.')
    max_k = min(max(topk), model.num_items)
    discounts = 1. / np.log2(np.arange(2, max_k + 2))
    for start in tqdm(range(0, len(eligible), args.eval_user_batch_size), desc='Evaluation'):
        users = eligible[start:start + args.eval_user_batch_size]
        best_scores = np.empty((len(users), 0), dtype=np.float32)
        best_ids = np.empty((len(users), 0), dtype=np.int64)
        user_tensor = torch.as_tensor(users, device=model.device)
        # Small CSR row block, still sparse; no full dense rating matrix.
        seen_rows = train_csr[users]
        for item_start in range(0, model.num_items, args.eval_item_batch_size):
            item_end = min(item_start + args.eval_item_batch_size, model.num_items)
            items = torch.arange(item_start, item_end, device=model.device)
            block = model.predict(user_tensor, items, cache, args.eval_pair_batch_size)
            if not torch.isfinite(block).all():
                raise FloatingPointError('Non-finite prediction scores.')
            scores = block.cpu().numpy()
            seen = seen_rows[:, item_start:item_end].tocoo()
            scores[seen.row, seen.col] = -np.inf
            best_scores, best_ids = merge_topk(best_scores, best_ids, scores, item_start, max_k)
        for local, user in enumerate(users):
            truth = heldout_csr.indices[heldout_csr.indptr[user]:heldout_csr.indptr[user + 1]]
            hits = np.isin(best_ids[local], truth) & np.isfinite(best_scores[local])
            for k in topk:
                cutoff = min(k, model.num_items)
                totals[f'recall@{k}'] += float(hits[:cutoff].sum()) / len(truth)
                dcg = float((hits[:cutoff] * discounts[:cutoff]).sum())
                ideal = discounts[:min(cutoff, len(truth))].sum()
                totals[f'ndcg@{k}'] += float(dcg / ideal)
    return {key: float(value / len(eligible)) for key, value in totals.items()}


def train(args, model, train_loader, train_csr, valid_csr, user_tfidf, item_tfidf, device, topk):
    vae_params = list(model.user_network.parameters()) + list(model.item_network.parameters())
    vae_ids = {id(p) for p in vae_params}
    other_params = [p for p in model.parameters() if id(p) not in vae_ids]
    optimizer = make_optimizer([
        {'params': vae_params, 'lr': args.lr * .1},
        {'params': other_params, 'lr': args.lr}], args)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    path = Path(args.output_dir) / 'model_best.pt'
    rng = np.random.default_rng(args.random_seed)
    best, patience, best_epoch = (-float('inf'), -float('inf')), 0, 0
    monitor = getattr(args, 'monitor_metric', None) or f'recall@{10 if 10 in topk else min(topk)}'
    metric_name, cutoff = monitor.split('@')
    monitor_k = int(cutoff)
    secondary = f'{"ndcg" if metric_name == "recall" else "recall"}@{monitor_k}'
    best_results = None
    offset = model.num_users if args.item_id_offset is None else args.item_id_offset
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        with phase('joint_training_epoch', epoch=epoch):
            model.train()
            total, count, dropped, capped_batches = 0., 0, 0, 0
            for data in tqdm(train_loader, desc=f'Train {epoch}/{args.epochs}'):
                users, items, labels = decode_batch(data, model.num_users, model.num_items, offset)
                negatives = None
                if args.loss_type == 'bpr':
                    positive = labels > 0
                    users, items, labels = users[positive], items[positive], labels[positive]
                    if not users.numel():
                        continue
                    # Tensor.numpy() shares externally owned storage. Some SciPy/
                    # NumPy versions reset WRITEABLE during paired-index broadcasting,
                    # which can fail on those views. Give SciPy owned index arrays.
                    user_index = np.array(users.numpy(), dtype=np.intp, copy=True)
                    item_index = np.array(items.numpy(), dtype=np.intp, copy=True)
                    membership = np.asarray(train_csr[user_index, item_index]).reshape(-1)
                    if not (membership > 0).all():
                        raise ValueError('A positive training sample is missing from train_csr_matrix; check ID mapping.')
                    keep, sampled = sample_unseen_items(user_index, train_csr, rng)
                    dropped += users.numel() - len(keep)
                    users, items, labels = users[keep], items[keep], labels[keep]
                    negatives = torch.as_tensor(sampled, device=device)
                if not users.numel():
                    continue
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.calculate_loss(users, items, labels, user_tfidf, item_tfidf, negatives)
                capped_batches += int(bool(model.last_graph_stats.get('node_cap_hit') or model.last_graph_stats.get('edge_cap_hit')))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch}: {model.last_losses}')
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                total += loss.detach().item() * users.numel()
                count += users.numel()
            if not count:
                raise ValueError('No usable training samples. Check labels, IDs and the selected loss type.')
            print(f'Epoch {epoch}: loss={total / count:.6f}, samples={count}, '
                  f'no-negative={dropped}, capped_batches={capped_batches}, elapsed={time.time() - started:.2f}s; '
                  f'last_batch={model.last_losses}', flush=True)
        results = test(model, valid_csr, train_csr, user_tfidf, item_tfidf, args, [monitor_k])
        score = (results[monitor], results[secondary])
        if score > best:
            best, patience, best_epoch = score, 0, epoch
            best_results = dict(results)
            save_checkpoint(path, model, args=vars(args), epoch=epoch, validation=results)
        else:
            patience += 1
        if getattr(args, 'progress_file', None):
            progress_path = Path(args.progress_file)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'event': 'validation', 'epoch': int(epoch),
                                         'metrics': plain_metadata(results),
                                         'best_epoch': int(best_epoch)}, allow_nan=False) + '\n')
                stream.flush()
        print(f'Best epoch: {best_epoch}; {monitor}/{secondary}={best}', flush=True)
        if patience >= args.early_stopping:
            break
        scheduler.step()
    model = load_checkpoint(path, model, device)
    model.best_validation_results = best_results
    return model


def _run(args):
    global PROFILE
    eval_state = None
    checkpoint_path = None
    if args.eval_only:
        checkpoint_path = Path(args.checkpoint or Path(args.output_dir) / 'model_best.pt')
        eval_state = read_checkpoint(checkpoint_path, 'cpu')
        # Recover architecture, sampler and data settings from the completed run.
        # Explicit CLI arguments override saved settings, for relocated datasets.
        keep = {'eval_only', 'checkpoint', 'output_dir', 'gpu_id', 'num_workers',
                'eval_on_test', 'cli_overrides', 'benchmark_only', 'benchmark_scales',
                'benchmark_bundle', 'profile_training'} | set(args.cli_overrides)
        for name, value in eval_state.get('args', {}).items():
            if hasattr(args, name) and name not in keep:
                setattr(args, name, plain_metadata(value))
        if args.checkpoint and 'output_dir' not in args.cli_overrides:
            args.output_dir = str(checkpoint_path.parent)
    seed_everything(args.random_seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.benchmark_scales or args.gpu_id < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError('Requested GPU does not exist; use --gpu_id 0 or --gpu_id -1 for CPU.')
        device = torch.device(f'cuda:{args.gpu_id}')
    print(f'Device: {device}' + (' (scaling preparation; child processes use requested GPU)' if args.benchmark_scales else ''), flush=True)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    # Legacy peaks are preserved in PROFILE when optional phase counters reset.
    start_memory_measurement(device)
    if args.num_workers > 0:
        print('Memory measurement: main-process RSS only; worker memory is excluded. '
              'Use --num-workers 0 for single-process comparisons.', flush=True)
    if args.benchmark_only or args.benchmark_scales or args.profile_training:
        PROFILE = ResourceProfiler(device, args.profile_rss_interval)
    bundle_info = {}
    with phase('data_loading'):
        if args.benchmark_bundle:
            loaders, matrices, stats, bundle_info = load_bundle(args.benchmark_bundle)
        else:
            loaders, matrices, stats = get_data(args, binary=True, num_workers=args.num_workers, train_shuffle=True)
    train_loader, _, _ = loaders
    train_csr, valid_csr, test_csr = [binary_csr(m) for m in matrices]
    n_users, n_items, n_words = (int(stats[k]) for k in ('num_users', 'num_items', 'num_words'))
    # Keep input features on CPU; fetch only seed rows inside the model.
    for key in ('user_tfidf', 'item_tfidf'):
        if isinstance(stats[key], torch.Tensor):
            stats[key] = stats[key].detach().cpu()
    user_tfidf, item_tfidf = stats['user_tfidf'], stats['item_tfidf']
    if tuple(user_tfidf.shape) != (n_users, n_words) or tuple(item_tfidf.shape) != (n_items, n_words):
        raise ValueError('TF-IDF shape disagrees with the dataset statistics.')
    for split, matrix in zip(('train', 'valid', 'test'), (train_csr, valid_csr, test_csr)):
        if matrix.shape != (n_users, n_items):
            raise ValueError(f'{split} CSR shape disagrees with the dataset statistics.')
    for split, matrix in (('valid', valid_csr), ('test', test_csr)):
        if train_csr.multiply(matrix).nnz:
            raise ValueError(f'{split} has positive pairs also in train; masked ranking would be invalid.')
    topk = sorted(set(int(k) for k in args.topk.split(',')))
    if not topk or min(topk) < 1:
        raise ValueError('--topk must list positive integers.')
    print(f'Dataset: {args.dataset}; users={n_users}; items={n_items}; train_edges={train_csr.nnz}', flush=True)
    dataset_stats = dict(users=n_users, items=n_items, vocabulary=n_words,
                         train_interactions=int(train_csr.nnz),
                         valid_interactions=int(valid_csr.nnz),
                         test_interactions=int(test_csr.nnz),
                         training_density=float(train_csr.nnz / (n_users * n_items)),
                         **bundle_info)
    # Preserve the source-ID mapping alongside checkpoints for later inference.
    if args.eval_only and checkpoint_path is not None:
        mapping_path = checkpoint_path.parent / 'id_mapping.npz'
        if mapping_path.is_file():
            with np.load(mapping_path, allow_pickle=False) as mapping:
                for key in ('user_raw_ids', 'item_raw_ids'):
                    if not np.array_equal(mapping[key], stats[key]):
                        raise ValueError('Evaluation ID mapping differs from the saved training run.')
    if args.benchmark_scales:
        return run_scaling(args, (train_csr, valid_csr, test_csr), stats,
                           eval_state, __file__)
    if not args.eval_only and 'user_raw_ids' in stats and 'item_raw_ids' in stats:
        np.savez(output / 'id_mapping.npz', user_raw_ids=stats['user_raw_ids'],
                 item_raw_ids=stats['item_raw_ids'])
    with phase('vae_model_initialization'):
        user_model = MacridVAE(n_users, n_words, args.emb_dim, args.dropout, args.num_clusters,
                              device, tau=args.user_temperature, anneal_cap=args.rating_anneal_cap, std=args.user_std).to(device)
        item_model = MacridVAE(n_items, n_words, args.emb_dim, args.dropout, args.num_clusters,
                              device, tau=args.item_temperature, anneal_cap=args.text_anneal_cap, std=args.item_std).to(device)
    if args.pretrain_epochs and not args.eval_only:
        user_loader, _ = get_pretain_user_data(args, binary=True, num_workers=args.num_workers, train_shuffle=True)
        with phase('user_vae_pretraining'):
            user_model = pretrain(args, user_model, user_loader, device, 'user')
        item_loader, _ = get_pretain_item_data(args, binary=True, num_workers=args.num_workers, train_shuffle=True)
        with phase('item_vae_pretraining'):
            item_model = pretrain(args, item_model, item_loader, device, 'item')
    with phase('gnn_and_fusion_initialization'):
        model = twoD_MIR(args, args.graph_layers, n_items, n_users, n_words, args.emb_dim,
                        args.dropout, args.num_clusters, device, user_model, item_model,
                        lambda_user=args.lambda_user, lambda_item=args.lambda_item,
                        lambda_align=args.lambda_align, lambda_cor=args.lambda_cor,
                        lambda_reg=args.lambda_reg, training=True)
        model.set_train_graph(train_csr)
    print(f'Neighbor sampling: fanouts={model.fanouts}, '
          f'max_nodes={args.max_subgraph_nodes}, max_edges={args.max_subgraph_edges}', flush=True)
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    with (output / ('eval_config.json' if args.eval_only else 'config.json')).open('w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if args.eval_only:
        with phase('checkpoint_to_model'):
            model.load_state_dict(eval_state['model_state'])
        print(f'Loaded saved epoch {eval_state.get("epoch", "unknown")}; skipping all training.', flush=True)
        del eval_state
    else:
        with phase('joint_training_with_validation'):
            model = train(args, model, train_loader, train_csr, valid_csr, user_tfidf, item_tfidf, device, topk)
        # Includes pretraining and validation inside train(), but excludes the
        # final validation/test calls below. Peak counters survive deallocation.
        save_memory_report(args, device, 'training_pipeline')
    if args.benchmark_only:
        heldout = test_csr if args.benchmark_split == 'test' else valid_csr
        report = benchmark_model(model, train_csr, heldout, user_tfidf, item_tfidf,
                                 args, merge_topk, PROFILE, dataset_stats)
        save_memory_report(args, device, 'inference_process')
        return report
    if args.profile_training and not args.eval_only:
        write_json(output / 'training_profile.json',
                   dict(dataset_stats=dataset_stats, config=vars(args),
                        resources=PROFILE.report()))
    if getattr(args, 'tuning_run', False) and not args.eval_only:
        # Reuse the validation result that selected the saved checkpoint.
        validation = model.best_validation_results
    else:
        validation = test(model, valid_csr, train_csr, user_tfidf, item_tfidf, args, topk)
    results = {f'v_{k}': v for k, v in validation.items()}
    if args.eval_on_test:
        testing = test(model, test_csr, train_csr, user_tfidf, item_tfidf, args, topk)
        results.update({f't_{k}': v for k, v in testing.items()})
    with (output / 'results.json').open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    with (output / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results))
        writer.writeheader()
        writer.writerow(results)
    if args.eval_only:
        # Only an independent eval-only run is labeled as inference memory.
        save_memory_report(args, device, 'inference_process')
    return results



def run(args):
    global PROFILE
    PROFILE = None
    try:
        return _run(args)
    finally:
        if PROFILE is not None:
            PROFILE.close()
            PROFILE = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default='citeulike', help='Dataset')
    parser.add_argument('-gl', '--graph_layers', type=int, default=2, help='graph_layers number')
    parser.add_argument('-dim', '--emb_dim', type=int, default=32, help='Embedding size')
    parser.add_argument('-ep', '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, default=400, help='Batch data size in training/testing')
    parser.add_argument('-dr', '--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-optim', '--optimizer', type=str, default='Adam', help='Optimization algorithm')
    parser.add_argument('-lr', '--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-ncl', '--num_clusters', type=int, default=4, help='Number of clusters, i.e., number of prototypes or number of user interests')
    parser.add_argument('-utau', '--user_temperature', type=float, default=0.1, help='Temperature value')
    parser.add_argument('-itau', '--item_temperature', type=float, default=0.1, help='Temperature value')
    parser.add_argument('-ranneal', '--rating_anneal_cap', type=float, default=200, help='Annealing value for KL divergence for rating')
    parser.add_argument('-tanneal', '--text_anneal_cap', type=float, default=200, help='Annealing value for KL divergence for text')
    parser.add_argument('-ustd', '--user_std', type=float, default=0.1, help='Temperature value')
    parser.add_argument('-istd', '--item_std', type=float, default=0.1, help='Temperature value')
    parser.add_argument('-fname', '--file_name', type=str, default=None, help='Selected params from args to construct file name, separated by comma')
    parser.add_argument('-gpu', '--gpu_id', type=int, default=1, help='GPU ID')
    parser.add_argument('-mdl', '--model_name', type=str, default='C2-IDN', help='Model name')
    parser.add_argument('-topk', '--topk', type=str, default='10,20,30,40,50', help='List of top-k to evaluate, separated by comma')
    parser.add_argument('-eot', '--eval_on_test', type=int, default=0, help='1 to evaluate on test set. 0 to only evaluate on validation set')
    parser.add_argument('-rs', '--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('-ldr_user', '--lambda_user', type=float, default=0.001, help='Lambda in loss term for user')
    parser.add_argument('-ldr_item', '--lambda_item', type=float, default=0.001, help='Lambda in loss term for item')
    parser.add_argument('-ldr_align', '--lambda_align', type=float, default=0.001, help='Lambda in loss term for align')
    parser.add_argument('-ldr_cor', '--lambda_cor', type=float, default=0.001, help='Lambda in loss term for cor for regularization')
    parser.add_argument('-ldr_reg', '--lambda_reg', type=float, default=0.0001, help='Lambda in loss term for regularization')
    parser.add_argument('-es', '--early_stopping', type=int, default=10, help='Number of steps for early stopping')
    parser.set_defaults(gpu_id=0, batch_size=128)
    parser.add_argument('--item-id-offset', type=int, default=None,
                        help='Loader item offset: default num_users; use 0 for local IDs.')
    parser.add_argument('--loss-type', choices=['bpr', 'bce'], default='bpr')
    parser.add_argument('--graph-normalization', choices=['symmetric', 'none'], default='symmetric')
    parser.add_argument('--edge-chunk-size', type=int, default=65536)
    parser.add_argument('--neighbor-sizes', default='10,5',
                        help='Neighbors per hop from seeds outward; repeat last value for additional layers.')
    parser.add_argument('--max-subgraph-nodes', type=int, default=20000,
                        help='Hard total user+item node limit per sampled subgraph; seeds are always kept.')
    parser.add_argument('--max-subgraph-edges', type=int, default=100000,
                        help='Hard user-item edge limit per sampled subgraph.')
    parser.add_argument('--eval-graph-batch-size', type=int, default=128)
    parser.add_argument('--eval-seed', type=int, default=None,
                        help='Fixed evaluation sampling seed; defaults to random_seed.')
    parser.add_argument('--eval-cache-dir', default=None,
                        help='Temporary disk cache parent; defaults to output-dir. Files are removed after evaluation.')
    parser.add_argument('--cor-sample-size', type=int, default=256,
                        help='Max batch nodes for distance correlation; 0 uses all.')
    parser.add_argument('--pretrain-epochs', type=int, default=None,
                        help='Defaults to --epochs; 0 skips VAE pretraining.')
    parser.add_argument('--pretrain-lr', type=float, default=.01)
    parser.add_argument('--lr-step-size', type=int, default=2)
    parser.add_argument('--lr-gamma', type=float, default=.1)
    parser.add_argument('--grad-clip', type=float, default=0., help='0 disables gradient clipping.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--eval-user-batch-size', type=int, default=32)
    parser.add_argument('--eval-item-batch-size', type=int, default=256)
    parser.add_argument('--eval-text-batch-size', type=int, default=64)
    parser.add_argument('--eval-pair-batch-size', type=int, default=4096)
    parser.add_argument('--output-dir', default='./runs/c2_idn_sampled')
    parser.add_argument('--monitor-metric', default=None,
                        help='Checkpoint/early-stopping metric: recall@K or ndcg@K.')
    parser.add_argument('--progress-file', default=None,
                        help='Append JSONL validation events after each joint-training epoch.')
    parser.add_argument('--tuning-run', action='store_true',
                        help='Validation-only trial; reuse best-epoch metrics without repeated final evaluation.')
    parser.add_argument('--eval-only', action='store_true',
                        help='Load an existing model_best.pt and skip VAE pretraining and joint training.')
    parser.add_argument('--checkpoint', default=None,
                        help='Your own training checkpoint for --eval-only; default output-dir/model_best.pt.')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Fresh eval-only process: offline encoding, single-user cached requests, phase memory; skip quality evaluation.')
    parser.add_argument('--benchmark-users', type=int, default=100)
    parser.add_argument('--benchmark-repeats', type=int, default=3)
    parser.add_argument('--benchmark-warmup', type=int, default=5)
    parser.add_argument('--benchmark-topn', type=int, default=20)
    parser.add_argument('--benchmark-candidates', type=int, default=0,
                        help='0: full item catalog. Otherwise seeded random candidate subset.')
    parser.add_argument('--benchmark-split', choices=['valid', 'test'], default='test')
    parser.add_argument('--benchmark-uncached-users', type=int, default=0,
                        help='Additional user requests with block-wise GNN/VAE encoding; can be slow. 0 disables.')
    parser.add_argument('--benchmark-scales', default=None,
                        help='Comma-separated fractions in (0,1], e.g. .25,.5,.75,1. Fresh subprocess per compact induced graph; frozen weights and text inputs.')
    parser.add_argument('--benchmark-bundle', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--profile-training', action='store_true',
                        help='Record pretraining, epoch compute and validation encoder phases in training_profile.json.')
    parser.add_argument('--profile-rss-interval', type=float, default=.01,
                        help='Main-process RSS sampling interval in seconds (default 0.01).')
    add_dataset_arguments(parser)
    args = parser.parse_args()
    args.cli_overrides = sorted({parser._option_string_actions[token.split('=', 1)[0]].dest
                                 for token in sys.argv[1:]
                                 if token.split('=', 1)[0] in parser._option_string_actions})
    if (args.benchmark_only or args.benchmark_scales or args.benchmark_bundle) and not args.eval_only:
        parser.error('Deployment benchmarking requires --eval-only and a trained checkpoint.')
    if args.benchmark_bundle and args.benchmark_scales:
        parser.error('--benchmark-bundle cannot be combined with --benchmark-scales.')
    if args.profile_training and args.eval_only:
        parser.error('--profile-training is a training-mode option.')
    for name in ('benchmark_users', 'benchmark_repeats', 'benchmark_topn'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive.')
    if min(args.benchmark_warmup, args.benchmark_candidates, args.benchmark_uncached_users) < 0:
        parser.error('Benchmark warmup, candidate count and uncached user count cannot be negative.')
    if not np.isfinite(args.profile_rss_interval) or args.profile_rss_interval <= 0:
        parser.error('--profile-rss-interval must be finite and positive.')
    if args.benchmark_scales:
        try:
            fractions = [float(value) for value in args.benchmark_scales.split(',')]
            if not fractions or any(not np.isfinite(x) or not 0 < x <= 1 for x in fractions):
                raise ValueError
        except ValueError:
            parser.error('--benchmark-scales must list fractions in (0,1].')
    if args.checkpoint and not args.eval_only:
        parser.error('--checkpoint requires --eval-only.')
    if args.tuning_run and (args.eval_only or args.eval_on_test):
        parser.error('--tuning-run requires training mode and --eval_on_test 0.')
    if args.monitor_metric is not None:
        parts = args.monitor_metric.split('@')
        if len(parts) != 2 or parts[0] not in ('recall', 'ndcg') or not parts[1].isdigit() or int(parts[1]) < 1:
            parser.error('--monitor-metric must be recall@K or ndcg@K with positive K.')
    try:
        args.neighbor_sizes = [int(value) for value in args.neighbor_sizes.split(',')]
    except ValueError:
        parser.error('--neighbor-sizes must be comma-separated positive integers.')
    if not args.neighbor_sizes or min(args.neighbor_sizes) < 1:
        parser.error('All neighbor fanouts must be positive; full-neighbor (-1) sampling is disabled.')
    if args.pretrain_epochs is None:
        args.pretrain_epochs = args.epochs
    for name in ('epochs', 'batch_size', 'emb_dim', 'num_clusters', 'early_stopping',
                 'edge_chunk_size', 'lr_step_size', 'max_subgraph_nodes', 'max_subgraph_edges',
                 'eval_graph_batch_size', 'eval_user_batch_size',
                 'eval_item_batch_size', 'eval_text_batch_size', 'eval_pair_batch_size'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive.')
    if args.graph_layers < 0 or args.pretrain_epochs < 0 or args.cor_sample_size < 0:
        parser.error('Layer count, pretraining epochs and correlation sample size cannot be negative.')
    if not 0 <= args.dropout < 1 or args.user_temperature <= 0 or args.item_temperature <= 0:
        parser.error('Dropout must be in [0,1); temperatures must be positive.')
    if args.num_workers < 0 or args.grad_clip < 0:
        parser.error('Worker count and gradient clip cannot be negative.')
    if args.pretrain_batch_size is not None and args.pretrain_batch_size < 1:
        parser.error('--pretrain-batch-size must be positive.')
    if not np.isfinite(args.positive_threshold):
        parser.error('--positive-threshold must be finite.')
    return args


if __name__ == '__main__':
    run(parse_args())
