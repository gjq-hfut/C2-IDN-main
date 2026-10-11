"""Data adapter for the neighbor-sampled model.py and main.py.
"""
from pathlib import Path
import pickle
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_DIR = '/path/to/music'
DEFAULT_DATA_PREFIX = 'Musical'
# Reuse one loaded bundle across get_data() and both VAE pretraining calls.
# Do not attach matrices to args: main.py serializes args to JSON/checkpoints.
_DATA_CACHE = None


def add_dataset_arguments(parser):
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--data-prefix', default=DEFAULT_DATA_PREFIX,
                        help='Filename prefix; independent of the dataset display name.')
    for option in ('train', 'valid', 'test', 'raw', 'user-tfidf', 'item-tfidf', 'word-vocab'):
        parser.add_argument(f'--{option}-path', default=None)
    for kind in ('user', 'item'):
        parser.add_argument(f'--{kind}-tfidf-id-space', choices=['auto', 'raw', 'local'], default='auto',
                            help='Dictionary key namespace. Matrix rows must already use local ID order.')
    parser.add_argument('--positive-threshold', type=float, default=0.,
                        help='binary=True maps rating > threshold to 1, otherwise 0. TF-IDF is unchanged.')
    parser.add_argument('--pretrain-batch-size', type=int, default=None,
                        help='VAE batch size; default min(batch_size, 128).')
    return parser


def _paths(args):
    directory = Path(getattr(args, 'data_dir', DEFAULT_DATA_DIR))
    prefix = getattr(args, 'data_prefix', DEFAULT_DATA_PREFIX)
    suffixes = dict(train='train_dataframe.pkl', valid='eval_dataframe.pkl',
                    test='test_dataframe.pkl', raw='raw_df.pkl',
                    user_tfidf='ifidf_by_user_new.pkl', item_tfidf='ifidf_by_item_new.pkl',
                    word_vocab='word2vec_index.pkl')
    # 'ifidf' is the spelling in the uploaded source filenames.
    return {name: Path(getattr(args, f'{name}_path', None) or directory / f'{prefix}_{suffix}')
            for name, suffix in suffixes.items()}


def _read_pickle(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def _integer_ids(values, name):
    values = pd.to_numeric(np.asarray(values), errors='raise')
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional ID column.')
    if not np.isfinite(values).all() or (values < 0).any() or (values >= 2**63).any():
        raise ValueError(f'{name} must contain finite, nonnegative int64 IDs.')
    result = values.astype(np.int64)
    if not np.equal(values, result).all():
        raise ValueError(f'{name} contains non-integer IDs.')
    return result


def _frame_ids(frame, name):
    for column in ('userID', 'itemID'):
        if column not in frame:
            raise ValueError(f'{name} is missing column {column!r}.')
    users = _integer_ids(frame['userID'], f'{name}.userID')
    items = _integer_ids(frame['itemID'], f'{name}.itemID')
    if users.size != items.size:
        raise ValueError(f'{name} has unequal user/item column lengths.')
    return users, items


def _read_rating_arrays(file_path, binary=False, positive_threshold=0.):
    frame = _read_pickle(file_path)
    users, items = _frame_ids(frame, str(file_path))
    if 'rating' not in frame:
        raise ValueError(f'{file_path} is missing the rating column.')
    ratings = np.asarray(frame['rating'], dtype=np.float32)
    if ratings.ndim != 1 or ratings.size != users.size or not np.isfinite(ratings).all():
        raise ValueError(f'{file_path}: invalid rating column.')
    if binary:
        ratings = (ratings > positive_threshold).astype(np.float32)
    return users, items, ratings


def read_rating_file(file_path, binary=False, positive_threshold=0.):
    """Legacy helper: return source IDs (not remapped) and float32 labels."""
    arrays = _read_rating_arrays(file_path, binary, positive_threshold)
    return tuple(torch.from_numpy(array) for array in arrays)


def _map_ids(ids, vocabulary, name):
    positions = np.searchsorted(vocabulary, ids)
    valid = positions < len(vocabulary)
    if valid.any():
        valid[valid] = vocabulary[positions[valid]] == ids[valid]
    if not valid.all():
        raise ValueError(f'{name}: IDs not present in raw_df ID vocabulary: {ids[~valid][:5].tolist()}')
    return positions.astype(np.int64, copy=False)


def _dictionary_keys(data, raw_ids, id_space, name):
    # Normalize integer-string dictionary keys while detecting collisions.
    normalized = _integer_ids(list(data.keys()), f'{name} dictionary keys')
    if len(np.unique(normalized)) != len(normalized):
        raise ValueError(f'{name}: duplicate IDs after converting dictionary keys to integers.')
    key_map = dict(zip(normalized.tolist(), data.keys()))
    candidates = {}
    if all(int(key) in key_map for key in raw_ids):
        candidates['raw'] = raw_ids
    local_ids = np.arange(len(raw_ids), dtype=np.int64)
    if all(int(key) in key_map for key in local_ids):
        candidates['local'] = local_ids
    if id_space == 'auto':
        if not candidates:
            raise ValueError(f'{name}: keys do not cover raw IDs or all local IDs; check TF-IDF preprocessing.')
        if len(candidates) == 2 and not np.array_equal(raw_ids, local_ids):
            raise ValueError(f'{name}: ambiguous raw/local dictionary IDs. Set the corresponding '
                             '--user-tfidf-id-space or --item-tfidf-id-space explicitly.')
        id_space = 'raw' if 'raw' in candidates else 'local'
    if id_space not in candidates:
        raise ValueError(f'{name}: dictionary does not cover all {id_space} IDs.')
    return [key_map[int(key)] for key in candidates[id_space]], id_space


def _to_csr(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
        if values.layout != torch.strided:
            coo = values.to_sparse_coo().coalesce()
            indices = coo.indices().numpy()
            return sp.csr_matrix((coo.values().numpy(), (indices[0], indices[1])), shape=coo.shape,
                                 dtype=np.float32)
        values = values.numpy()
    return sp.csr_matrix(values, dtype=np.float32)


def _load_tfidf(path, raw_ids, num_words, id_space, name):
    data = sp.load_npz(path) if Path(path).suffix == '.npz' else _read_pickle(path)
    resolved_space = 'local-matrix'
    if isinstance(data, dict):
        keys, resolved_space = _dictionary_keys(data, raw_ids, id_space, name)
        blocks, rows = [], []
        for key in keys:
            row = _to_csr(data[key])
            if row.shape == (num_words, 1) and num_words != 1:
                row = row.T.tocsr()
            if row.shape != (1, num_words):
                raise ValueError(f'{name} ID {key}: row shape {row.shape}, expected (1, {num_words}).')
            rows.append(row)
            if len(rows) == 512:
                blocks.append(sp.vstack(rows, format='csr'))
                rows = []
        if rows:
            blocks.append(sp.vstack(rows, format='csr'))
        matrix = sp.vstack(blocks, format='csr') if blocks else sp.csr_matrix((0, num_words), dtype=np.float32)
    else:
        if id_space == 'raw':
            raise ValueError(f'{name}: raw key alignment requires a dictionary; matrix rows must be in local ID order.')
        matrix = _to_csr(data)
    if matrix.shape != (len(raw_ids), num_words):
        raise ValueError(f'{name}: shape {matrix.shape}, expected ({len(raw_ids)}, {num_words}); '
                         'check entity row order and vocabulary/TF-IDF dimensions.')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f'{name}: TF-IDF must be finite and nonnegative.')
    print(f'{name}: shape={matrix.shape}, nnz={matrix.nnz}, ID space={resolved_space}', flush=True)
    return matrix


def _positive_csr(users, items, labels, shape, threshold):
    positive = labels > threshold
    graph = sp.csr_matrix((np.ones(int(positive.sum()), dtype=np.float32),
                           (users[positive], items[positive])), shape=shape)
    graph.sum_duplicates()
    graph.data.fill(1.)
    graph.sort_indices()
    return graph


def _load_bundle(args, binary):
    global _DATA_CACHE
    paths = _paths(args)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing data files; set --data-dir/--data-prefix or individual paths:\n' + '\n'.join(missing))
    threshold = float(getattr(args, 'positive_threshold', 0.))
    if not np.isfinite(threshold):
        raise ValueError('--positive-threshold must be finite.')
    spaces = tuple(getattr(args, f'{kind}_tfidf_id_space', 'auto') for kind in ('user', 'item'))
    key = (tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size) for p in paths.values()),
           bool(binary), threshold, spaces)
    if _DATA_CACHE is not None and _DATA_CACHE[0] == key:
        return _DATA_CACHE[1]
    _DATA_CACHE = None
    raw_users, raw_items = _frame_ids(_read_pickle(paths['raw']), 'raw_df')
    user_ids, item_ids = np.unique(raw_users), np.unique(raw_items)
    del raw_users, raw_items
    if not len(user_ids) or not len(item_ids):
        raise ValueError('raw_df must contain at least one user and one item.')
    word_vocab = _read_pickle(paths['word_vocab'])
    num_words = len(word_vocab)
    if num_words < 1:
        raise ValueError('Word vocabulary cannot be empty.')
    user_text = _load_tfidf(paths['user_tfidf'], user_ids, num_words, spaces[0], 'user TF-IDF')
    item_text = _load_tfidf(paths['item_tfidf'], item_ids, num_words, spaces[1], 'item TF-IDF')
    splits, matrices = {}, {}
    for split in ('train', 'valid', 'test'):
        users, items, labels = _read_rating_arrays(paths[split], binary, threshold)
        users = _map_ids(users, user_ids, f'{split} user')
        items = _map_ids(items, item_ids, f'{split} item')
        splits[split] = (users, items, labels)
        matrices[split] = _positive_csr(users, items, labels, (len(user_ids), len(item_ids)),
                                         0. if binary else threshold)
    if not len(splits['train'][0]):
        raise ValueError('Training split is empty.')
    stats = dict(num_users=len(user_ids), num_items=len(item_ids), num_words=num_words,
                 num_ratings=len(splits['train'][0]), word_vocab=word_vocab,
                 user_tfidf=user_text, item_tfidf=item_text,
                 user_raw_ids=user_ids, item_raw_ids=item_ids)
    for split in ('valid', 'test'):
        users, items, labels = splits[split]
        stats[f'num_{split}_users'] = len(np.unique(users))
        stats[f'num_{split}_items'] = len(np.unique(items))
        stats[f'num_{split}_ratings'] = len(labels)
    bundle = dict(splits=splits, matrices=matrices, stats=stats)
    _DATA_CACHE = (key, bundle)
    return bundle


class CSRDataset(Dataset):
    """Triples in original row order; repeated user/item/pair rows are retained."""
    def __init__(self, x, y, z):
        self.x = np.asarray(x, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        self.z = np.asarray(z, dtype=np.float32)
        if not (len(self.x) == len(self.y) == len(self.z)):
            raise ValueError('Unequal interaction array lengths.')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]


class SparseTextDataset(Dataset):
    """Only the requested row is densified; DataLoader assembles the batch."""
    def __init__(self, x):
        self.x = x.tocsr() if sp.issparse(x) else _to_csr(x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return idx, self.x.getrow(idx).toarray().reshape(-1)


# Preserve the original misspelled public names used by existing callers.
pretainuserDataset = SparseTextDataset
pretainitemDataset = SparseTextDataset


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _loader(dataset, args, batch_size, num_workers, shuffle, seed_offset=0):
    if batch_size < 1 or num_workers < 0:
        raise ValueError('Batch size must be positive and worker count nonnegative.')
    generator = torch.Generator()
    generator.manual_seed(int(args.random_seed) + seed_offset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=bool(shuffle) and len(dataset) > 0,
                      drop_last=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available() and getattr(args, 'gpu_id', 0) >= 0,
                      worker_init_fn=seed_worker, generator=generator)


def get_data(args, binary=True, num_workers=0, train_shuffle=True):
    bundle = _load_bundle(args, binary)
    stats = bundle['stats']
    offset = getattr(args, 'item_id_offset', None)
    offset = stats['num_users'] if offset is None else int(offset)
    if offset < 0:
        raise ValueError('--item-id-offset cannot be negative.')
    loaders = []
    for seed_offset, split in enumerate(('train', 'valid', 'test')):
        users, items, labels = bundle['splits'][split]
        dataset = CSRDataset(users, items + offset, labels)
        loaders.append(_loader(dataset, args, args.batch_size, num_workers,
                               train_shuffle if split == 'train' else False, seed_offset))
        print(f'{split}: samples={len(dataset)}, positive_edges={bundle["matrices"][split].nnz}', flush=True)
    print(f'Loader IDs: local users; local items + {offset}. CSR/TF-IDF use local IDs.', flush=True)
    return tuple(loaders), tuple(bundle['matrices'][s] for s in ('train', 'valid', 'test')), stats


def get_sparse_word_matrix_user(args):
    return _load_bundle(args, True)['stats']['user_tfidf']


def get_sparse_word_matrix_item(args):
    return _load_bundle(args, True)['stats']['item_tfidf']


def _pretrain_data(args, kind, binary, num_workers, train_shuffle):
    stats = _load_bundle(args, binary)['stats']
    batch_size = getattr(args, 'pretrain_batch_size', None)
    batch_size = min(args.batch_size, 128) if batch_size is None else int(batch_size)
    loader = _loader(SparseTextDataset(stats[f'{kind}_tfidf']), args, batch_size,
                     num_workers, train_shuffle, 10 if kind == 'user' else 20)
    # Entity IDs in these loaders are always LOCAL, including item IDs.
    return loader, stats


def get_pretain_user_data(args, binary=True, num_workers=0, train_shuffle=True):
    return _pretrain_data(args, 'user', binary, num_workers, train_shuffle)


def get_pretain_item_data(args, binary=True, num_workers=0, train_shuffle=True):
    return _pretrain_data(args, 'item', binary, num_workers, train_shuffle)
