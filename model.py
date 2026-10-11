"""C2-IDN: bounded neighbor-sampled GNN training and ordered pair scoring.
"""
from typing import List
from contextlib import nullcontext
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as fn

def mlp_layers(layer_dims):
    mlp_modules = []
    for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layer = nn.Linear(d_in, d_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.normal_(layer.bias, std=0.01)
        mlp_modules.append(layer)
        del layer
    return nn.Sequential(*mlp_modules)

class MacridVAE(nn.Module):

    # Bound the same posterior log-variance for both sampling and KL loss.
    # This is a numerical safeguard that constrains the posterior family.
    LOGVAR_MIN = -10.0
    LOGVAR_MAX = 10.0

    def __init__(self, num_user: int, num_items: int, emb_size: int, dropout: float, num_prototypes: int, device, tau: float=1.0, anneal_cap: float=0.2, std: float=0.025):
        super(MacridVAE, self).__init__()
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('MacridVAE: tau must be finite and positive.')
        if not math.isfinite(std) or std < 0:
            raise ValueError('MacridVAE: std must be finite and nonnegative.')
        if not math.isfinite(anneal_cap) or anneal_cap < 0:
            raise ValueError('MacridVAE: anneal_cap must be finite and nonnegative.')
        self.num_user = num_user
        self.num_items = num_items
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_prototypes = num_prototypes
        self.tau = tau
        self.anneal_cap = anneal_cap
        self.std = std
        self.device = device
        self.encoder_layer_dims = [self.num_items] + [100] + [self.emb_size * 2]
        self.encoder = mlp_layers(self.encoder_layer_dims)
        self.node_embeddings = nn.Embedding(self.num_user, self.emb_size)
        nn.init.xavier_normal_(self.node_embeddings.weight)
        self.item_embeddings = nn.Embedding(self.num_items, self.emb_size)
        nn.init.xavier_normal_(self.item_embeddings.weight)
        self.prototype_embeddings = nn.Embedding(self.num_prototypes, self.emb_size)
        nn.init.xavier_normal_(self.prototype_embeddings.weight)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.linear = nn.Linear(self.emb_size * 2, self.emb_size)
        self.embed_dic = self.init_weight()

    @staticmethod
    def _require_finite(value, name):
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f'MacridVAE: {name} contains NaN/Inf; '
                'check input data, parameters, and the previous optimizer update.'
            )

    def _bounded_logvar(self, logvar):
        # Check BEFORE clamping: do not silently turn an existing Inf into 10.
        self._require_finite(logvar, 'raw logvar')
        return logvar.clamp(min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)

    def init_weight(self):
        all_weights = nn.ParameterDict()
        initializer = nn.init.xavier_normal_
        for k in range(self.num_prototypes):
            all_weights['person_weight1_%d' % k] = nn.Parameter(initializer(torch.empty(self.emb_size * 2, self.emb_size)))
        return all_weights

    def reparameterize(self, mu, logvar):
        logvar = self._bounded_logvar(logvar)
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + epsilon * std
        return mu

    def encode(self, input_rating, index, device):
        if input_rating.ndim != 2 or input_rating.shape != (index.numel(), self.num_items):
            raise ValueError('MacridVAE: expected TF-IDF shape [batch_size, vocabulary_size].')
        self._require_finite(input_rating, 'TF-IDF input')
        ID_feature1 = self.node_embeddings(index).to(device)
        prototypes = self.prototype_embeddings.weight
        items = self.item_embeddings.weight
        ID_feature = fn.normalize(ID_feature1, dim=1)
        cates_list = []
        for k in range(self.num_prototypes):
            prototype_k = prototypes[k].reshape(1, self.emb_size).expand(len(ID_feature), -1)
            interest = torch.cat((prototype_k, ID_feature), -1)
            interest = torch.matmul(interest, self.embed_dic['person_weight1_%d' % k])
            cates_logits = items.matmul(interest.transpose(0, 1)) / self.tau
            cates_logits = cates_logits.T
            cates_list.append(cates_logits)
        cates = torch.stack(cates_list, dim=0)
        self._require_finite(cates, 'prototype assignment logits (check tau)')
        cates = cates.softmax(dim=0)
        mu_list = []
        logvar_list = []
        z_list = []
        cates_k_list = []
        for k in range(self.num_prototypes):
            cates_k = cates[k, :, :].reshape(-1, self.num_items)
            cates_k = cates_k.to(device)
            cates_k_list.append(cates_k)
            x_k = input_rating * cates_k
            h = self.encoder(x_k)
            self._require_finite(h, f'encoder output for facet {k}')
            mu = h[:, :self.emb_size]
            mu = fn.normalize(mu, dim=1)
            logvar = self._bounded_logvar(-h[:, self.emb_size:])
            mu_list.append(mu)
            logvar_list.append(logvar)
            z = self.reparameterize(mu, logvar)
            z = fn.normalize(z, dim=1)
            z_list.append(z)
        return (z_list, mu_list, logvar_list, cates_k_list, items, prototypes, ID_feature)

    def decode(self, z_list, cates_k_list, items, need_prob_k=False):
        probs = None
        prob_k_list = []
        for k in range(self.num_prototypes):
            z_k = z_list[k]
            cates_k = cates_k_list[k]
            logits_k = z_k.matmul(items.transpose(0, 1)) / self.tau
            probs_k = logits_k
            probs_k = probs_k * cates_k
            if need_prob_k:
                prob_k_list.append(probs_k)
            probs = probs_k if probs is None else probs + probs_k
        return (probs, prob_k_list if need_prob_k else None)

    def forward(self, input_rating, index, device, need_prob_k=False, need_z=False):
        z_list, mu_list, logvar_list, cates_k_list, items, prototypes, ID_feature = self.encode(input_rating, index, device)
        probs, prob_k_list = self.decode(z_list, cates_k_list, items, need_prob_k)
        logits = probs
        outputs = [logits, mu_list, logvar_list, ID_feature]
        if need_prob_k:
            outputs.append(prob_k_list)
        if need_z:
            outputs.append(z_list)
        return tuple(outputs)

    def calculate_kl_loss(self, logvar):
        kl_loss = None
        for i in range(self.num_prototypes):
            lv = self._bounded_logvar(logvar[i])
            # expm1(lv) - lv equals exp(lv) - 1 - lv, with better
            # accuracy near zero. Keep the original KL reduction/weighting.
            kl_ = 0.5 * torch.mean(torch.sum(torch.expm1(lv) - lv, dim=1))
            kl_loss = kl_ if kl_loss is None else kl_loss + kl_
        self._require_finite(kl_loss, 'KL loss')
        return kl_loss

    @staticmethod
    def calculate_ce_loss(input_rating, logits, device):
        input_rating = input_rating.to(device)
        MacridVAE._require_finite(input_rating, 'reconstruction target')
        MacridVAE._require_finite(logits, 'decoder logits')
        ce_loss = -(fn.log_softmax(logits, 1) * input_rating).sum(dim=1).mean()
        MacridVAE._require_finite(ce_loss, 'reconstruction loss')
        return ce_loss

    def calculate_loss(self, input_rating, index, device, need_z=False):
        anneal = self.anneal_cap
        logits, mu, logvar, ID_feature, z_list = self.forward(input_rating, index, device, need_z=True)
        kl_loss = self.calculate_kl_loss(logvar)
        ce_loss = self.calculate_ce_loss(input_rating, logits, device)
        loss = ce_loss + kl_loss * anneal
        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f'MacridVAE: total loss is NaN/Inf; '
                f'CE={ce_loss.detach().item():.6g}, '
                f'KL={kl_loss.detach().item():.6g}, anneal_cap={anneal}'
            )
        return loss

    def get_param_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def get_params(self):
        return self.parameters()

class Cal_interest(nn.Module):

    def __init__(self, num_prototypes, dropout, embed_dim, device):
        super(Cal_interest, self).__init__()
        self.num_prototypes = num_prototypes
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.dropout_layer = nn.Dropout(self.dropout)
        self.device = device
        self.all_weights = self.init_weight()
        self.bn = nn.BatchNorm1d(self.embed_dim)

    def init_weight(self):
        all_weights = nn.ParameterDict()
        initializer = nn.init.xavier_normal_
        for k in range(self.num_prototypes):
            all_weights['weight1_%d' % k] = nn.Parameter(initializer(torch.empty(self.embed_dim * 2, self.embed_dim)))
        return all_weights

    def forward(self, user_z_list, item_z_list):
        interest_list = []
        for k in range(self.num_prototypes):
            user_tensor, item_tensor = (user_z_list[k], item_z_list[k])
            interest_tensor = torch.cat((user_tensor, item_tensor), -1)
            interest_tensor = torch.matmul(interest_tensor, self.all_weights['weight1_%d' % k])
            interest_tensor = self.dropout_layer(interest_tensor)
            interest_list.append(interest_tensor)
        interest_list = torch.stack(interest_list)
        return interest_list

    def concate(self, tensor1, tensor2):
        N = tensor1.size()[0]
        M = tensor2.size()[0]
        tensor1_repeated_in_chunks = tensor1.repeat_interleave(M, dim=0)
        tensor2_repeated_alternating = tensor2.repeat(N, 1)
        return (tensor1_repeated_in_chunks, tensor2_repeated_alternating)

class Alignment(nn.Module):

    def __init__(self, num_prototypes):
        super(Alignment, self).__init__()
        self.num_prototypes = num_prototypes

    def forward(self, cf_u_list, text_u_list, cf_v_list, text_v_list, cf_interest_list, text_interest_list):
        reg_term1 = self.regularization_term(cf_u_list, text_u_list)
        reg_term2 = self.regularization_term(cf_v_list, text_v_list)
        reg_term3 = self.regularization_term(cf_interest_list, text_interest_list)
        return reg_term1 + reg_term2 + reg_term3

    def regularization_term(self, cf_z_list, text_z_list):
        reg_term = self.mutual_information_reg(cf_z_list, text_z_list)
        return reg_term

    def mutual_information_reg(self, list_a: List[Tensor], list_b: List[Tensor]):
        assert len(list_a) == self.num_prototypes
        assert len(list_a) == len(list_b)
        if self.num_prototypes == 1:
            return list_a.sum() * 0.0
        mi_reg = None
        for i in range(self.num_prototypes):
            x_pos = list_a[i, :, :]
            y_pos = list_b[i, :, :]
            f_values = [(x_pos * list_b[j, :, :]).sum(dim=-1) for j in range(len(list_b)) if j != i]
            soft_plus_f_values = [fn.softplus(value).unsqueeze(dim=-1) for value in f_values]
            e_n = torch.cat(soft_plus_f_values, dim=-1).mean(dim=-1)
            e_p = -fn.softplus(-(x_pos * y_pos).sum(dim=-1))
            loss_i = (e_n - e_p).mean()
            mi_reg = loss_i if mi_reg is None else loss_i + mi_reg
        return mi_reg / self.num_prototypes

class Fusion(nn.Module):

    def __init__(self, num_prototypes, embed_dim, dropout, device):
        super(Fusion, self).__init__()
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.device = device
        self.all_weights = self.init_weight()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.key_linear1 = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.xavier_normal_(self.key_linear1.weight)
        nn.init.normal_(self.key_linear1.bias, std=0.01)
        self.query_linear1 = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.xavier_normal_(self.query_linear1.weight)
        nn.init.normal_(self.query_linear1.bias, std=0.01)

    def init_weight(self):
        all_weights = nn.ParameterDict()
        initializer = nn.init.xavier_normal_
        for k in range(self.num_prototypes):
            all_weights['key_linear_%d' % k] = nn.Parameter(initializer(torch.empty(self.embed_dim, self.embed_dim * 2)))
            all_weights['query_linear_%d' % k] = nn.Parameter(initializer(torch.empty(self.embed_dim, self.embed_dim * 2)))
        return all_weights

    def forward(self, cf_interest_list, text_interest_list, id_interest):
        interest_list = []
        for k in range(self.num_prototypes):
            cf_interest_tensor = cf_interest_list[k, :, :]
            text_interest_tensor = text_interest_list[k, :, :]
            interest_tensor = torch.cat((cf_interest_tensor, text_interest_tensor), dim=1)
            key = torch.matmul(cf_interest_tensor, self.all_weights['key_linear_%d' % k])
            query = torch.matmul(text_interest_tensor, self.all_weights['query_linear_%d' % k])
            cf_interest_tensor_pro = key * interest_tensor
            cf_interest_tensor_score = fn.relu(cf_interest_tensor_pro)
            text_interest_tensor_pro = query * interest_tensor
            text_interest_tensor_score = fn.relu(text_interest_tensor_pro)
            cf_interest_tensor_final = interest_tensor * cf_interest_tensor_score
            text_interest_tensor_final = interest_tensor * text_interest_tensor_score
            interest_tensor_final = (cf_interest_tensor_final + text_interest_tensor_final) / 2
            interest_list.append(interest_tensor_final)
        interest_list_tensor = torch.stack(interest_list, dim=0).permute(1, 0, 2)
        key1 = self.key_linear1(interest_list_tensor)
        key2 = self.query_linear1(interest_list_tensor)
        id_interest = id_interest.unsqueeze(1)
        id_attention = torch.bmm(id_interest, key1.permute(0, 2, 1))
        id_attention = fn.softmax(id_attention, dim=-1)
        final_interest = torch.bmm(id_attention, key2)
        return final_interest

class single_DGNN(nn.Module):
    """Interest-specific sparse propagation; never allocate a dense N x N graph.

Symmetric normalization stabilizes the learned edge weights. Set
--graph-normalization none to reproduce the original unnormalized aggregation.
Layer zero participates in the layer mean, as in the uploaded implementation.
"""
    def __init__(self, layers, num_prototypes, emb_size, dropout, device=None,
                 normalization='symmetric', edge_chunk_size=65536):
        super().__init__()
        self.layers = layers
        self.num_prototypes = num_prototypes
        self.dropout = nn.Dropout(dropout)
        self.normalization = normalization
        self.edge_chunk_size = edge_chunk_size

    def forward(self, user_embeddings, item_embeddings, edge_index):
        n_users, n_items = user_embeddings.shape[1], item_embeddings.shape[1]
        row, col = edge_index
        user_layers, item_layers = [user_embeddings], [item_embeddings]
        for _ in range(self.layers):
            next_users, next_items = [], []
            for k in range(self.num_prototypes):
                users, items = user_embeddings[k], item_embeddings[k]
                chunks = []
                for start in range(0, row.numel(), self.edge_chunk_size):
                    r = row[start:start + self.edge_chunk_size]
                    c = col[start:start + self.edge_chunk_size]
                    chunks.append(fn.relu((users[r] * items[c]).sum(dim=-1)))
                weights = torch.cat(chunks) if chunks else users.new_empty(0)
                if self.normalization == 'symmetric':
                    # The +1 accounts for the self-loop at each node.
                    du = users.new_ones(n_users).index_add(0, row, weights)
                    di = items.new_ones(n_items).index_add(0, col, weights)
                    edge_weights = weights * du[row].rsqrt() * di[col].rsqrt()
                    user_self, item_self = users / du[:, None], items / di[:, None]
                else:
                    edge_weights = weights
                    user_self, item_self = users, items
                if row.numel():
                    graph = torch.sparse_coo_tensor(
                        edge_index, edge_weights, (n_users, n_items),
                        device=users.device).coalesce()
                    user_messages = torch.sparse.mm(graph, items)
                    item_messages = torch.sparse.mm(graph.t(), users)
                else:
                    # Isolated seeds may leave the other bipartite side empty.
                    user_messages, item_messages = torch.zeros_like(users), torch.zeros_like(items)
                next_users.append(self.dropout(user_self + user_messages))
                next_items.append(self.dropout(item_self + item_messages))
            user_embeddings, item_embeddings = torch.stack(next_users), torch.stack(next_items)
            user_layers.append(user_embeddings)
            item_layers.append(item_embeddings)
        return torch.stack(user_layers).mean(0), torch.stack(item_layers).mean(0)


def text_rows(matrix, ids, device):
    """Gather only requested TF-IDF rows from torch/numpy/scipy containers."""
    if isinstance(matrix, torch.Tensor):
        if matrix.layout != torch.strided:
            matrix = matrix.to_sparse_coo().coalesce()
        rows = matrix.index_select(0, ids.to(matrix.device))
        if rows.layout != torch.strided:
            rows = rows.to_dense()
        return rows.to(device=device, dtype=torch.float32)
    rows = matrix[ids.detach().cpu().numpy()]
    if sp.issparse(rows):
        rows = rows.toarray()
    return torch.as_tensor(np.asarray(rows), device=device, dtype=torch.float32)


class NeighborSampler:
    """CPU-only bipartite neighbor sampling with strict per-batch limits.

Only sampled traversal edges are included (not a potentially huge induced
subgraph). Seeds are never discarded; fanouts run from seeds outward. This
is a sampled subgraph approximation, not an unbiased full-graph estimator.
The same node is expanded once, when first reached. If a cap is reached,
additional neighbors/edges are skipped; last_stats records the truncation.
"""
    def __init__(self, train_matrix, fanouts, max_nodes=20000, max_edges=100000):
        graph = sp.csr_matrix(train_matrix, dtype=np.float32, copy=True)
        if not np.isfinite(graph.data).all():
            raise ValueError('Training graph contains non-finite values.')
        graph.sum_duplicates()
        graph.data = (graph.data > 0).astype(np.float32)
        graph.eliminate_zeros()
        graph.sort_indices()
        reverse = graph.tocsc()
        # Retain sparse indices only; neither adjacency is a GPU tensor.
        self.user_ptr, self.user_items = graph.indptr, graph.indices
        self.item_ptr, self.item_users = reverse.indptr, reverse.indices
        self.num_users, self.num_items = graph.shape
        self.fanouts = tuple(int(f) for f in fanouts)
        self.max_nodes, self.max_edges = int(max_nodes), int(max_edges)
        if any(f < 1 for f in self.fanouts) or max_nodes < 1 or max_edges < 1:
            raise ValueError('Neighbor fanouts and subgraph limits must be positive.')
        self.last_stats = {}

    @staticmethod
    def _positions(degree, fanout, rng):
        """Floyd sampling: O(fanout) memory even for extremely high degree."""
        count = min(degree, fanout)
        if count == degree:
            return np.arange(degree, dtype=np.int64)
        chosen, result = set(), []
        for upper in range(degree - count, degree):
            candidate = int(rng.integers(upper + 1))
            selected = upper if candidate in chosen else candidate
            chosen.add(selected)
            result.append(selected)
        return np.asarray(result, dtype=np.int64)

    def sample(self, seed_users, seed_items, rng):
        seed_users = np.unique(np.asarray(seed_users, dtype=np.int64))
        seed_items = np.unique(np.asarray(seed_items, dtype=np.int64))
        if ((seed_users < 0) | (seed_users >= self.num_users)).any():
            raise ValueError('User seed ID out of range.')
        if ((seed_items < 0) | (seed_items >= self.num_items)).any():
            raise ValueError('Item seed ID out of range.')
        if len(seed_users) + len(seed_items) > self.max_nodes:
            raise ValueError('Batch seed count exceeds --max-subgraph-nodes. '
                             'Reduce --batch_size / --eval-graph-batch-size, or increase that limit.')
        users, items = set(seed_users.tolist()), set(seed_items.tolist())
        frontier = [(0, int(u)) for u in seed_users] + [(1, int(i)) for i in seed_items]
        edges, node_cap, edge_cap = set(), False, False
        for fanout in self.fanouts:
            following = []
            # Randomize expansion order to avoid favoring one node type at caps.
            for pos in rng.permutation(len(frontier)):
                kind, node = frontier[int(pos)]
                ptr, idx = (self.user_ptr, self.user_items) if kind == 0 else (self.item_ptr, self.item_users)
                start, end = int(ptr[node]), int(ptr[node + 1])
                selected = self._positions(end - start, fanout, rng)
                for offset in selected:
                    neighbor = int(idx[start + int(offset)])
                    edge = (node, neighbor) if kind == 0 else (neighbor, node)
                    if edge in edges:
                        continue
                    if len(edges) >= self.max_edges:
                        edge_cap = True
                        break
                    target = items if kind == 0 else users
                    if neighbor not in target:
                        if len(users) + len(items) >= self.max_nodes:
                            node_cap = True
                            continue
                        target.add(neighbor)
                        following.append((1 - kind, neighbor))
                    edges.add(edge)
                if edge_cap:
                    break
            if edge_cap or not following:
                break
            frontier = following
        user_ids = np.asarray(sorted(users), dtype=np.int64)
        item_ids = np.asarray(sorted(items), dtype=np.int64)
        if edges:
            global_edges = np.asarray(sorted(edges), dtype=np.int64).T
            local_edges = np.stack((np.searchsorted(user_ids, global_edges[0]),
                                    np.searchsorted(item_ids, global_edges[1])))
        else:
            local_edges = np.empty((2, 0), dtype=np.int64)
        self.last_stats = dict(sampled_users=len(user_ids), sampled_items=len(item_ids),
                               sampled_edges=len(edges), node_cap_hit=int(node_cap),
                               edge_cap_hit=int(edge_cap))
        return user_ids, item_ids, local_edges


class twoD_MIR(nn.Module):
    def __init__(self, args, graph_layers, num_items, num_users, num_words,
                 emb_size, dropout, num_prototypes, device, user_network,
                 item_network, lambda_user=1., lambda_item=.1,
                 lambda_align=.1, lambda_cor=.1, lambda_reg=.1, training=False):
        super().__init__()
        self.args = args
        self.num_users, self.num_items = num_users, num_items
        self.num_words, self.emb_size = num_words, emb_size
        self.num_prototypes = num_prototypes
        self.lambda_user, self.lambda_item = lambda_user, lambda_item
        self.lambda_align, self.lambda_cor, self.lambda_reg = lambda_align, lambda_cor, lambda_reg
        self.cor_sample_size = getattr(args, 'cor_sample_size', 256)
        self.user_network = user_network
        self.item_network = item_network
        self.linear = nn.Linear(emb_size, emb_size * num_prototypes)
        self.single_DGNN = single_DGNN(
            graph_layers, num_prototypes, emb_size, dropout,
            normalization=getattr(args, 'graph_normalization', 'symmetric'),
            edge_chunk_size=getattr(args, 'edge_chunk_size', 65536))
        self.cf_Cal_interest = Cal_interest(num_prototypes, dropout, emb_size, device)
        self.text_Cal_interest = Cal_interest(num_prototypes, dropout, emb_size, device)
        self.id_linear = nn.Linear(2 * emb_size, emb_size)
        self.user_encoder = mlp_layers([emb_size, emb_size * 4, emb_size])
        self.item_encoder = mlp_layers([emb_size, emb_size * 4, emb_size])
        self.Alignment = Alignment(num_prototypes)
        self.Fusion = Fusion(num_prototypes, emb_size, dropout, device)
        self.linear1 = nn.Linear(emb_size, 1)
        fanouts = getattr(args, 'neighbor_sizes', [10, 5])
        if isinstance(fanouts, str):
            fanouts = [int(value) for value in fanouts.split(',')]
        if graph_layers and not fanouts:
            raise ValueError('At least one neighbor fanout is required.')
        self.fanouts = tuple(fanouts[min(hop, len(fanouts) - 1)] for hop in range(graph_layers))
        self.max_subgraph_nodes = getattr(args, 'max_subgraph_nodes', 20000)
        self.max_subgraph_edges = getattr(args, 'max_subgraph_edges', 100000)
        self.sampler = None
        self.sample_rng = np.random.default_rng(getattr(args, 'random_seed', 42))
        self.last_graph_stats = {}
        self.to(device)
        self.train(training)

    @property
    def device(self):
        return self.linear.weight.device

    def set_train_graph(self, train_matrix):
        """Keep positive training CSR/CSC indices on CPU for local sampling only."""
        if train_matrix.shape != (self.num_users, self.num_items):
            raise ValueError('Training graph shape must be [num_users, num_items].')
        self.sampler = NeighborSampler(train_matrix, self.fanouts,
                                       self.max_subgraph_nodes, self.max_subgraph_edges)

    def encode_sampled_graph(self, seed_users, seed_items, rng=None):
        """Look up/project/propagate ONLY sampled nodes, then return seed rows.

No call normalizes or projects the full user/item embedding table. The full
ID parameter tables still exist as trainable model parameters on the device.
"""
        if self.sampler is None:
            raise RuntimeError('Call set_train_graph(train_csr_matrix) first.')
        seed_users_np = seed_users.detach().cpu().numpy()
        seed_items_np = seed_items.detach().cpu().numpy()
        user_ids, item_ids, edges = self.sampler.sample(
            seed_users_np, seed_items_np, self.sample_rng if rng is None else rng)
        user_tensor = torch.as_tensor(user_ids, device=self.device, dtype=torch.long)
        item_tensor = torch.as_tensor(item_ids, device=self.device, dtype=torch.long)
        users = fn.normalize(self.user_network.node_embeddings(user_tensor), dim=-1)
        items = fn.normalize(self.item_network.node_embeddings(item_tensor), dim=-1)
        users = self.linear(users).reshape(len(user_ids), self.num_prototypes, self.emb_size).permute(1, 0, 2)
        items = self.linear(items).reshape(len(item_ids), self.num_prototypes, self.emb_size).permute(1, 0, 2)
        edge_index = torch.as_tensor(edges, device=self.device, dtype=torch.long)
        users, items = self.single_DGNN(users, items, edge_index)
        user_positions = torch.as_tensor(np.searchsorted(user_ids, seed_users_np),
                                         device=self.device, dtype=torch.long)
        item_positions = torch.as_tensor(np.searchsorted(item_ids, seed_items_np),
                                         device=self.device, dtype=torch.long)
        self.last_graph_stats = dict(self.sampler.last_stats)
        return users[:, user_positions], items[:, item_positions]

    def _encode_text(self, network, texts, ids, encoder):
        z, mu, logvar, cates, words, _, _ = network.encode(texts, ids, self.device)
        raw = fn.normalize(torch.stack(z), dim=-1)
        encoded = encoder(raw)
        if self.training:
            logits, _ = network.decode(z, cates, words)
            loss = network.calculate_ce_loss(texts, logits, self.device)
            loss = loss + network.anneal_cap * network.calculate_kl_loss(logvar)
        else:
            loss = raw.new_zeros(())
        return raw, encoded, loss

    def _score(self, user_gnn, item_gnn, user_text, item_text):
        """Inputs are [K,B,D]; B is the ordered list of interactions, not B x B."""
        cf = self.cf_Cal_interest(user_gnn, item_gnn)
        text = self.text_Cal_interest(user_text, item_text)
        # The attention context also uses propagated representations.
        context = self.id_linear(torch.cat((user_gnn.mean(0), item_gnn.mean(0)), dim=-1))
        fused = self.Fusion(cf, text, context)
        return self.linear1(fused).reshape(-1), cf, text

    def forward(self, user_ids, item_ids, user_tfidf, item_tfidf,
                return_details=False):
        """Score matching pairs, preserving input order and all repetitions."""
        user_ids = user_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        item_ids = item_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if user_ids.numel() != item_ids.numel() or user_ids.numel() == 0:
            raise ValueError('Provide equally sized, nonempty user/item ID vectors.')
        unique_users, user_inverse = torch.unique(user_ids, sorted=True, return_inverse=True)
        unique_items, item_inverse = torch.unique(item_ids, sorted=True, return_inverse=True)
        user_gnn, item_gnn = self.encode_sampled_graph(unique_users, unique_items)
        user_raw, user_text, user_loss = self._encode_text(
            self.user_network, text_rows(user_tfidf, unique_users, self.device),
            unique_users, self.user_encoder)
        item_raw, item_text, item_loss = self._encode_text(
            self.item_network, text_rows(item_tfidf, unique_items, self.device),
            unique_items, self.item_encoder)
        scores, cf, text = self._score(
            user_gnn[:, user_inverse], item_gnn[:, item_inverse],
            user_text[:, user_inverse], item_text[:, item_inverse])
        if not return_details:
            return scores
        return scores, dict(user_gnn=user_gnn, item_gnn=item_gnn,
                            user_raw=user_raw, item_raw=item_raw, cf=cf, text=text,
                            user_loss=user_loss, item_loss=item_loss,
                            users=unique_users, items=unique_items)

    def create_cor_loss(self, users, items):
        nodes = torch.cat((users, items), dim=1)
        if self.num_prototypes < 2 or nodes.shape[1] < 2:
            return nodes.sum() * 0.0
        # Exact all-node distance correlation can exhaust memory (quadratic).
        # 0 selects all batch nodes; otherwise use at most cor_sample_size nodes.
        if self.cor_sample_size and nodes.shape[1] > self.cor_sample_size:
            ids = torch.randperm(nodes.shape[1], device=nodes.device)[:self.cor_sample_size]
            nodes = nodes[:, ids]
        distances = []
        for x in nodes:
            r = x.square().sum(-1, keepdim=True)
            d = (r + r.t() - 2 * (x @ x.t())).clamp_min(0).add(1e-8).sqrt()
            distances.append(d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean())
        terms = []
        for i in range(self.num_prototypes):
            for j in range(i + 1, self.num_prototypes):
                a, b = distances[i], distances[j]
                cov = (a * b).mean().clamp_min(0).add(1e-8).sqrt()
                va = a.square().mean().add(1e-8).sqrt()
                vb = b.square().mean().add(1e-8).sqrt()
                terms.append(cov / ((va * vb).sqrt() + 1e-10))
        return torch.stack(terms).mean()

    def calculate_loss(self, user_ids, item_ids, labels, user_tfidf, item_tfidf,
                       negative_item_ids=None):
        """BPR uses each positive plus a sampled unseen item; BCE uses every label."""
        original_count = user_ids.numel()
        if negative_item_ids is not None:
            if negative_item_ids.shape != item_ids.shape:
                raise ValueError('BPR needs one negative item for each positive item.')
            users = torch.cat((user_ids, user_ids))
            items = torch.cat((item_ids, negative_item_ids))
        else:
            users, items = user_ids, item_ids
        scores, info = self.forward(users, items, user_tfidf, item_tfidf, return_details=True)
        if negative_item_ids is not None:
            recommendation = fn.softplus(scores[original_count:] - scores[:original_count]).mean()
        else:
            recommendation = fn.binary_cross_entropy_with_logits(
                scores, labels.to(device=self.device, dtype=scores.dtype).reshape(-1))
        zero = scores.new_zeros(())
        cor = self.create_cor_loss(info['user_gnn'], info['item_gnn']) if self.lambda_cor else zero
        align = self.Alignment(info['user_gnn'], info['user_raw'], info['item_gnn'],
                               info['item_raw'], info['cf'], info['text']) if self.lambda_align else zero
        # L2 constrains the underlying trainable ID embeddings, not graph outputs.
        reg = (self.user_network.node_embeddings(info['users']).square().sum(-1).mean()
               + self.item_network.node_embeddings(info['items']).square().sum(-1).mean()) / 2
        loss = (recommendation + self.lambda_cor * cor + self.lambda_align * align
                + self.lambda_user * (info['user_loss'] + info['item_loss'])
                + self.lambda_reg * reg)
        self.last_losses = {name: value.detach().item() for name, value in dict(
            total=loss, recommendation=recommendation, correlation=cor, alignment=align,
            user_vae=info['user_loss'], item_vae=info['item_loss'], regularization=reg).items()}
        self.last_losses.update(self.last_graph_stats)
        return loss

    @torch.no_grad()
    def build_eval_cache(self, user_tfidf, item_tfidf, cache_dir,
                         graph_batch_size=128, text_batch_size=128, seed=42, profiler=None):
        """Encode small sampled subgraphs and stream seed representations to disk.

Encoding activation memory depends on graph/text batch limits; global ID
parameter tables and optimizer state still consume device memory. The
cache is a set of CPU disk-backed memmaps [N,K,D], NOT GPU tensors. All nodes
are eventually visited as seeds in separate batches, with no full-graph GNN
pass. Fixed seed AND fixed graph_batch_size give reproducible sampling.
Changing graph_batch_size may change sampled contexts. Candidate score chunk
sizes do not affect representations. Discard this cache after weight updates.
Temporary disk requirement: 2*(num_users+num_items)*K*D*4 bytes.
"""
        if self.training:
            raise RuntimeError('Call model.eval() before building the evaluation cache.')
        if graph_batch_size < 1 or text_batch_size < 1:
            raise ValueError('Evaluation batch sizes must be positive.')
        from pathlib import Path
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        cache = {}
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        try:
            for kind, network, texts, count, encoder in (
                    ('user', self.user_network, user_tfidf, self.num_users, self.user_encoder),
                    ('item', self.item_network, item_tfidf, self.num_items, self.item_encoder)):
                for feature in ('graph', 'text'):
                    name = f'{kind}_{feature}'
                    cache[name] = np.memmap(directory / f'{name}.dat', mode='w+',
                                            dtype=np.float32,
                                            shape=(count, self.num_prototypes, self.emb_size))
                with profiler.phase(f'{kind}_gnn_encoding') if profiler is not None else nullcontext():
                    for start in range(0, count, graph_batch_size):
                        end = min(start + graph_batch_size, count)
                        ids = torch.arange(start, end, device=self.device)
                        ug, ig = self.encode_sampled_graph(
                            ids if kind == 'user' else empty,
                            ids if kind == 'item' else empty, rng=rng)
                        embeddings = ug if kind == 'user' else ig
                        cache[f'{kind}_graph'][start:end] = embeddings.permute(1, 0, 2).cpu().numpy()
                        if profiler is not None:
                            profiler.observe_graph(self.last_graph_stats)
                        del ug, ig, embeddings
                        if start == 0 or end == count or (start // graph_batch_size) % 100 == 0:
                            print(f'Eval sampled {kind} graph: {end}/{count}', flush=True)
                    cache[f'{kind}_graph'].flush()
                with profiler.phase(f'{kind}_vae_encoding') if profiler is not None else nullcontext():
                    for start in range(0, count, text_batch_size):
                        end = min(start + text_batch_size, count)
                        ids = torch.arange(start, end, device=self.device)
                        _, encoded, _ = self._encode_text(
                            network, text_rows(texts, ids, self.device), ids, encoder)
                        cache[f'{kind}_text'][start:end] = encoded.permute(1, 0, 2).cpu().numpy()
                        del encoded
                        if start == 0 or end == count or (start // text_batch_size) % 100 == 0:
                            print(f'Eval {kind} text: {end}/{count}', flush=True)
                    cache[f'{kind}_text'].flush()
            return cache
        except Exception:
            self.close_eval_cache(cache)
            raise

    @staticmethod
    def close_eval_cache(cache):
        """Release file mappings before the enclosing temp directory is removed."""
        for array in cache.values():
            array._mmap.close()
        cache.clear()

    @torch.no_grad()
    def predict(self, user_ids, item_ids, cache, pair_batch_size=4096):
        """Load only requested cached nodes and score a bounded candidate block."""
        if self.training:
            raise RuntimeError('Call model.eval() before predict().')
        users = user_ids.detach().cpu().numpy().reshape(-1)
        items = item_ids.detach().cpu().numpy().reshape(-1)
        features = []
        for name, ids in (('user_graph', users), ('item_graph', items),
                          ('user_text', users), ('item_text', items)):
            # Advanced indexing copies only these rows from the disk-backed cache.
            block = np.asarray(cache[name][ids])
            features.append(torch.as_tensor(block, device=self.device).permute(1, 0, 2))
        ug, ig, ut, it = features
        result = ug.new_empty(len(users) * len(items))
        for start in range(0, result.numel(), pair_batch_size):
            positions = torch.arange(start, min(start + pair_batch_size, result.numel()), device=self.device)
            u, i = positions // len(items), positions % len(items)
            result[start:start + positions.numel()] = self._score(ug[:, u], ig[:, i], ut[:, u], it[:, i])[0]
        return result.reshape(len(users), len(items))

    def get_param_names(self):
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def get_params(self):
        return self.parameters()
