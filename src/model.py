from typing import List
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from layers import *
import pytorch_lightning as pl
from grakel.kernels import WeisfeilerLehman, WeisfeilerLehmanOptimalAssignment, Propagation, GraphletSampling, RandomWalkLabeled, PyramidMatch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import AUROC


def H(x): return -torch.sum(x * (x + 1e-12).log(), -1)
def JSD(x): return H(x.mean(-2)) - H(x).mean(-1)


def one_hot_embedding(labels, n_labels):
    eye = torch.eye(n_labels)
    return eye[labels]


def max_comp(E, d):
    E = list(E)

    if len(E) == 0:
        return E, d

    graph = csr_matrix((np.ones(len(E)), zip(*E)), [np.max(E) + 1] * 2)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    (unique, counts) = np.unique(labels, return_counts=True)
    max_elms = np.argwhere(labels == unique[np.argmax(counts)])

    max_ed_list = [e for e in E if (e[0] in max_elms) and (e[1] in max_elms)]

    dnew = dict([((int(k), d[k])) for k in max_elms.flatten()])

    return max_ed_list, dnew


class FeaturesKernel(nn.Module):
    def __init__(self, mask_count, mask_nodes_count, num_features, hops=3, device='cpu', k_type='v0', temp=False):
        super(FeaturesKernel, self).__init__()
        self.device = device
        self.hops = hops
        self.mask = nn.Parameter(torch.rand(mask_count, mask_nodes_count, num_features).float(), requires_grad=True)
        self.B = nn.Parameter(torch.rand(1, mask_count, mask_nodes_count).float(), requires_grad=True)
        self.k_type = k_type

        self.mask.data = torch.nn.functional.normalize(self.mask.data, dim=-1)
        if temp:
            self.temp = nn.Parameter(5 * torch.ones(1, 1, mask_count, 1).float(), requires_grad=True)
        else:
            self.temp = nn.Parameter(torch.tensor(1.).float(), requires_grad=False)

    def forward(self, x, edge_index, egonets_idx=None, packed_idx=None):
        if packed_idx is None:
            if egonets_idx is None:
                egonets_idx = [get_egonets(x, edge_index, i, self.hops)[0] for i in torch.arange(x.shape[0])]

            packed_idx = pad_sequence(egonets_idx, True, -1) + 1

        x_ = torch.cat([torch.zeros_like(x[:1, :]), x], 0)  # dummy feature
        egonet_features = x_[packed_idx, :]

        # normalize input features
        egonet_features = torch.nn.functional.normalize(egonet_features.to(torch.float32), dim=-1)
        res = None

        egonet_mask = (packed_idx > 0).float()[..., None, None]
        mask = torch.nn.functional.normalize(self.mask, dim=-1)
        inner = torch.tensordot(egonet_features, mask, dims=([2], [2])) * self.temp.abs()
        res = 1 - 0.5 * ((inner.softmax(-1) * egonet_mask).sum(1) - self.B).abs().sum(-1)
        return res


class GKernel(nn.Module):
    def __init__(self, nodes, labels, filters=8, max_cc=None, hops=3, kernels='wl', normalize=True, store_fit=False):
        super(GKernel, self).__init__()
        self.hops = hops

        A = torch.from_numpy(np.random.rand(filters, nodes, nodes)).float()
        A = ((A + A.transpose(-2, -1)) > 1).float()
        A = torch.stack([a - torch.diag(torch.diag(a)) for a in A], 0)
        self.P = nn.Parameter(A, requires_grad=False)

        self.X = nn.Parameter(torch.stack([one_hot_embedding(torch.randint(labels, (nodes,)), labels)
                              for fi in range(filters)], 0), requires_grad=False)
        self.Xp = nn.Parameter(torch.zeros((filters, nodes, labels)).float(), requires_grad=True)

        self.Padd = nn.Parameter(torch.randn(filters, nodes, nodes) * 0)
        self.Prem = nn.Parameter(torch.randn(filters, nodes, nodes) * 0)
        self.Padd.data = self.Padd.data + self.Padd.data.transpose(-2, -1)
        self.Prem.data = self.Prem.data + self.Prem.data.transpose(-2, -1)

        self.filters = filters
        self.store = [None] * filters

        self.gks = []
        for kernel in kernels.split('+'):
            if kernel == 'wl':
                self.gks.append(lambda x: WeisfeilerLehman(n_iter=3, normalize=normalize))
            if kernel == 'wloa':
                self.gks.append(lambda x: WeisfeilerLehmanOptimalAssignment(n_iter=3, normalize=normalize))
            if kernel == 'prop':
                self.gks.append(lambda x: Propagation(normalize=normalize))
            if kernel == 'rw':
                self.gks.append(lambda x: RandomWalkLabeled(normalize=normalize))
            if kernel == 'gl':
                self.gks.append(lambda x: GraphletSampling(normalize=normalize))
            if kernel == 'py':
                self.gks.append(lambda x: PyramidMatch(normalize=normalize))

        self.store_fit = store_fit
        self.stored = False

    def forward(self, x, edge_index, not_used=None, fixedges=None, node_indexes=[]):
        convs = [GKernelConv.apply(x, edge_index, self.P, self.Padd, self.Prem, self.X, self.Xp,
                                   self.hops, self.training, gk(None), self.stored, node_indexes) for gk in self.gks]
        return torch.cat(convs, -1)


def get_egonets(x, edge_index, i, hops=3):
    ego_nodes, ego_edges, _, _ = torch_geometric.utils.k_hop_subgraph(
        [i], num_hops=hops, edge_index=edge_index, relabel_nodes=True)
    return ego_nodes, ego_edges


def compute_egonets(x, edge_index, i, hops=3, num_nodes=None):
    fn, fe, _, _ = torch_geometric.utils.k_hop_subgraph([i], num_hops=hops, edge_index=edge_index, num_nodes=num_nodes)
    node_map = torch.arange(fn.max() + 1)
    node_map[fn] = torch.arange(fn.shape[0])
    ego_edges = node_map[fe]
    ego_nodes = x[fn, :]
    return ego_nodes, ego_edges


class GKernelConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edge_index, P, Padd, Prem, X, Xp, hops, training, gk, stored, node_indexes):
        # graph similarity here
        filters = P.shape[0]
        convs = []
        if not stored:
            egonets = [compute_egonets(x, edge_index, i, hops, x.shape[0]) for i in torch.arange(x.shape[0])]
            def G1(i): return [set([(e[0], e[1]) for e in egonets[i][1].t().numpy()]),
                               dict(zip(range(egonets[i][0].shape[0]), egonets[i][0].argmax(-1).numpy()))]
            Gs1 = [G1(i) for i in range(x.shape[0])]

            conv = GKernelConv.eval_kernel(x, Gs1, P, X, gk, False)
        else:
            conv = GKernelConv.eval_kernel(None, None, P, X, gk, True)[node_indexes, :]
            Gs1 = None

        ctx.save_for_backward(x, edge_index, P, Padd, Prem, X, Xp, conv)
        ctx.stored = stored
        ctx.node_indexes = node_indexes
        ctx.Gs1 = Gs1
        ctx.P = P
        ctx.X = X
        ctx.gk = gk

        return conv.float()

    @staticmethod
    def backward(ctx, grad_output):
        x, edge_index, P, Padd, Prem, X, Xp, conv = ctx.saved_tensors
        P = ctx.P

        # perform random edit for each non zero filter gradient:
        grad_padd = 0
        grad_prem = 0
        grad_xp = 0

        kindexes = torch.nonzero(torch.norm(grad_output, dim=0))[:, 0]
        Pnew = P.clone()
        Xnew = X.clone()

        for i in range(3):  # if the gradient of the edit w.r.t. the loss is 0 we try another edit operation
            for fi in kindexes:
                edit_graph = torch.rand((1,)).item() < 0.5 or X.shape[-1] == 1
                Pnew, Xnew = GKernelConv.random_edit(fi, Pnew, Padd, Prem, Xnew, Xp, edit_graph)
            if not ctx.stored:
                convnew = GKernelConv.eval_kernel(x, ctx.Gs1, Pnew, Xnew, ctx.gk, True)
            else:
                convnew = GKernelConv.eval_kernel(None, None, Pnew, Xnew, ctx.gk, True)[ctx.node_indexes, :]
            grad_fi = conv - convnew

            proj = (grad_fi * grad_output).sum(0)[:, None, None]
            kindexes = kindexes[proj[kindexes, 0, 0] == 0]
            if len(kindexes) == 0:
                break

        grad_padd += proj * (P - Pnew)
        grad_prem += proj * (Pnew - P)
        grad_xp += proj * (X - Xnew)

        th = 0
        ctx.P.data = (proj >= th) * Pnew + (proj < th) * P
        ctx.X.data = (proj >= th) * Xnew + (proj < th) * X

        return None, None, None, grad_padd * ((Padd).sigmoid() * (1 - (Padd).sigmoid())),\
            grad_prem * ((Prem).sigmoid() * (1 - (Prem).sigmoid())), None,\
            grad_xp * (Xp.sigmoid() * (1 - Xp.sigmoid())), None, None, None, None, None

    @staticmethod
    def eval_kernel(x, Gs1, P, X, gk, stored=False):
        filters = P.shape[0]
        nodes = P.shape[1]

        Gs2 = [max_comp(set([(e[0], e[1]) for e in torch_geometric.utils.dense_to_sparse(P[fi])[0].t().numpy()]),
                        dict(zip(range(nodes), X[fi].argmax(-1).flatten().detach().numpy()))) for fi in range(filters)]

        if not stored:
            gk.fit(Gs1)
            sim = gk.transform(Gs2)
            sim = np.nan_to_num(sim)
        else:
            sim = gk.transform(Gs2)
            sim = np.nan_to_num(sim)

        return torch.from_numpy(sim.T)

    @staticmethod
    def random_edit(i, Pin, Padd, Prem, X, Xp, edit_graph, n_edits=1, temp=0.1):
        filters = Pin.shape[0]

        P = Pin.clone()
        X = X.clone()
        if edit_graph:  # edit graph
            Pmat = P[i] * (Prem[i] * temp).sigmoid().data + (1 - P[i]) * \
                (Padd[i] * temp).sigmoid().data + 1e-8  # sample edits
            Pmat = Pmat * (1 - np.eye(Pmat.shape[-1]))
            Pmat = Pmat / Pmat.sum()
            inds = np.random.choice(Pmat.shape[0]**2, size=(n_edits,), replace=False, p=Pmat.flatten().numpy(),)
            inds = torch.from_numpy(np.stack(np.unravel_index(inds, Pmat.shape), 0)).to(Pmat.device)

            inds = torch.cat([inds, inds[[1, 0], :]], -1)  # symmetric edit
            P[i].data[inds[0], inds[1]] = 1 - P[i].data[inds[0], inds[1]]

            if (P[i].sum() == 0):  # avoid fully disconnected graphs
                P = Pin.clone()
        else:  # edit labels
            PX = (Xp[i] * temp).softmax(-1).data
            pi = 1 - PX.max(-1)[0]
            pi = pi / pi.sum(-1, keepdims=True)

            lab_ind = np.random.choice(X[i].shape[0], (n_edits,), p=pi.numpy())
            lab_val = [np.random.choice(PX.shape[1], size=(1,), replace=False, p=PX[j, :].numpy(),) for j in lab_ind]

            X[i].data[lab_ind, :] = 0
            X[i].data[lab_ind, lab_val] = 1

        return P, X


class Model(pl.LightningModule):
    def __init__(self, hparams, device='cpu'):
        super(Model, self).__init__()

        if (type(hparams) is dict):
            import argparse
            args = argparse.ArgumentParser()
            for k in hparams.keys():
                args.add_argument('--' + k, default=hparams[k])
            hparams = args.parse_args([])

        if not 'activation' in hparams.__dict__:
            hparams.activation = 'relu'

        if not 'temp' in hparams.__dict__:
            hparams.temp = False

        self.save_hyperparameters(hparams)

        in_features = hparams.in_features
        hidden = hparams.hidden
        num_classes = hparams.num_classes
        labels = hparams.labels
        mask_nodes_count = hparams.nodes
        filters = hparams.filters
        self.mode = hparams.mode
        self.loss_func = hparams.loss_func if hparams.loss_func is not None else torch.nn.CrossEntropyLoss()
        k_type = hparams.k_type

        self.features_conv_layers = nn.ModuleList()
        self.struct_conv_layers = nn.ModuleList()
        self.vq_layers = nn.ModuleList()

        self.auroc_test_eval = None
        if hparams.auroc_test:
            self.auroc_test_eval = AUROC(num_classes=num_classes)
        self.auroc_val_eval = None
        if hparams.auroc_val:
            self.auroc_val_eval = AUROC(num_classes=num_classes)

        self.features_conv_layers.append(FeaturesKernel(
            hidden, mask_nodes_count, in_features, device=device, k_type=k_type))

        n_kernels = len(hparams.kernel.split('+'))
        for i in range(1, hparams.layers):
            if self.mode == 'structure':
                self.struct_conv_layers.append(GKernel(hparams.nodes, hidden, hidden, max_cc=self.hparams.max_cc,
                                                       hops=hparams.hops, kernels=hparams.kernel))
            elif self.mode == 'feature':
                self.features_conv_layers.append(FeaturesKernel(
                    hidden, mask_nodes_count, hidden, device=device, k_type=k_type))
            elif self.mode == 'combined':
                self.features_conv_layers.append(FeaturesKernel(
                    hidden, mask_nodes_count, 2 * hidden, device=device, k_type=k_type))

            commitment_cost = 0.25
            decay = 0.99
            self.vq_layers.append(VectorQuantizerEMA(hidden * n_kernels, hidden, commitment_cost, decay=decay))

        activation = nn.ReLU
        if hparams.activation == 'sigmoid':
            activation = nn.Sigmoid

        self.fc = nn.Sequential(nn.Linear(hidden * n_kernels * hparams.layers, hidden), activation(),
                                nn.Linear(hidden, hidden), activation(), nn.Linear(hidden, num_classes))

        self.eye = torch.eye(hidden)
        self.lin = nn.Linear(hidden, hidden)

        self.automatic_optimization = False

        def _regularizers(x):
            if hparams.jsd_weight > 0:
                jsdiv = hparams.jsd_weight * JSD(x.softmax(-1))
                return -jsdiv
            else:
                return x.sum().detach() * 0

        self.regularizers = _regularizers
        self.mask = nn.Parameter(torch.ones(hidden).float(), requires_grad=False)

    def one_hot_embedding(self, labels):
        self.eye = self.eye.to(labels.device)
        return self.eye[labels]

    def forward(self, data):
        if 'nidx' not in data.__dict__:
            data.nidx = None
        batch = data.batch
        loss = data.x.sum().detach() * 0
        responses = []
        _, responses = self.forward_feature(data, responses, True)
        x = torch.cat(responses, -1)

        pooling_op = None
        if self.hparams.pooling == 'add':
            pooling_op = global_add_pool
        if self.hparams.pooling == 'max':
            pooling_op = global_max_pool
        if self.hparams.pooling == 'mean':
            pooling_op = global_mean_pool

        return self.fc(pooling_op(x, batch)), responses, loss

    def forward_feature(self, data, responses: List = None, calc_mask: bool = False):
        x = data.x
        for layer, vq in zip(self.features_conv_layers, [None] + list(self.vq_layers)):
            x, responses = self.single_layer_forward(layer, x, data.edge_index, data.egonets_idx, responses, calc_mask)
        return x, responses

    def single_layer_forward(self, layer, input_data, edge_index, egonets_idx, responses: List = None, calc_mask: bool = False):
        x = layer(input_data, edge_index, egonets_idx)
        if self.hparams.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.hparams.dropout, training=self.training)
        if calc_mask and self.mask is not None:
            x = x * self.mask[None, :].repeat(1, x.shape[-1] // self.mask.shape[-1])
        if responses is not None:
            responses.append(x)
        return x, responses

    def configure_optimizers(self):
        graph_params = set(self.struct_conv_layers.parameters())
        cla_params = set(self.parameters()) - graph_params
        optimizer = torch.optim.Adam([{'params': list(graph_params), 'lr': self.hparams.lr_graph},
                                      {'params': list(cla_params), 'lr': self.hparams.lr}])

        return optimizer

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()

        optimizer.zero_grad()
        output, responses, _ = self(data)
        loss_ce = self.loss_func(output, data.y.squeeze(-1))

        loss_jsd = torch.stack([self.regularizers(x) for x in responses]).mean()

        loss = loss_ce + loss_jsd
        loss.backward()
        optimizer.step()

        acc = 100 * torch.mean((output.argmax(-1) == data.y).float()).detach().cpu()
        self.log('acc', acc, on_step=False, on_epoch=True)
        self.log('loss', loss.item(), on_step=False, on_epoch=True)
        self.log('loss_jsd', loss_jsd.item(), on_step=False, on_epoch=True)
        self.log('loss_ce', loss_ce.item(), on_step=False, on_epoch=True)

    def validation_step(self, data, batch_idx):
        with torch.no_grad():
            output, x1, _ = self(data)
            loss = self.loss_func(output, data.y.squeeze(-1))
            acc = 100 * torch.mean((output.argmax(-1) == data.y).float()).detach().cpu()
            self.log('val_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True)
            if self.auroc_val_eval is not None:
                self.auroc_val_eval.update(output, data.y)

    def validation_epoch_end(self, outputs):
        if self.auroc_val_eval is not None:
            roc = self.auroc_val_eval.compute()
            self.log('val_roc', roc, on_step=False, on_epoch=True)
            self.auroc_val_eval.reset()

    def test_step(self, data, batch_idx):
        with torch.no_grad():
            output, x1, _ = self(data)
            loss = self.loss_func(output, data.y.squeeze(-1))

            acc = 100 * torch.mean((output.argmax(-1) == data.y).float()).detach().cpu()
            self.log('test_loss', loss.item(), on_step=False, on_epoch=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True)
            if self.auroc_test_eval is not None:
                self.auroc_test_eval.update(output, data.y)

    def test_epoch_end(self, outputs):
        if self.auroc_test_eval is not None:
            roc = self.auroc_test_eval.compute()
            self.log('test_roc', roc, on_step=False, on_epoch=True)
            self.auroc_test_eval.reset()


def clean_graph(data):  # remove isolated nodes
    connected_nodes = set([i.item() for i in data.edge_index.flatten()])
    isolated_nodes = [i for i in range(data.x.shape[0]) if i not in connected_nodes]
    mask = torch.ones((data.x.shape[0],)).bool()
    mask[isolated_nodes] = False
    mapping = -torch.ones((data.x.shape[0],)).long()
    mapping[mask] = torch.arange(mask.sum())

    data.edge_index = mapping[data.edge_index]
    if 'nidx' in data.__dict__:
        data.nidx = data.nidx[mask]
    data.x = data.x[mask, :].int()
    data.batch = data.batch[mask]
    return data
