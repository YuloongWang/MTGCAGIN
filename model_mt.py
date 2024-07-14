import math
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GINConv, ChebConv, GCNConv
from einops import rearrange, repeat
from util import MyNNConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from sklearn.covariance import GraphicalLasso


class ModuleTimestamping(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, t, sampling_endpoints):
        return self.rnn(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]]


class ModuleTimestamping_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, t, sampling_endpoints):
        return self.lstm(t[:sampling_endpoints[-1]])[0][[p-1 for p in sampling_endpoints]]



class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]]))  # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0

        # self.cheb_conv = ChebConv(in_channels=input_dim, out_channels=hidden_dim, K=3)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())

        # self.n1 = nn.Sequential(nn.Linear(116, 8, bias=False), nn.ReLU(), nn.Linear(8, hidden_dim * hidden_dim))
        # self.conv1 = MyNNConv(input_dim, hidden_dim, self.n1, normalize=False)

        # self.scores = ChebConv(in_channels=input_dim, out_channels=1, K=3)

    def forward(self, v, a, n=116):
        v_aggregate = torch.sparse.mm(a, v)
        v_aggregate += self.epsilon * v  # assumes that the adjacency matrix includes self-loop

        # v_aggregate = self.cheb_conv(v, a._indices())  # a._values()

        # a_size = a.size(0)
        # b_size = n
        # target_matrix = torch.zeros(a_size, b_size)
        # for i in range(a_size // b_size):
        #     diagonal_matrix = torch.eye(b_size)
        #     target_matrix[i * b_size: (i + 1) * b_size] = diagonal_matrix
        # v_aggregate = self.conv1(v, a._indices(), a._values(), target_matrix.to('cuda:0'))

        v_combine = self.mlp(v_aggregate)
        return v_combine


class SABP(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5, Conv=GCNConv, non_linearity=torch.tanh):
        super(SABP, self).__init__()
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        score = self.score_layer(x, edge_index, edge_attr).squeeze()
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm, torch.sigmoid(score)


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32)


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU())
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=2):
        # assumes shape [... x node x ... x feature]
        # x[~perm.bool()] = 0
        x_readout = x.mean(node_axis)  # [26, 8, 128]
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1]))  # [208, 128]
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1], -1)  # [26, 8, 116]
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis, len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx)  # [26, 8, 116]
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1, 0, 2)


class ModuleGARO(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, upscale=1.0, **kwargs):
        super().__init__()
        self.embed_query = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.embed_key = nn.Linear(hidden_dim, round(upscale*hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_q = self.embed_query(x.mean(node_axis, keepdims=True))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q, rearrange(x_k, 't b n c -> t b c n'))/np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)


class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend)  # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix


class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_atlas, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.ModuleList()
        self.out = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.ModuleList()
        for i in range(num_atlas):
            attn_list = nn.ModuleList()
            for j in range(num_atlas):
                attn_list.append(nn.MultiheadAttention(input_dim, num_heads))
            self.multihead_attn.append(attn_list)
            self.out.append(nn.Linear(input_dim, input_dim))
            self.layer_norm1.append(nn.LayerNorm(input_dim))
            self.layer_norm2.append(nn.LayerNorm(input_dim))
            self.mlp.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim)))

    def forward(self, x):  # [26, 8, 128] for list
        x_attend_all = []
        attn_matrix_all = []
        for num, (OUT, MA, LN1, MLP, LN2) in enumerate(zip(self.out, self.multihead_attn, self.layer_norm1, self.mlp, self.layer_norm2)):
            x_attend = OUT(sum(MA[i](x[num], x[i], x[i])[0] for i in range(len(x)))/len(x))  # query, key, value  (sum or linear ?)
            attn_matrix = sum(MA[i](x[num], x[i], x[i])[1] for i in range(len(x)))/len(x)
            x_attend = x_attend + x[num]  # skip connection
            x_attend = self.dropout(x_attend)
            x_attend2 = LN1(x_attend)
            x_attend2 = MLP(x_attend2)
            x_attend = x_attend + self.dropout(x_attend2)  # skip connection
            x_attend = LN2(x_attend)

            x_attend_all.append(x_attend)
            attn_matrix_all.append(attn_matrix)
        return torch.cat(x_attend_all, dim=-1), torch.cat(attn_matrix_all, dim=-1)


class ModelMTGCAIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5, cls_token='sum', readout='sero', garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0)
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='garo': readout_module = ModuleGARO
        elif readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None

        self.num_classes = num_classes
        self.sparsity = sparsity
        self.num_layers = num_layers

        # define modules
        self.percentile = Percentile()
        self.timestamp_encoder = nn.ModuleList()
        self.initial_linear = nn.ModuleList()
        # self.weights = []
        for i in range(len(input_dim)):
            self.timestamp_encoder.append(ModuleTimestamping_LSTM(input_dim[i], hidden_dim, hidden_dim))
            self.initial_linear.append(nn.Linear(input_dim[i]+hidden_dim, hidden_dim))
            # self.initial_linear.append(nn.Linear(hidden_dim+hidden_dim, hidden_dim))
            # self.weights.append(nn.Parameter(torch.randn(self.num_layers)))
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        # self.feature_linear = nn.Linear(hidden_dim*len(input_dim), hidden_dim)
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # self.linear_weights = torch.nn.Parameter(torch.randn(self.num_layers))
        self.pool = nn.ModuleList()
        # self.cls_token2 = nn.ModuleList()

        for i in range(num_layers):
            gnn_list = nn.ModuleList()
            nested_list = nn.ModuleList()
            pool_list = nn.ModuleList()
            # trans_list = nn.ModuleList()
            for j in range(len(input_dim)):  # 三层嵌套列表
                gnn_list.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim, epsilon=False))
                nested_list.append(readout_module(hidden_dim=hidden_dim, input_dim=math.ceil(input_dim[j]), dropout=0.3))
                # pool_list.append(TopKPooling(hidden_dim, ratio=0.5, multiplier=1, nonlinearity=torch.sigmoid))
                pool_list.append(SABP(hidden_dim, ratio=0.5))
                # trans_list.append(ModuleTransformer(hidden_dim*len(input_dim), 2*hidden_dim*len(input_dim), num_heads=num_heads, dropout=0.2))
            self.gnn_layers.append(gnn_list)
            self.pool.append(pool_list)
            self.readout_modules.append(nested_list)
            # self.cls_token2.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=26))
            # self.transformer_modules.append(trans_list)
            self.transformer_modules.append(MultimodalTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, num_atlas=len(input_dim), dropout=0.3))
            # self.feature_linear.append(nn.Linear(hidden_dim * len(input_dim), hidden_dim))
            self.linear_layers.append(nn.Linear(hidden_dim*len(input_dim), num_classes))

    def _collate_adjacency(self, a, sparsity, sparse=True):
        i_list = []
        v_list = []

        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > self.percentile(_a, 100-sparsity))
                # thresholded_a = self._glasso(_a)
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                # _v = _a[_i[:, 0], _i[:, 1]]
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))

    @staticmethod
    def _glasso(a):
        i = torch.tensor(GraphicalLasso(alpha=0.5).fit(a.detach().cpu().numpy()).precision_, dtype=torch.float32)
        return i.to(torch.bool).to(a.device)

    @staticmethod
    def _get_position_encoding(v, seq_len=116, d_model=128):
        batch, dynamic = v.shape[:2]
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        position_encoding = pe.expand(batch, dynamic, -1, -1)
        return position_encoding

    def forward(self, v, a, t, sampling_endpoints, roi):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        logit = 0.0
        reg_ortho = 0.0
        attention = {'node-attention': [], 'time-attention': []}
        latent_list = []
        minibatch_size, num_timepoints = a[roi[0]].shape[:2]
        num_nodes = {}
        time_encoding = {}
        h = {}
        # layer_out = {}  # Res
        # weight = {}  # weight Res
        batch = {}
        # linear_weight = F.sigmoid(self.linear_weights)
        # linear_weight = F.softmax(torch.arange(0, 0.5 * self.num_layers, 0.5), dim=0)
        score_all = {}
        pool_weight = 0
        for i, r in enumerate(roi):
            # weight[r] = F.sigmoid(self.weights[i])  # weight Res
            num_nodes[r] = a[r].shape[2]
            batch[r] = torch.arange(a[r].size(0) * a[r].size(1)).unsqueeze(1).repeat(1, a[r].size(2)).view(-1).to(a[r].device)
            time_encoding[r] = self.timestamp_encoder[i](t[r], sampling_endpoints)
            time_encoding[r] = repeat(time_encoding[r], 'b t c -> t b n c', n=num_nodes[r])
            # v[r] = self._get_position_encoding(v[r]).to(a[r].device)
            h[r] = torch.cat([a[r], time_encoding[r]], dim=3)  # v is one-hot encoder, size: [8, 26, 116, 116]
            h[r] = rearrange(h[r], 'b t n c -> (b t n) c')
            h[r] = self.initial_linear[i](h[r])  # h: torch.Size([9048, 128]) <- torch.Size([9048, 116+128])
            a[r] = self._collate_adjacency(a[r], self.sparsity)  # torch.Size([9048, 9048]) <- torch.Size([3, 26, 116, 116])
            # layer_out[r] = []  # Res
            score_all[r] = 0
        for layer, (G, P, R, T, L) in enumerate(zip(self.gnn_layers, self.pool, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h_readout_list = []
            node_attn_list = []
            # latent = []
            for i, r in enumerate(roi):
                # layer_out[r].append(h[r])  # Res
                h[r] = G[i](h[r], a[r])
                # h[r]:[24128(8*26*116), 128], a[r]._indices:[2, 839529], a[r]._values:[839529, 1], batch:[24128]
                x, indices, values, b, perm, score = P[i](h[r], a[r]._indices(), a[r]._values(), batch[r])
                # a[r] = torch.sparse_coo_tensor(indices=indices, values=values, size=a[r].size())
                # score_all[r] += score
                # pool_weight += P[i].weight
                pool_weight += P[i].score_layer.lin.weight
                # h[r] = h[r] + weight[r][layer] * layer_out[r][layer]  # weight Res
                # h[r] = h[r] + 0.7 * layer_out[r][layer]  # Res
                # h_bridge = rearrange(h[r]*score.unsqueeze(1), '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes[r])
                h_bridge = rearrange(h[r], '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes[r])
                h_score = rearrange(score, '(b t n) -> t b n', t=num_timepoints, b=minibatch_size, n=num_nodes[r])
                h_readout = (h_bridge * self.dropout(h_score.unsqueeze(-1))).mean(2)
                node_attn = h_score.permute(1, 0, 2)
                # h_readout, node_attn = R[i](h_bridge, h_score, node_axis=2)  # [26, 3, 128], [3, 26, 116] <- [26, 3, 116, 128]
                if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1, h_readout.shape[1], -1)])
                h_readout_list.append(h_readout)
                node_attn_list.append(node_attn)
                ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')
                matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0, 2, 1))
                reg_ortho += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(h_bridge.size(2), device=matrix_inner.device)).triu().norm(dim=(1, 2)).mean()
                # print(reg_ortho)
            h_attend, time_attn = T(h_readout_list)  # [26, 3, 128*n], [3, 26, 26*n] <- [26, 3, 128]*n
            latent = self.dropout(self.cls_token(h_attend))  # [3, 128] <- [26, 3, 128]
            # readout = torch.cat(h_readout_list, dim=2)  # 请自行填入合适的维度
            # readout = self.dropout(self.feature_linear[layer](torch.cat(h_readout_list, dim=2)))
            # readout = self.feature_linear[layer](self.dropout(torch.cat(h_readout_list, dim=2)))
            # h_attend, time_attn = T(readout)  # [26, 3, 128], [3, 26, 26] <- [26, 3, 128]

            # latent = self.cls_token(h_attend)  # [3, 128] <- [26, 3, 128]
            # latent = self.cls_token2[layer](h_attend.permute(1, 2, 0)).squeeze(2)  # [3, 128] <- [26, 3, 128]
            logit += self.dropout(L(latent))  # [3, 2] <- [3, 2(128)]
            # logit += self.dropout(L(latent) * linear_weight[layer])  # [3, 2] <- [3, 2(128)] * [w]

            attention['node-attention'].append(torch.cat(node_attn_list, dim=-1))  # 只append了最后一个的node_attn
            attention['time-attention'].append(time_attn)
            latent_list.append(latent)

        attention['node-attention'] = torch.stack(attention['node-attention'], dim=1).detach().cpu()
        attention['time-attention'] = torch.stack(attention['time-attention'], dim=1).detach().cpu()
        latent = torch.stack(latent_list, dim=1)

        return logit, attention, latent, reg_ortho, pool_weight, score_all


# Percentile class based on
# https://github.com/aliutkus/torchpercentile
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
