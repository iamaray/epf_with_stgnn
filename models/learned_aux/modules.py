import torch
import torch.nn as nn
import numpy as np
from pyinform import transfer_entropy
from .ats_constr import ConvConstr, NonOverlapConvConstr, IndependConvConstr, LinProjConstr, EmbeddingConstr


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        # batch_size: int,
        c_in: int,
        conv_out: int,
        conv_kern: int,
        non_overlap_conv_out: int,
        non_overlap_conv_kern: int,
        independ_conv_out_mult: int,
        independ_conv_kern: int,
        lin_proj_out: int,
        embed_out: int,
        embed_conv_chans: int,
        embed_kern_size: int,
            activation=nn.GELU()):
        super(FeatureExtractor, self).__init__()
        # elementary convolution
        self.conv = ConvConstr(
            c_in,
            conv_out,
            conv_kern,
            activation)

        # non-overlapping convolution
        self.non_overlap_conv = NonOverlapConvConstr(
            c_in,
            non_overlap_conv_out,
            seq_len, non_overlap_conv_kern,
            activation)

        # independent convolution
        self.independ_conv = IndependConvConstr(
            c_in,
            independ_conv_out_mult,
            independ_conv_kern,
            activation)

        # linear projection
        self.lin_proj = LinProjConstr(
            c_in,
            lin_proj_out,
            activation)

        # # identity
        # self.id = IdConstr(activation)
        # embedding
        self.emb = EmbeddingConstr(
            num_variates=c_in,
            seq_len=seq_len,
            embedding_dim=embed_out,
            conv_chans=embed_conv_chans,
            kernel_size=embed_kern_size
        )

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        x_non_overlap = self.non_overlap_conv(x)
        x_independ_conv = self.independ_conv(x)
        x_lin_proj = self.lin_proj(x)
        x_id = torch.clone(x)
        x_emb = self.emb(x)

        return [x_id, x_emb, x_conv, x_non_overlap, x_independ_conv, x_lin_proj]


class AttentionModule(nn.Module):
    def __init__(
            self,
            seq_len: int,
            num_variates: int,
            num_ats: int,
            agg_activation=nn.GELU(),
            atten_activation=nn.Sigmoid()):

        super(AttentionModule, self).__init__()

        self.agg_activation = agg_activation
        self.atten_activation = atten_activation

        self.agg_layer = nn.Linear(in_features=seq_len, out_features=1)

        self.atten_proj_inner = nn.Linear(
            in_features=num_variates, out_features=num_ats)

        self.atten_proj_outer = nn.Linear(
            in_features=num_ats, out_features=num_ats)

        self.batch_norm = nn.BatchNorm1d(num_features=num_ats)

    def forward(self, x: torch.Tensor):
        # x has shape [batch_size, input_variates, sequence_length]
        variates = torch.split(x, split_size_or_sections=1, dim=1)
        agg_vec = []
        for v in variates:
            agg_vec.append(self.agg_layer(v))

        agg_vec = torch.cat(agg_vec, dim=-1)
        proj_agg_vec = self.agg_activation(self.atten_proj_inner(agg_vec))

        atten_out = self.atten_activation(
            self.atten_proj_outer(proj_agg_vec)).squeeze(1)
        # atten_out has shape [batch_size, num_ats]
        return self.batch_norm(atten_out)


class TemporalCutoffModule(nn.Module):
    def __init__(
            self,
            num_input_variates: int,
            seq_len: int):
        super(TemporalCutoffModule, self).__init__()

        self.n = num_input_variates
        self.seq_len = seq_len
        self.cutoff_param_layers = [nn.Linear(in_features=self.seq_len, out_features=1)
                                    for i in range(self.n)]

        self.lags = torch.arange(self.seq_len) - self.seq_len

    def forward(self, x):

        variates = torch.split(x, split_size_or_sections=1, dim=1)

        cutoff_params = torch.cat([
            self.cutoff_param_layers[i](variates[i]) for i in range(self.n)], dim=1)

        look_ahead_params = (cutoff_params * self.lags) + 1

        indicator = look_ahead_params > 0

        with torch.no_grad():
            look_ahead_plus_ind = look_ahead_params + indicator

        ret = torch.cat(
            [variates[i] * ((look_ahead_plus_ind[:, i, :] - look_ahead_params[:, i, :]).unsqueeze(1)) for i in range(self.n)], dim=1)

        return ret


class batch_const_mul(nn.Module):
    def __init__(self):
        super(batch_const_mul, self).__init__()

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor):

        assert all(
            [len(x.shape) == 2,
             len(y.shape) == 3,
             x.shape[0] == y.shape[0],
             x.shape[1] == y.shape[1]])

        x_expanded = x.unsqueeze(-1)
        return x_expanded * y


class ATSConstructor(nn.Module):
    def __init__(
        self,
        seq_len: int,
        # batch_size: int,
        c_in: int,
        conv_out: int,
        conv_kern: int,
        non_overlap_conv_out: int,
        non_overlap_conv_kern: int,
        independ_conv_out_mult: int,
        independ_conv_kern: int,
        lin_proj_out: int,
        embed_out: int,
        embed_conv_chans: int,
        embed_kern_size: int,
        feat_extractor_activation: None = nn.GELU(),
        agg_activation: None = nn.GELU(),
        atten_activation: None = nn.Sigmoid()
    ):
        super(ATSConstructor, self).__init__()

        self.num_ats = conv_out + non_overlap_conv_out + \
            (c_in * independ_conv_out_mult) + lin_proj_out + embed_out + c_in
        self.feature_extr = FeatureExtractor(
            seq_len=seq_len,
            c_in=c_in,
            conv_out=conv_out,
            conv_kern=conv_kern,
            non_overlap_conv_out=non_overlap_conv_out,
            non_overlap_conv_kern=non_overlap_conv_kern,
            independ_conv_out_mult=independ_conv_out_mult,
            independ_conv_kern=independ_conv_kern,
            lin_proj_out=lin_proj_out,
            embed_out=embed_out,
            embed_conv_chans=embed_conv_chans,
            embed_kern_size=embed_kern_size,
            activation=feat_extractor_activation)

        self.atten = AttentionModule(
            seq_len=seq_len,
            num_variates=c_in,
            num_ats=self.num_ats,
            agg_activation=agg_activation,
            atten_activation=atten_activation)

        self.apply_atten = batch_const_mul()

        self.seq_len = seq_len

    def forward(self, x):
        # pass input signals to feature extractor
        ats = self.feature_extr(x)
        # print(torch.cat(ats, dim=1) != torch.cat(ats, dim=1))

        # pad the ats to all be the same length
        ats_padded = [nn.ConstantPad1d(
            (self.seq_len - a.shape[-1], 0), 0)(a) for a in ats]

        # ats[2] = nn.ConstantPad1d((self.seq_len - ats[2].shape[-1], 0), 0)(ats[2])
        # ats_padded = nn.utils.rnn.pad_sequence(
        #     [a.permute(2, 0, 1) for a in ats[1:]],
        #     batch_first=True).permute(0,2,3,1)
        # fix shape
        # ats_padded = torch.cat(
        #     [a.squeeze(0) for a in torch.split(ats_padded, dim=0, split_size_or_sections=1)],
        #     dim=1)
        ats = torch.cat(ats_padded, dim=1)
        # print(ats != ats)

        # compute attention weights
        atten_weights = self.atten(x)
        # print(atten_weights != atten_weights)
        # multiply attention to get channel sparse ATS
        sparse_ats = self.apply_atten(atten_weights, ats)

        return sparse_ats


class ContinuityLoss(nn.Module):
    def __init__(self, beta):
        super(ContinuityLoss, self).__init__()
        self.beta = beta
        self.prod = batch_const_mul()

    def forward(self, x):
        num_ats = x.shape[1]
        seq_len = x.shape[2]
        scale = self.beta / (num_ats * seq_len)

        variates = torch.split(x, split_size_or_sections=1, dim=1)
        stds = []

        for v in variates:
            stds.append(torch.std(v, dim=-1))

        stds = torch.cat(stds, dim=-1)
        stds = 1 / stds
        offset = x[:, :, 1:] - x[:, :, :seq_len-1]
        # compute the sum
        batch_loss = scale * torch.sum(  # sum along sequence length
            torch.sum(                    # sum along channel dimension
                torch.pow(                # square the inside
                    self.prod(stds, offset), 2), dim=1), dim=-1)

        return torch.mean(batch_loss)


class GraphContinuityLoss(nn.Module):
    def __init__(self, beta):
        super(GraphContinuityLoss, self).__init__()
        self.variate_loss = ContinuityLoss(beta)

    def forward(self, x):
        agg_loss = 0.0
        vars = [v.squeeze(2) for v in torch.split(
            x, split_size_or_sections=1, dim=2)]

        for v in vars:
            agg_loss += self.variate_loss(v)

        return (1 / x.shape[2]) * agg_loss


class TransferEntropyLoss(nn.Module):
    def __init__(self, beta, history_len):
        super(TransferEntropyLoss, self).__init__()
        self.history_len = history_len
        self.beta = beta

    def forward(self, x):
        scale = self.beta * (1 / (x.shape[1] * x.shape[2] * x.shape[3]))
        aggs = []
        variates = [v[:, :, 0, :]
                    for v in torch.split(x, split_size_or_sections=1, dim=2)]
        # print(variates[0].shape)
        for i, v in enumerate(variates):
            curr_agg = 0

            for j in range(1, v.shape[1]):
                v_curr = v[:, 0, :].cpu().detach().numpy()
                xs = v[:, j, :].cpu().detach().numpy()
                # ws = torch.cat(torch.split(torch.cat([v[:, 1:j, :], v[:, j+1:, :]], dim=1), split_size_or_sections=1, dim=1), dim=1).transpose(0,1).detach().numpy()
                # print(ws.shape)
                curr_agg += transfer_entropy(xs, v_curr, k=self.history_len)

            aggs.append(curr_agg)

        return scale * np.mean(aggs)
