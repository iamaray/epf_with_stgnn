class STGNN_ATS(nn.Module):
    def __init__(
            self,
            ats_constr: ATSConstructor,
            forward_ats_constr: ATSConstructor,
            ots_model: nn.Module,
            ats_model: nn.Module,
            forward_model: nn.Module,
            cats=True,
            f_ats=True):
        super(STGNN_ATS, self).__init__()
        # torch.set_default_device('cuda')
        self.cats = cats
        self.f_ats = f_ats

        # ATS constructors
        self.ats_constr = ats_constr
        self.forward_ats_constr = forward_ats_constr

        self.ots_model = ots_model
        self.ats_model = ats_model
        self.forward_model = forward_model

        self.proj = None
        if self.f_ats:
            self.proj = nn.Conv1d(
                in_channels=(self.ots_model.pred_length * 3),
                out_channels=self.ots_model.pred_length,
                kernel_size=1,
                device='cuda')

        else:
            self.proj = nn.Conv1d(
                in_channels=(self.ots_model.pred_length * 2),
                out_channels=self.ots_model.pred_length,
                kernel_size=1,
                device='cuda')

    def forward(self, data):
        x = data.x

        batch_size, input_channels, num_variates, seq_len = x.shape
        ots = x[:, 0, :, :]

        ats = self.ats_constr(ots)

        ats = ats.view(x.shape[0], ats.shape[1] //
                       x.shape[2], x.shape[2], x.shape[3])
        ots_pred = None

        if self.cats:
            # pass x through OTS model
            ots_pred = self.ots_model(ots.unsqueeze(1))

            # pass ATS output into ATS model
            ats_pred = self.ats_model(ats)

            resid_in = torch.cat([ots_pred, ats_pred], dim=1)

            forward_ats = None
            if self.f_ats:
                x_stacked = x.view(
                    batch_size, input_channels * num_variates, seq_len)

                forward_ats = self.forward_ats_constr(x_stacked)
                forward_ats = forward_ats.view(
                    x.shape[0], forward_ats.shape[1] // x.shape[2], x.shape[2], x.shape[3])

                # pass F-ATS output into F-ATS model
                forward_pred = self.forward_model(forward_ats)
                # concatenate all three predictions and project
                resid_in = torch.cat([resid_in, forward_pred], dim=1)

            resid = self.proj(resid_in)
            # add to OTS prediction
            return ots_pred + resid, ats, forward_ats

        else:
            ots_pred = self.ots_model(x)

        return ots_pred, None, None


def construct_ats_stgnn(
        data,
        ats_constr_params,
        fwd_ats_constr_params,
        ots_model_params,
        ats_model_params,
        forward_model_params,
        cats=True,
        f_ats=True):

    batch_size, c_in, num_nodes, seq_len = data.x.shape
    _, pred_len, _ = data.y.shape

    ats_constr = ATSConstructor(
        seq_len=seq_len,
        c_in=num_nodes,
        conv_out=ats_constr_params['conv_out'],
        conv_kern=ats_constr_params['conv_kern'],
        non_overlap_conv_out=ats_constr_params['non_overlap_conv_out'],
        non_overlap_conv_kern=ats_constr_params['non_overlap_conv_kern'],
        independ_conv_out_mult=ats_constr_params['independ_conv_out_mult'],
        independ_conv_kern=ats_constr_params['independ_conv_kern'],
        lin_proj_out=ats_constr_params['lin_proj_out'],
        embed_out=ats_constr_params['embed_out'],
        embed_conv_chans=ats_constr_params['embed_conv_chans'],
        embed_kern_size=ats_constr_params['embed_kern_size'],
        feat_extractor_activation=ats_constr_params['feat_extractor_activation'],
        agg_activation=ats_constr_params['agg_activation'],
        atten_activation=ats_constr_params['atten_activation']
    )

    fwd_constr = ATSConstructor(
        seq_len=seq_len,
        c_in=num_nodes * c_in,
        conv_out=fwd_ats_constr_params['conv_out'],
        conv_kern=fwd_ats_constr_params['conv_kern'],
        non_overlap_conv_out=fwd_ats_constr_params['non_overlap_conv_out'],
        non_overlap_conv_kern=fwd_ats_constr_params['non_overlap_conv_kern'],
        independ_conv_out_mult=fwd_ats_constr_params['independ_conv_out_mult'],
        independ_conv_kern=fwd_ats_constr_params['independ_conv_kern'],
        lin_proj_out=fwd_ats_constr_params['lin_proj_out'],
        embed_out=fwd_ats_constr_params['embed_out'],
        embed_conv_chans=fwd_ats_constr_params['embed_conv_chans'],
        embed_kern_size=fwd_ats_constr_params['embed_kern_size'],
        feat_extractor_activation=fwd_ats_constr_params['feat_extractor_activation'],
        agg_activation=fwd_ats_constr_params['agg_activation'],
        atten_activation=fwd_ats_constr_params['atten_activation']
    )

    c_in_ots = c_in
    if cats:
        c_in_ots = 1

    ots_model = STGNN(
        K=ots_model_params['K'],
        beta=ots_model_params['beta'],
        c_in=c_in_ots,
        num_nodes=num_nodes,
        batch_size=batch_size,
        sequence_length=seq_len,
        residual_chans=ots_model_params['residual_chans'],
        conv_chans=ots_model_params['conv_chans'],
        gc_support_len=ots_model_params['gc_support_len'],
        gc_order=ots_model_params['gc_order'],
        gc_dropout=ots_model_params['gc_dropout'],
        skip_chans=ots_model_params['skip_chans'],
        end_chans=ots_model_params['end_chans'],
        num_layers=ots_model_params['num_layers'],
        dilation_multiplier=ots_model_params['dilation_multiplier'],
        pred_length=pred_len,
        dropout_factor=ots_model_params['dropout_factor'],
        use_graph_conv=ots_model_params['use_graph_conv'],
        use_temp_conv=ots_model_params['use_temp_conv'],
        use_diffusion=ots_model_params['use_diffusion'],
        adj_type=ots_model_params['adj_type']
    )

    ats_model = STGNN(
        K=ats_model_params['K'],
        beta=ats_model_params['beta'],
        c_in=ats_constr.num_ats // num_nodes,
        num_nodes=num_nodes,
        batch_size=batch_size,
        sequence_length=seq_len,
        residual_chans=ats_model_params['residual_chans'],
        conv_chans=ats_model_params['conv_chans'],
        gc_support_len=ats_model_params['gc_support_len'],
        gc_order=ats_model_params['gc_order'],
        gc_dropout=ats_model_params['gc_dropout'],
        skip_chans=ats_model_params['skip_chans'],
        end_chans=ats_model_params['end_chans'],
        num_layers=ats_model_params['num_layers'],
        dilation_multiplier=ats_model_params['dilation_multiplier'],
        pred_length=pred_len,
        dropout_factor=ats_model_params['dropout_factor'],
        use_graph_conv=ats_model_params['use_graph_conv'],
        use_temp_conv=ats_model_params['use_temp_conv'],
        use_diffusion=ats_model_params['use_diffusion'],
        adj_type=ats_model_params['adj_type']
    )

    forward_model = STGNN(
        K=forward_model_params['K'],
        beta=forward_model_params['beta'],
        c_in=fwd_constr.num_ats // num_nodes,
        num_nodes=num_nodes,
        batch_size=batch_size,
        sequence_length=seq_len,
        residual_chans=forward_model_params['residual_chans'],
        conv_chans=forward_model_params['conv_chans'],
        gc_support_len=forward_model_params['gc_support_len'],
        gc_order=forward_model_params['gc_order'],
        gc_dropout=forward_model_params['gc_dropout'],
        skip_chans=forward_model_params['skip_chans'],
        end_chans=forward_model_params['end_chans'],
        dilation_multiplier=forward_model_params['dilation_multiplier'],
        num_layers=forward_model_params['num_layers'],
        pred_length=pred_len,
        dropout_factor=forward_model_params['dropout_factor'],
        use_graph_conv=forward_model_params['use_graph_conv'],
        use_temp_conv=forward_model_params['use_temp_conv'],
        use_diffusion=forward_model_params['use_diffusion'],
        adj_type=forward_model_params['adj_type']
    )

    return STGNN_ATS(
        ats_constr=ats_constr,
        forward_ats_constr=fwd_constr,
        ots_model=ots_model,
        ats_model=ats_model,
        forward_model=forward_model,
        cats=cats,
        f_ats=f_ats
    )
