class ConvConstr(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            kern_size: float,
            activation=nn.GELU()):
        super(ConvConstr, self).__init__()

        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kern_size,
            padding=(kern_size - 1) // 2)

    def forward(self, x):
        return self.activation(self.conv(x))


class NonOverlapConvConstr(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            seq_len: int,
            kern_size: float,
            activation=nn.GELU()):

        super(NonOverlapConvConstr, self).__init__()

        self.activation = activation

        pad = ((kern_size - (seq_len % kern_size)) % kern_size) // 2

        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kern_size,
            padding=pad,
            stride=kern_size)

    def forward(self, x):
        return self.activation(self.conv(x))


class IndependConvConstr(nn.Module):
    def __init__(
            self,
            c_in: int,
            out_mult: int,
            kern_size: float,
            activation=nn.GELU()):

        super(IndependConvConstr, self).__init__()

        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in * out_mult,
            kernel_size=kern_size,
            padding=((kern_size - 1) // 2),
            groups=c_in)

    def forward(self, x):
        return self.conv(x)


class LinProjConstr(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            activation=nn.GELU()):

        super(LinProjConstr, self).__init__()

        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            groups=1)
        # self.conv = nn.Linear(c_in, c_out)

    def forward(self, x):
        return self.activation(self.conv(x))


class IdConstr(nn.Module):
    def __init__(self, activation=nn.GELU()):
        super(IdConstr, self).__init__()

        self.activation = activation

    def forward(self, x):
        return self.activation(x)


class EmbeddingConstr(nn.Module):
    def __init__(self, num_variates, seq_len, embedding_dim, conv_chans=32, kernel_size=3, activation=nn.GELU()):
        super(EmbeddingConstr, self).__init__()
        self.activation = activation
        # Convolutional layer parameters
        self.conv_chans = conv_chans
        self.kernel_size = kernel_size

        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=num_variates, out_channels=conv_chans,
                               kernel_size=kernel_size, padding=kernel_size//2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(conv_chans)

        # Fully connected layer to produce embeddings for each variate independently
        self.fc = nn.Linear(in_features=conv_chans * seq_len,
                            out_features=embedding_dim * num_variates)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch_size, num_variates, seq_len]
        x_conv = self.conv1(x)
        x_bn = self.bn1(x_conv)
        x_act = self.relu(x_bn)

        batch_size, num_variates, seq_len = x.size()
        x_reshaped = x_act.view(self.conv_chans * seq_len, -1).transpose(0, 1)
        embeddings_reshaped = self.fc(x_reshaped)
        embeddings = embeddings_reshaped.view(batch_size, num_variates, -1)

        # Transpose to get the desired shape [batch_size, embedding_dim, num_variates]
        embeddings = embeddings.transpose(1, 2)

        return self.activation(embeddings)
