import torch
import torch.nn as nn
import torch.nn.functional as F


class FGN(nn.Module):
    def __init__(
            self,
            pre_length,
            embed_size,
            feature_size,
            seq_length,
            hidden_size,
            hard_thresholding_fraction=1,
            hidden_size_factor=1,
            sparsity_threshold=0.01):

        super(FGN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        if torch.cuda.is_available():
            self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) -
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) +
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) +
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) -
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) +
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) +
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) +
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
            torch.einsum('bli,ii->bli', o2_real, self.w3[0]) -
            torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) +
            self.b3[0]
        )

        o3_imag = F.relu(
            torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) +
            torch.einsum('bli,ii->bli', o2_real, self.w3[1]) +
            self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, data):
        x = data.x

        # x = x.permute(0, 2, 1).contiguous()
        x = x.contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x.to(self.embeddings_10.dtype), self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x, None, None


def construct_fgn(data, embed_size=32, hidden_size=32, hidden_size_factor=1, sparsity_threshold=0.01, hard_thresholding_fraction=1):
    """Initialize FGN model based on input data shape and hyperparameters.

    Args:
        data (torch.Tensor): Input tensor of shape (B, N, L) where:
            B is batch size
            N is number of nodes/features
            L is sequence length
        embed_size (int): Size of embedding dimension
        hidden_size (int): Size of hidden layers
        hidden_size_factor (int): Factor to multiply hidden size
        sparsity_threshold (float): Threshold for sparsity in Fourier domain
        hard_thresholding_fraction (float): Fraction for hard thresholding

    Returns:
        FGN: Initialized FGN model
    """
    B, N, L = data.x.shape
    _, _, P = data.y.shape

    model = FGN(
        pre_length=P,
        embed_size=embed_size,
        feature_size=N,  # Number of features/nodes
        seq_length=L,  # Input sequence length
        hidden_size=hidden_size,
        hidden_size_factor=hidden_size_factor,
        sparsity_threshold=sparsity_threshold,
        hard_thresholding_fraction=hard_thresholding_fraction
    )

    return model
