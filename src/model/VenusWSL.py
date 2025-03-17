import torch.nn as nn
import torch.nn.functional as F


class PredictorPLMConv(nn.Module):
    def __init__(self, plm_embed_dim, num_labels):
        super().__init__()
        self.conv1 = nn.Conv1d(plm_embed_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * 256, 256)
        self.fc2 = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(0.5)

    # x is plm embedding
    def forward(self, x, attention_mask):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PredictorPLM(nn.Module):
    def __init__(
        self,
        plm_embed_dim,
        attn_dim,
        num_labels):
        super().__init__()
        self.linear = nn.Linear(plm_embed_dim, attn_dim)
        self.activate = nn.ReLU()
        self.linear_2 = nn.Linear(attn_dim, attn_dim)
        self.classifier = Attention1dPoolingHead(attn_dim, num_labels, 0.5)

    # x is plm embedding
    def forward(self, x, attention_mask):
        x = self.activate(self.linear(x))
        x = self.linear_2(x)
        if len(x.shape) == 4:
            b, n_samples, n_residues, attn_dim = x.shape
            x = x.view(-1, n_residues, attn_dim)  # (batch_size * n_samples, n_residues, attn_dim)
            x = self.classifier(x, attention_mask).view(b, n_samples, -1)
        else:
            x = self.classifier(x, attention_mask)
        return x


class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.25
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(hidden_size)
        self.attention1d_projection = Attention1dPoolingProjection(hidden_size, num_labels, dropout)

    def forward(self, x, input_mask=None):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x


class Attention1dPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = MaskedConv1d(hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        return out


class Attention1dPoolingProjection(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.25) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.final(x)
        return x


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)
