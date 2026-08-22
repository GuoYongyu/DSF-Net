# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class PositionalEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            max_len: int = 5000
        ):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe: torch.Tensor = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(
            self,
            c_in: int,
            d_model: int
        ):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(
            self,
            c_in: int,
            d_model: int
        ):
        super(FixedEmbedding, self).__init__()

        w: torch.Tensor = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)


    def forward(self, x) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            embed_type: str = 'fixed',
            freq: str = ['h', 't', 's', 'm',
                         'a', 'w', 'd', 'b'][0]
        ):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = embed(minute_size, d_model)
        self.hour_embed = embed(hour_size, d_model)
        self.weekday_embed = embed(weekday_size, d_model)
        self.day_embed = embed(day_size, d_model)
        self.month_embed = embed(month_size, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            freq: str = ['h', 't', 's', 'm',
                         'a', 'w', 'd', 'b'][0]
        ):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(
            self,
            c_in: int,
            d_model: int,
            embed_type: str = 'fixed',
            freq: str = ['h', 't', 's', 'm',
                         'a', 'w', 'd', 'b'][0],
            dropout: float = 0.1
        ):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model,
            embed_type=embed_type,
            freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model,
            freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
    

class InceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_kernels: int = 6,
            init_weight: bool = True
        ):
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def fft_for_period(x: torch.Tensor, k: int = 2) -> tuple[torch.Tensor]:
    xf: torch.Tensor = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    top_list: torch.Tensor = torch.topk(frequency_list, k)[1]
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            top_k: int,
            num_kernels: int
        ):
        super(TimesBlock, self).__init__()

        self.seq_len = None  # dynamically set in forward
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(
                in_channels=d_model,
                out_channels=d_model * 2,
                num_kernels=num_kernels
            ),
            nn.GELU(),
            InceptionBlock(
                in_channels=d_model * 2,
                out_channels=d_model,
                num_kernels=num_kernels
            )
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        self.seq_len = seq_len

        period_list, period_weight = fft_for_period(x, k=self.top_k)

        res = list()
        for i in range(self.top_k):
            period = max(1, int(period_list[i]))

            if seq_len % period != 0:
                new_seq_len = ((seq_len // period) + 1) * period
                pad_len = new_seq_len - seq_len
                padding = torch.zeros([batch_size, pad_len, d_model], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                new_seq_len = seq_len
                out = x

            out = out.reshape(batch_size, new_seq_len // period, period, d_model)
            out = out.permute(0, 3, 1, 2).contiguous()
            out: torch.Tensor = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, d_model)
            out = out.reshape(batch_size, -1, d_model)
            res.append(out[:, :seq_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * period_weight, dim=-1)
        res = res + x
        return res
    

class TimesNet(nn.Module):
    def __init__(
            self,
            input_dim: int = 32,    # dim of features
            hidden_dim: int = 64,   # dim of hidden states
            num_layers: int = 2,    # num of TimeBlock layers
            output_num: int = 24,   # length of prediction
            output_dim: int = 128,  # dim of output
            top_k: int = 3,         # select top_k periods in fft
            num_kernels: int = 6,   # num of conv kernels in inception block
            dropout: float = 0.1
        ):
        super(TimesNet, self).__init__()
        self.name = "TimeFeatureNet(TimesNet)"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.dropout = dropout

        self.embedding = DataEmbedding(
            c_in=input_dim,
            d_model=hidden_dim,
            embed_type="timeF",  # time feature embedding
            freq="h",            # frequency: hour
            dropout=dropout
        )
        self.model = nn.ModuleList([
            TimesBlock(
                d_model=hidden_dim,
                top_k=top_k,
                num_kernels=num_kernels
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.extractor = self.extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate"
            ),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_num),
        )

    
    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        """
        Augment x by adding cosine and sine encoding to the last dimension
        """
        # x.shape = (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        pos = pos.unsqueeze(0).repeat(batch_size, 1)
        sin_enc = torch.sin(pos * 0.01)
        cos_enc = torch.cos(pos * 0.01)
        return torch.cat([x, sin_enc.unsqueeze(-1), cos_enc.unsqueeze(-1)], dim=-1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        # x: (batch_size, seq_len, input_dim)
        means = x.mean(1, keepdim=True).detach()
        x = x.sub(means)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x.div(stdev)

        enc_out: torch.Tensor = self.embedding(x, None)
        # enc_out: (batch_size, seq_len, hidden_dim)

        for block in self.model:
            enc_out = self.norm(block(enc_out))

        enc_out = enc_out.permute(0, 2, 1)
        out: torch.Tensor = self.extractor(enc_out)
        out = out.permute(0, 2, 1)
        # out: (batch_size, output_num, output_dim)
        return out
