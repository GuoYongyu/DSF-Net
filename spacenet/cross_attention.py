# coding: utf-8
import torch
import torch.nn as nn


class CrossAttentionEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 4,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_heads: int = 4,
            output_num: int = 24,
            output_dim: int = 128,
        ):
        super(CrossAttentionEncoder, self).__init__()
        self.name = "SpaceFeatureNet(CrossAttentionEncoder)"

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        # distinguish 4 types of spatial features
        self.spatial_embed = nn.Embedding(
            num_embeddings=4,
            embedding_dim=hidden_dim,
        )
        self.extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim * 4,
                out_channels=output_dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate"
            ),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_num),
        )


    def forward(
            self,
            down_feat: torch.Tensor,
            up_feat: torch.Tensor,
            in_feat: torch.Tensor,
            out_feat: torch.Tensor
        ) -> torch.Tensor:
        # input: (batch_size, seq_len, input_dim)
        device = down_feat.device

        # feature mapping (0: down, 1: up, 2: in, 3: out)
        down_enc = self.proj(down_feat) + self.spatial_embed(torch.tensor(0, device=device))
        up_enc = self.proj(up_feat) + self.spatial_embed(torch.tensor(1, device=device))
        in_enc = self.proj(in_feat) + self.spatial_embed(torch.tensor(2, device=device))
        out_enc = self.proj(out_feat) + self.spatial_embed(torch.tensor(3, device=device))

        # cross attention
        spatial_features = torch.stack([down_enc, up_enc, in_enc, out_enc], dim=1)

        for attn_layer in self.encoder_layers:
            updated = list()
            for i in range(4):
                # query.shape: (batch_size, seq_len, hidden_dim)
                query = spatial_features[:, i]
                # key_value.shape: (batch_size, 3 * seq_len, hidden_dim)
                key_value = torch.cat([spatial_features[:, j] for j in range(4) if j != i], dim=1)
                attn_out: torch.Tensor = attn_layer(
                    query=query,
                    key=key_value,
                    value=key_value,
                    need_weights=False
                )[0]
                updated.append(attn_out)
            spatial_features = torch.stack(updated, dim=1)
            # spatial_features: (batch_size, 4, seq_len, hidden_dim)

        # fuse 4 types of spatial features
        cat_features = torch.cat([
            spatial_features[:, 0],
            spatial_features[:, 1],
            spatial_features[:, 2],
            spatial_features[:, 3],
        ], dim=-1).permute(0, 2, 1)
        out: torch.Tensor = self.extractor(cat_features)
        out = out.permute(0, 2, 1)
        # out: (batch_size, output_num, output_dim)
        return out
