import torch
import torch.nn as nn

import torch.fft

from einops import rearrange, repeat
from .layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from .layers.Embed import PatchEmbedding
from .layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    TwoStageAttentionLayer,
)
from math import ceil
from .PatchTST import FlattenHead
import warnings

warnings.filterwarnings("ignore")


class Crossformer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(self, config):
        super(Crossformer, self).__init__()
        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = config.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * config.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * config.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (self.win_size ** (config.e_layers - 1))
        )
        self.head_nf = config.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            config.d_model,
            self.seg_len,
            self.seg_len,
            self.pad_in_len - config.seq_len,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, config.enc_in, self.in_seg_num, config.d_model)
        )
        self.pre_norm = nn.LayerNorm(config.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    config,
                    1 if l is 0 else self.win_size,
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    1,
                    config.dropout,
                    self.in_seg_num
                    if l is 0
                    else ceil(self.in_seg_num / self.win_size**l),
                    config.factor,
                )
                for l in range(config.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(
                1, config.enc_in, (self.pad_out_len // self.seg_len), config.d_model
            )
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        config,
                        (self.pad_out_len // self.seg_len),
                        config.factor,
                        config.d_model,
                        config.n_heads,
                        config.d_ff,
                        config.dropout,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    self.seg_len,
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    # activation=config.activation,
                )
                for l in range(config.e_layers + 1)
            ],
        )
        if self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.head = FlattenHead(
                config.enc_in, self.head_nf, config.seq_len, head_dropout=config.dropout
            )
        elif self.task_name == "classification":
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(config.dropout)
            self.projection = nn.Linear(self.head_nf * config.enc_in, config.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_enc.shape[0],
        )
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        return dec_out

    def anomaly_detection(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
