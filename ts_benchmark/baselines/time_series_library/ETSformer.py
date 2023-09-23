import torch
import torch.nn as nn
from .layers.Embed import DataEmbedding
from .layers.ETSformer_EncDec import (
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    Transform,
)


class ETSformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2202.01381
    """

    def __init__(self, config):
        super(ETSformer, self).__init__()
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        if (
            self.task_name == "classification"
            or self.task_name == "anomaly_detection"
            or self.task_name == "imputation"
        ):
            self.pred_len = config.seq_len
        else:
            self.pred_len = config.pred_len

        assert (
            config.e_layers == config.d_layers
        ), "Encoder and decoder layers must be equal"

        # Embedding
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.embed, config.freq, config.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.enc_in,
                    config.seq_len,
                    self.pred_len,
                    config.top_k,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ]
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.c_out,
                    self.pred_len,
                    dropout=config.dropout,
                )
                for _ in range(config.d_layers)
            ],
        )
        self.transform = Transform(sigma=0.2)

        if self.task_name == "classification":
            self.act = torch.nn.functional.gelu
            self.dropout = nn.Dropout(config.dropout)
            self.projection = nn.Linear(
                config.d_model * config.seq_len, config.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def anomaly_detection(self, x_enc):
        res = self.enc_embedding(x_enc, None)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def classification(self, x_enc, x_mark_enc):
        res = self.enc_embedding(x_enc, None)
        _, growths, seasons = self.encoder(res, x_enc, attn_mask=None)

        growths = torch.sum(torch.stack(growths, 0), 0)[:, : self.seq_len, :]
        seasons = torch.sum(torch.stack(seasons, 0), 0)[:, : self.seq_len, :]

        enc_out = growths + seasons
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)

        # Output
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
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
