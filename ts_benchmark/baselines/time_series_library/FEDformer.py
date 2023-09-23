import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding
from .layers.AutoCorrelation import AutoCorrelationLayer
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, config, version="fourier", mode_select="random", modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDformer, self).__init__()
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(config.moving_avg)
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.embed, config.freq, config.dropout
        )
        self.dec_embedding = DataEmbedding(
            config.dec_in, config.d_model, config.embed, config.freq, config.dropout
        )

        if self.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=config.d_model, L=1, base="legendre"
            )
            decoder_self_att = MultiWaveletTransform(
                ich=config.d_model, L=1, base="legendre"
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=config.d_model,
                base="legendre",
                activation="tanh",
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=config.d_model,
                out_channels=config.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=my_Layernorm(config.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att, config.d_model, config.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, config.d_model, config.n_heads
                    ),
                    config.d_model,
                    config.c_out,
                    config.d_ff,
                    moving_avg=config.moving_avg,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.d_layers)
            ],
            norm_layer=my_Layernorm(config.d_model),
            projection=nn.Linear(config.d_model, config.c_out, bias=True),
        )

        if self.task_name == "imputation":
            self.projection = nn.Linear(config.d_model, config.c_out, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(config.d_model, config.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(config.dropout)
            self.projection = nn.Linear(
                config.d_model * config.seq_len, config.num_class
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
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
