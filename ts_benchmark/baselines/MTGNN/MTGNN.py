import torch
import pickle
from pathlib import Path
from .layer import *
import scipy.sparse as sp
import numpy as np
import os
import pandas as pd


def load_adj(pkl_filename):
    """
    为什么gw的邻接矩阵要做对称归一化，而dcrnn的不做？其实做了，在不同的地方，是为了执行双向随机游走算法。
    所以K-order GCN需要什么样的邻接矩阵？
    这个应该参考ASTGCN，原始邻接矩阵呢？参考dcrnn
    为什么ASTGCN不采用对称归一化的拉普拉斯矩阵？
    :param pkl_filename: adj_mx.pkl
    :return:
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pkl_filename):
    try:
        with Path(pkl_filename).open("rb") as f:
            pkl_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with Path(pkl_filename).open("rb") as f:
            pkl_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pkl_filename, ":", e)
        raise

    return pkl_data


def load_sparse_adj(adj_path):
    adj = sp.load_npz(os.path.join(adj_path))
    adj = adj.tocsc()

    return adj


class MTGNN(nn.Module):
    def __init__(
        self,
        config,
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=7,
        device=torch.device("cuda"),
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=12,
        in_dim=1,
        out_dim=12,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ):
        super(MTGNN, self).__init__()
        self.gcn_true = gcn_true
        self.num_nodes = config.num_node
        self.dropout = dropout
        if config.dataset_name == "PEMS-BAY":
            self.buildA_true = False
            _, _, self.predefined_A = load_adj("../dataset/adj_mx_bay.pkl")
            self.predefined_A = torch.tensor(self.predefined_A).to(device)
        elif config.dataset_name == "PEMSD7M":
            self.buildA_true = False
            self.predefined_A = sp.load_npz("../dataset/pemsd7m_adj.npz")["data"]
            self.predefined_A = torch.tensor(self.predefined_A).to(device)

        elif config.dataset_name == "NYC_TAXI":
            self.buildA_true = False
            self.predefined_A = pd.read_csv("../dataset/nyc_taxi_adj.csv", header=None).values.astype(
                np.float32
            )
            self.predefined_A = torch.tensor(self.predefined_A).to(device)

        elif config.dataset_name == "NYC_BIKE":
            self.buildA_true = False
            self.predefined_A = pd.read_csv("../dataset/nyc_bike_adj.csv", header=None).values.astype(
                np.float32
            )
            self.predefined_A = torch.tensor(self.predefined_A).to(device)

        else:
            self.buildA_true = True
            self.predefined_A = None
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.gc = graph_constructor(
            self.num_nodes,
            subgraph_size,
            node_dim,
            device,
            alpha=tanhalpha,
            static_feat=static_feat,
        )

        self.seq_length = config.seq_len
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (dilation_exponential**layers - 1)
                    / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (dilation_exponential**j - 1)
                        / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.gate_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )
                    self.gconv2.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                config.num_node,
                                self.seq_length - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                config.num_node,
                                self.receptive_field - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=config.pred_len,
            kernel_size=(1, 1),
            bias=True,
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )

        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input: torch.Tensor, idx=None):
        input = input.permute(0, 3, 2, 1)
        seq_len = input.size(3)
        assert (
            seq_len == self.seq_length
        ), "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0)
            )

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
