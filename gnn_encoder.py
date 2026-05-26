import torch
import torch.nn as nn

from torch_geometric.data import Data, Batch
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class GNNEncoder(nn.Module):
    def __init__(self, k=8):

        super(GNNEncoder, self).__init__()

        self.layer1 = DynamicEdgeConv(
            MLP(6, 64),
            k=k,
            aggr='mean'
        )

        self.layer2 = DynamicEdgeConv(
            MLP(128, 128),
            k=k,
            aggr='mean'
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.layer_list = [self.layer1, self.layer2]

    def forward(self, data):

        x = data.x
        batch = data.batch

        for layer in self.layer_list:
            x = layer(x, batch)

        x = global_mean_pool(x, batch)

        x = self.final_mlp(x)

        return x


def collate_fn_gnn(batch):

    data_list = []
    labels = []

    for b in batch:

        tensor_data = torch.from_numpy(b["data"].to_numpy()).T
        tensor_data = tensor_data.to(dtype=torch.float32)

        this_graph_item = Data(x=tensor_data)

        data_list.append(this_graph_item)

        labels.append(
            torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0)
        )

    labels = torch.cat(labels, dim=0)

    packed_data = Batch.from_data_list(data_list)

    return packed_data, labels