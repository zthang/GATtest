import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from curvGN import curvGN
from torch_geometric.utils import add_self_loops, remove_self_loops,degree,softmax

class Net(torch.nn.Module):
    def __init__(self, dataset, w_mul):
        super(Net, self).__init__()
        self.lin = torch.nn.Linear(2*dataset.num_classes, dataset.num_classes)
        self.conv1 = GATConv(w_mul, dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(w_mul, 9 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)
        self.curv_conv1 = curvGN(dataset.num_features, 8, 8, w_mul)
        self.curv_conv2 = curvGN(9 * 8, dataset.num_classes, dataset.num_classes, w_mul)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        # x_curv = self.curv_conv1(x, data.edge_index)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)

def call(data, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = 'data/curvature/graph_' + "Cora" + '.edge_list_FormanRicci'
    f = open(filename)
    cur_list = list(f)
    ricci_cur = [[] for i in range(2 * len(cur_list))]
    for i in range(len(cur_list)):
        ricci_cur[i] = [num(s) for s in cur_list[i].split(' ', 2)]
        ricci_cur[i + len(cur_list)] = [ricci_cur[i][1], ricci_cur[i][0], ricci_cur[i][2]]
    ricci_cur = sorted(ricci_cur)
    eg_index0 = [i[0] for i in ricci_cur]
    eg_index1 = [i[1] for i in ricci_cur]
    eg_index = torch.stack((torch.tensor(eg_index0), torch.tensor(eg_index1)), dim=0)
    data.edge_index = eg_index
    w_mul = [i[2] for i in ricci_cur]
    w_mul = w_mul + [0 for i in range(data.x.size(0))]
    w_mul = torch.tensor(w_mul, dtype=torch.float)
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.w_mul = w_mul.view(-1, 1)
    data.w_mul = data.w_mul.to(device)
    return Net(dataset, data.w_mul).to(device), data.to(device)