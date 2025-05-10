import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class HGTModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_classes, metadata):
        super(HGTModel, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # 每层的输入和输出维度
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            # 添加HGT层
            self.convs.append(HGTConv(in_channels=in_dim, out_channels=out_dim, 
                                      metadata=metadata))
        # 分类器
        #self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # 假设我们对所有节点进行分类
        out=x_dict['movie']
        #out = self.classifier(x_dict['movie'])  # 替换为你的节点类型
        return out
