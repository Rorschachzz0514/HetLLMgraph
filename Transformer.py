from transformers import BertConfig, BertModel, BertTokenizer
import torch.nn as nn

class CustomTransformer(nn.Module):
    def __init__(self, hidden_size, nhead, dim_feedforward, dropout):
        super(CustomTransformer, self).__init__()
        # 定义一个Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,  # 输入特征的维度
            nhead=nhead,         # 多头注意力机制的头数
            dim_feedforward=dim_feedforward,  # 前馈网络的维度
            dropout=dropout,
            batch_first=True# Dropout概率
        )
        # 定义两个Transformer层
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2  # 定义两层Transformer
        )

    def forward(self, src, src_key_padding_mask=None):
        # src: 输入张量，形状为 (sequence_length, batch_size, hidden_size)
        # src_key_padding_mask: 用于指示哪些位置是填充的，形状为 (batch_size, sequence_length)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output