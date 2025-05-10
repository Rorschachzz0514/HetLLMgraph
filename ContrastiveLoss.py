import torch
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, hgt_embedding, transformer_embedding,temperature):
        """
        InfoNCE 损失函数。
        """
        """
            InfoNCE Loss 实现
            hgt_embedding: 异构图嵌入 [batch_size, embedding_dim]
            transformer_embedding: 语言模型嵌入 [batch_size, embedding_dim]
            temperature: 温度参数
            """
        # 计算所有嵌入之间的相似度矩阵
        # 使用余弦相似度：sim(z_i, z_j) = (z_i / ||z_i||) * (z_j / ||z_j||)
        hgt_embedding = F.normalize(hgt_embedding, p=2, dim=-1)  # 归一化
        transformer_embedding = F.normalize(transformer_embedding, p=2, dim=-1)  # 归一化
        similarity_matrix = torch.mm(hgt_embedding, transformer_embedding.t()) / temperature

        # 构造标签：每个样本的正样本是其自身
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

        # 计算 InfoNCE 损失
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
