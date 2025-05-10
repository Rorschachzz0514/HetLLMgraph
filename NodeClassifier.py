import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class NodeClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(NodeClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings):
        return self.fc(embeddings)