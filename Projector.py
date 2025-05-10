import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class Projector(nn.Module):
    def __init__(self, hidden_size, out_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, embeddings):
        return self.fc(embeddings)