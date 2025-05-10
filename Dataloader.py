from torch.utils.data import Dataset, DataLoader
import torch
class MovieKeywordDataset(Dataset):
    def __init__(self, keywords, labels):
        self.keywords = keywords
        self.labels = labels
        #self.tokenizer = tokenizer
        #self.max_length = max_length

    def __len__(self):
        return len(self.keywords)

    def __getitem__(self, idx):
        keyword = self.keywords[idx]
        label = self.labels[idx]
        return {
            'keywords': keyword,
            'labels': torch.tensor(label, dtype=torch.long)
        }
