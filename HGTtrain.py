from torch_geometric.datasets import IMDB
from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
import torch.nn.functional as F
from HGTEncoder import HGTModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from NodeClassifier import NodeClassifier
from torch.optim.lr_scheduler import StepLR

# 加载IMDB数据集
dataset = IMDB(root='./IMDB')
data = dataset[0]



# 获取节点特征和标签
x_dict = {node_type: data[node_type].x for node_type in data.node_types}
edge_index_dict = data.edge_index_dict

# 获取标签
labels = data['movie'].y
train_mask = data['movie'].train_mask

# 定义切分比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    train_mask = {}
    val_mask = {}
    test_mask = {}


    num_nodes = data["movie"].num_nodes
    indices = torch.arange(num_nodes)

    # 划分训练集、验证集和测试集
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=random_state)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)),
                                             random_state=random_state)

    train_mask["movie"] = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, train_idx, True)
    val_mask["movie"] = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, val_idx, True)
    test_mask["movie"] = torch.zeros(num_nodes, dtype=torch.bool).scatter_(0, test_idx, True)

    return train_mask, val_mask, test_mask


# 划分数据集
train_mask, val_mask, test_mask = split_dataset(data)
# 创建 DataLoader
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metadata=(data.node_types,data.edge_types)
# 初始化模型
model = HGTModel(in_channels=3066, hidden_channels=256, out_channels=128,
                 num_layers=2, num_classes=3,
                 metadata=metadata).to(device)
classifier=NodeClassifier(128,3).to(device)
criterion = nn.CrossEntropyLoss()

# 初始化优化器
optimizer = torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=0.001)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

# 将数据移动到设备
x_dict = {k: v.to(device) for k, v in x_dict.items()}
#edge_index_dict=edge_index_dict.to(device)
edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
#edge_index_dict = {k: v.to_sparse().indices().to(device) for k, v in edge_index_dict.items()}
labels = labels.to(device)
best_loss=100
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(x_dict, edge_index_dict)
#     loss = F.cross_entropy(out, labels)
#     loss.backward()
#     optimizer.step()
#     return loss.item()


def train(model, x_dict, edge_index_dict,optimizer,train_mask,labels):
    model.train()
    optimizer.zero_grad()
    out = model(x_dict, edge_index_dict)
    out=classifier(out)
    loss = F.cross_entropy(out[train_mask['movie']], labels[train_mask['movie']])
    loss.backward()
    optimizer.step()

    return loss.item()

# def model_test():
#     model.eval()
#     with torch.no_grad():
#         out = model(x_dict, edge_index_dict)
#         pred = out.argmax(dim=1)
#         accuracy = accuracy_score(labels.cpu(), pred.cpu())
#         precision = precision_score(labels.cpu(), pred.cpu(), average='weighted')
#         recall = recall_score(labels.cpu(), pred.cpu(), average='weighted')
#         f1 = f1_score(labels.cpu(), pred.cpu(), average='weighted')
#         return accuracy, precision, recall, f1
def valid_evaluate(model,x_dict, edge_index_dict,val_mask,labels):
    model.eval()
    #x_dict = x_dict[val_mask]
    #labels = labels[val_mask]
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        out=classifier(out)
        pred = out.argmax(dim=1)

        accuracy = accuracy_score(labels[val_mask['movie']].cpu(), pred[val_mask['movie']].cpu())
        precision = precision_score(labels[val_mask['movie']].cpu(), pred[val_mask['movie']].cpu(), average='weighted')
        recall = recall_score(labels[val_mask['movie']].cpu(), pred[val_mask['movie']].cpu(), average='weighted')
        f1 = f1_score(labels[val_mask['movie']].cpu(), pred[val_mask['movie']].cpu(), average='weighted')
        loss = F.cross_entropy(out[val_mask['movie']], labels[val_mask['movie']])
        #loss.backward()
        print('=====================valid========================')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        return loss,accuracy, precision, recall, f1


def Test_evaluate(model, x_dict, edge_index_dict,test_mask,labels):
    model.eval()
    #x_dict = x_dict[test_mask]
    #labels = labels[test_mask]
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        out=classifier(out)

        pred = out.argmax(dim=1)

        accuracy = accuracy_score(labels[test_mask['movie']].cpu(), pred[test_mask['movie']].cpu())
        precision = precision_score(labels[test_mask['movie']].cpu(), pred[test_mask['movie']].cpu(), average='weighted')
        recall = recall_score(labels[test_mask['movie']].cpu(), pred[test_mask['movie']].cpu(), average='weighted')
        f1 = f1_score(labels[test_mask['movie']].cpu(), pred[test_mask['movie']].cpu(), average='weighted')
        print('=====================union_model========================')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        return accuracy, precision, recall, f1

# 训练模型
for epoch in range(200):
    train_loss = train(model,x_dict,edge_index_dict, optimizer,train_mask,labels)
    if train_loss<best_loss:
        best_loss=train_loss
        torch.save(model, './model/HGT.pth')

    valid_loss,accuracy, precision, recall, f1 = valid_evaluate(model,x_dict,edge_index_dict,val_mask,labels)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    scheduler.step()

    # loss = train()
    # print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
    # accuracy, precision, recall, f1 = model_test()
    # print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

test_loss = Test_evaluate(model,x_dict,edge_index_dict,test_mask,labels)
#print(f'Test Loss: {test_loss:.4f}')
# accuracy, precision, recall, f1 = model_test()
# print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')