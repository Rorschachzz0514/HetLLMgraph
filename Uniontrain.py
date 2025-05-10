from Projector import Projector
from NodeClassifier import NodeClassifier
import numpy as np
from transformers import  BertModel, BertTokenizer
from Transformer import CustomTransformer
from torch.utils.data import DataLoader, Dataset
from Dataloader import MovieKeywordDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch_geometric.datasets import IMDB
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from ContrastiveLoss import ContrastiveLoss
from Transformer import CustomTransformer
from HGTEncoder import HGTModel
from Dataloader import Dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#加载异构图数据集
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
def maskfinder(total_length, train_mask, batch_index, batch_size):
    """
    根据总数据集大小、训练掩码、当前批次索引和批次大小，生成当前批次的掩码。

    参数:
        total_length (int): 数据集的总长度。
        train_mask (torch.Tensor): 总训练掩码，形状为 [total_length]，布尔类型。
        batch_index (int): 当前批次的索引。
        batch_size (int): 批次大小。

    返回:
        batch_mask (torch.Tensor): 当前批次的掩码，布尔类型。
    """
    # 获取总数据集长度
    total_length = len(train_mask)

    # 获取 train_mask 中所有 True 的索引
    true_indices = torch.nonzero(train_mask, as_tuple=True)[0]

    # 计算当前批次的起始和结束索引
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(true_indices))  # 确保不超过 true_indices 的长度

    # 创建一个全为 False 的掩码
    batch_mask = torch.zeros(total_length, dtype=torch.bool)

    # 将当前批次的 True 索引设置为 True
    #if start_idx < len(true_indices):  # 确保还有数据
    batch_mask[true_indices[start_idx:end_idx]] = True

    return batch_mask
def HGT_test_evaluate(hgt,phgt,classifier, x_dict, edge_index_dict,test_mask,all_labels,tokenizer,bert,transformer,pbert,movie_keywords):
    #hgt.eval()
    #x_dict = x_dict[test_mask]
    #labels = labels[test_mask]
    #hgt.eval()
    #phgt.eval()
    # classifier.eval()
    # bert.eval()
    # transformer.eval()
    # pBert.eval()
    hgt.eval()
    phgt.eval()
    classifier.eval()
    bert.eval()
    transformer.eval()
    pBert.eval()
    with torch.no_grad():

        test_keywords=np.array(movie_keywords)[test_mask['movie']].tolist()
        test_labels = all_labels[test_mask['movie']].cpu()
        inputs = tokenizer(test_keywords, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        outputs=bert(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        transformer_output = transformer(last_hidden_state, src_key_padding_mask=~attention_mask.bool())
        # 加权平均池化
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 768).float()
        weighted_sum = torch.sum(transformer_output * attention_mask, dim=1)
        weight_sum = torch.sum(attention_mask, dim=1)
        weighted_pooling_embeddings = (weighted_sum / weight_sum).detach().cpu().numpy()
        #optimizer.zero_grad()
        weighted_pooling_embeddings = torch.tensor(weighted_pooling_embeddings, dtype=torch.float32).to(device)

        out=pbert(weighted_pooling_embeddings)
        out=classifier(out)
        pred = out.argmax(dim=1).cpu()
        #_,pred = torch.max(out,dim=1)

        print('=====================bert_model========================')

        accuracy = accuracy_score(test_labels, pred)
        precision = precision_score(test_labels, pred, average='weighted')
        recall = recall_score(test_labels, pred, average='weighted')
        f1 = f1_score(test_labels, pred, average='weighted')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        print('=====================hgt_model========================')

        out = hgt(x_dict, edge_index_dict)
        out = phgt(out)
        out = classifier(out)
        pred = out.argmax(dim=1).cpu()

        accuracy = accuracy_score(test_labels, pred[test_mask['movie']])
        precision = precision_score(test_labels ,pred[test_mask['movie']], average='weighted')
        recall = recall_score(test_labels, pred[test_mask['movie']], average='weighted')
        f1 = f1_score(test_labels, pred[test_mask['movie']], average='weighted')
        #print('=====================union_model========================')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        return accuracy, precision, recall, f1

# 划分数据集
train_mask, val_mask, test_mask = split_dataset(data)



#加载文本信息
movie_keywords=np.load("./data/movie_keywords.npy",allow_pickle=True).tolist()
#取训练数据集
words_data=MovieKeywordDataset(np.array(movie_keywords)[train_mask['movie']].tolist(),labels[train_mask['movie']])
batch_size=128
dataloader = DataLoader(words_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metadata=(data.node_types,data.edge_types)

# 加载预训练的分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载预训练的BERT模型
bert = torch.load("./model/Bert.pth").to(device)

# 加载预训练的双层transformer模型
transformer=torch.load('./model/Transformer.pth').to(device)

# 加载预训练的HGT模型9
#hgt=torch.load("./model/HGT.pth").to(device)
hgt=HGTModel(in_channels=3066, hidden_channels=256, out_channels=128,
                 num_layers=2, num_classes=3,
                 metadata=metadata).to(device)
# 加载Bert映射器
pBert=Projector(768,128).to(device)

# 加载Hgt映射器
pHgt=Projector(128,128).to(device)

# 加载分类器
classifier=NodeClassifier(128,3).to(device)

#定义交叉熵损失
criterion = nn.CrossEntropyLoss()

#定义对比损失
contrastive=ContrastiveLoss()

#定义权重系数
#alpha=torch.nn.Parameter(torch.tensor(1.0))

# 初始化优化器
optimizer = torch.optim.Adam(list(pHgt.parameters())+list(classifier.parameters())+list(hgt.parameters())+list(pBert.parameters())+list(transformer.parameters()), lr=0.001)
#optimizer = torch.optim.Adam(list(pBert.parameters())+list(classifier.parameters())+list(transformer.parameters())+list(bert.parameters()), lr=0.001)
#optimizer = torch.optim.Adam(list(pBert.parameters()) + list(classifier.parameters()) + list(transformer.parameters()) + list(bert.parameters()), lr=0.0001)
#optimizer = torch.optim.Adam(list(pBert.parameters()) + list(transformer.parameters()) + list(bert.parameters()), lr=0.01)

#optimizer = torch.optim.Adam(list(pHgt.parameters())+list(classifier.parameters())+list(hgt.parameters()), lr=0.001)
#optimizer = torch.optim.Adam(list(hgt.parameters())+list(pHgt.parameters())+list(classifier.parameters()), lr=0.001)

#步长衰减
#scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# 数据迁徙
x_dict = {k: v.to(device) for k, v in x_dict.items()}
#edge_index_dict=edge_index_dict.to(device)
edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
#edge_index_dict = {k: v.to_sparse().indices().to(device) for k, v in edge_index_dict.items()}
all_labels = labels.to(device)
best_loss=100
num_epochs=100
total_length=len(data['movie'].y)
alpha=0.1
beta=0.9
# 训练模型
for epoch in range(num_epochs):
    print(f"================epoch:{epoch}====================")
    for batch_index,batch in enumerate(dataloader):
        #optimizer.zero_grad()
        batch_texts=batch['keywords']
        #batch_texts = torch.tensor(batch['keywords'], dtype=torch.float).to(device)
        labels=batch['labels'].to(device)
        # 先计算索引
        # # 当前 batch 的起始和结束索引
        # start_idx = batch_index * batch_size
        # end_idx = start_idx + len(batch['labels'])  # 注意：最后一个 batch 可能不满 batch_size
        #
        # # 构造掩码
        # mask = torch.zeros(total_length, dtype=torch.bool)
        # mask[start_idx:end_idx] = True
        mask=maskfinder(len(all_labels),train_mask['movie'],batch_index,batch_size)
        #============================Bert+Transformer部分================================================
        # 使用分词器对文本进行处理
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        # 获取BERT模型的输出
        outputs = bert(**inputs)

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state

        # 调整维度以适应Transformer输入
        #last_hidden_state = last_hidden_state.permute(1, 0, 2)

        # 获取注意力掩码
        #attention_mask = inputs['attention_mask'].permute(1, 0)
        attention_mask = inputs['attention_mask']
        # #transformer.train()
        # bert.train()
        # transformer.train()
        # pBert.train()
        # 将BERT输出传递给自定义Transformer层
        transformer_output = transformer(last_hidden_state, src_key_padding_mask=~attention_mask.bool())



        # 加权平均池化
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 768).float()
        weighted_sum = torch.sum(transformer_output * attention_mask, dim=1)
        weight_sum = torch.sum(attention_mask, dim=1)
        weighted_pooling_embeddings = (weighted_sum / weight_sum).detach().cpu().numpy()
        #optimizer.zero_grad()
        weighted_pooling_embeddings = torch.tensor(weighted_pooling_embeddings, dtype=torch.float32).to(device)

        pBertoutputs = pBert(weighted_pooling_embeddings)

        #============================Hgt部分================================================

        #hgt.train()
        optimizer.zero_grad()

        hgtout = hgt(x_dict, edge_index_dict)
        pHgtout=pHgt(hgtout)
        classifier_hgt_out = classifier(pHgtout)
        classifier_bert_out=classifier(pBertoutputs)
        #
        # # #计算对比损失
        contrastive_loss=contrastive(pHgtout[mask], pBertoutputs, 0.1)
        # # 计算对比交叉熵损失
        # 计算 hgt分类损失
        hgt_classifier_loss=F.cross_entropy(classifier_hgt_out[mask], labels)
        # 计算bert分类损失
        bert_classifier_loss=F.cross_entropy(classifier_bert_out, labels)
        cross_loss = F.cross_entropy(pHgtout[mask], labels)#这里的labels是batch中的labels，并不是整体数据集的labels
        #cross_loss = F.cross_entropy(out, labels)#这里的labels是batch中的labels，并不是整体数据集的labels

        # print(contrastive_loss)
        # print(bert_classifier_loss)
        # print(hgt_classifier_loss)
        # 将对比交叉熵损失和hgt分类损失相加
        #loss=alpha*contrastive_loss+beta*hgt_classifier_loss
        loss=hgt_classifier_loss+contrastive_loss+bert_classifier_loss
        #loss=cross_loss

        #print(alpha)
        loss.backward()
        optimizer.step()






        # loss = criterion(pBertoutputs, labels)
        #
        # loss.backward()
        # optimizer.step()
        print("Loss:",loss.item())
        # if cross_loss<best_loss:
        #     best_loss=cross_loss
        #     torch.save(bert,'./union_model/Bert.pth')
        #     torch.save(transformer, './union_model/Transformer.pth')
        #     torch.save(pBert, './union_model/pbert.pth')
        #     torch.save(classifier, 'model/BertClassifier.pth')
        # print('=========label============')
        # print(labels[:6])
        # print('=========output============')
        # print(outputs[:6])
    #调整学习率
    scheduler.step()
    HGT_test_evaluate(hgt,pHgt,classifier,x_dict,edge_index_dict,test_mask,all_labels,tokenizer,bert,transformer,pBert,movie_keywords)


#HGT_test_evaluate(hgt,pHgt,classifier,x_dict,edge_index_dict,test_mask,all_labels)