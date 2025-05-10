from torch_geometric.datasets import IMDB
import numpy as np
from transformers import  BertModel, BertTokenizer
from Transformer import CustomTransformer
from torch.utils.data import DataLoader, Dataset
from NodeClassifier import NodeClassifier
from Dataloader import MovieKeywordDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch_geometric.datasets import IMDB
from torch.optim.lr_scheduler import StepLR

batch_size=256
num_epochs=100
num_classes=3
movie_keywords=np.load("./data/movie_keywords.npy",allow_pickle=True).tolist()
#dataloader = DataLoader(movie_keywords, batch_size=batch_size, shuffle=True)
dataset = IMDB("./IMDB")
data=dataset[0]
labels=data['movie']['y']
data=MovieKeywordDataset(movie_keywords,labels)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

#device = torch.device("cuda:0")
# 加载预训练的分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载预训练的BERT模型
model = BertModel.from_pretrained('bert-base-uncased')
# inputs = tokenizer(movie_keywords, return_tensors='pt', padding=True, truncation=True, max_length=512)
#inputs = {k: v.to(device) for k, v in inputs.items()}

# 获取模型输出
#outputs = model(**inputs)
#print(outputs)
#last_hidden_state = outputs.last_hidden_state

 #last_hidden_state = last_hidden_state.permute(1, 0, 2)
#attention_mask = inputs['attention_mask'].permute(1, 0)
custom_transformer = CustomTransformer(
    hidden_size=768,  # BERT的隐藏层维度
    nhead=8,          # 多头注意力机制的头数
    dim_feedforward=2048,  # 前馈网络的维度
    dropout=0.1,# Dropout概率
)
# transformer_output = custom_transformer(last_hidden_state, src_key_padding_mask=~attention_mask.bool())
# transformer_output = transformer_output.permute(1, 0, 2)
# print(transformer_output)
# 定义自定义Transformer层

# 定义分类器
node_classifier = NodeClassifier(hidden_size=768, num_classes=num_classes)

# 定义优化器
#optimizer = torch.optim.Adam(list(model.parameters()) + list(custom_transformer.parameters()), lr=1e-5)
optimizer = torch.optim.Adam(list(custom_transformer.parameters()) + list(node_classifier.parameters())+list(model.parameters()), lr=1e-3)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

#定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
best_loss=100
# 训练模型
for epoch in range(num_epochs):
    print(f"================epoch:{epoch}====================")
    for batch in dataloader:
        batch_texts=batch['keywords']
        labels=batch['labels']
        # 使用分词器对文本进行处理
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # 获取BERT模型的输出
        outputs = model(**inputs)

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state

        # 调整维度以适应Transformer输入
        #last_hidden_state = last_hidden_state.permute(1, 0, 2)

        # 获取注意力掩码
        #attention_mask = inputs['attention_mask'].permute(1, 0)
        attention_mask = inputs['attention_mask']

        # 将BERT输出传递给自定义Transformer层
        transformer_output = custom_transformer(last_hidden_state, src_key_padding_mask=~attention_mask.bool())



        # 加权平均池化
        attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, 768).float()
        weighted_sum = torch.sum(transformer_output * attention_mask, dim=1)
        weight_sum = torch.sum(attention_mask, dim=1)
        weighted_pooling_embeddings = (weighted_sum / weight_sum).detach().cpu().numpy()
        optimizer.zero_grad()
        weighted_pooling_embeddings = torch.tensor(weighted_pooling_embeddings, dtype=torch.float32)

        outputs = node_classifier(weighted_pooling_embeddings)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        print("Loss:",loss.item())
        if loss<best_loss:
            best_loss=loss
            torch.save(model,'./model/Bert.pth')
            torch.save(custom_transformer, './model/Transformer.pth')
            torch.save(node_classifier, 'model/BertClassifier.pth')
        # print('=========label============')
        # print(labels[:6])
        # print('=========output============')
        # print(outputs[:6])
    #调整学习率
    scheduler.step()
    with torch.no_grad():  # 关闭梯度计算
        # print("Loss:",loss.item())

        #inputs = weighted_pooling_embeddings
        #labels = batch['labels']

        outputs = outputs
        _, preds = torch.max(outputs, 1)  # 获取预测结果
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    # torch.save(custom_transformer, './model/Transformer.pth')
    # torch.save(node_classifier, './model/BertClassifier.pth')

    #model = torch.load('model.pth')

    # 调整维度以恢复原始维度
        #transformer_output = transformer_output.permute(1, 0, 2)

        #print(transformer_output)
        # 计算损失并进行反向传播
        #loss = compute_loss(transformer_output)  # 替换为你的损失计算函数
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

    #print(f"Epoch {epoch + 1}, Loss: {loss.item()}")