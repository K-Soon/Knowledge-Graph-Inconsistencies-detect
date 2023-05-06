import networkx as nx
import numpy as np
import torch
import torch.nn as nn

# 构建节点
# 人物节点
person_node = ['person_1', 'person_2', 'person_3']
# 音乐节点
music_node = ['music_1', 'music_2', 'music_3']
# 电影节点
movie_node = ['movie_1', 'movie_2', 'movie_3']

# 定义节点特征
# 人物出生日期
person_birthday = {'person_1': '1990-01-01', 'person_2': '1995-05-01', 'person_3': '1988-06-02'}
# 人物死亡日期
person_deathday = {'person_1': '2050-12-31', 'person_2': '2090-01-01'}
# 音乐发行日期
music_release_date = {'music_1': '2010-01-01', 'music_2': '2015-05-01', 'music_3': '2020-06-02'}
# 电影发行日期
movie_release_date = {'movie_1': '2010-01-01', 'movie_2': '2015-05-01', 'movie_3': '2020-06-02'}

# 构建图
G = nx.MultiDiGraph()

# 添加人物节点
for person in person_node:
    G.add_node(person)
    # 添加人物出生日期属性
    G.nodes[person]['birthday'] = person_birthday[person]
    # 添加人物死亡日期属性
    if person in person_deathday:
        G.nodes[person]['deathday'] = person_deathday[person]

# 添加音乐节点
for music in music_node:
    G.add_node(music)
    # 添加音乐发行日期属性
    G.nodes[music]['release_date'] = music_release_date[music]

# 添加电影节点
for movie in movie_node:
    G.add_node(movie)
    # 添加电影发行日期属性
    G.nodes[movie]['release_date'] = movie_release_date[movie]

# 添加边
G.add_edge('person_1', 'music_1', relation='composer')
G.add_edge('person_2', 'music_1', relation='producer')
G.add_edge('person_3', 'music_3', relation='composer')
G.add_edge('person_2', 'movie_2', relation='producer')
G.add_edge('person_1', 'movie_3', relation='actor')
G.add_edge('music_1', 'movie_1', relation='soundtrack')
G.add_edge('music_2', 'movie_2', relation='soundtrack')
G.add_edge('music_3', 'movie_3', relation='soundtrack')


import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x




# 转换标签为one-hot编码
y_train = torch.zeros(len(train_nodes), n_class)
y_train[range(len(train_nodes)), train_labels] = 1
y_val = torch.zeros(len(val_nodes), n_class)
y_val[range(len(val_nodes)), val_labels] = 1

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(200):
    model.train()
    logits = model(data)
    train_loss = loss_fn(logits[train_nodes], train_labels)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 验证模型
    model.eval()
    logits = model(data)
    val_loss = loss_fn(logits[val_nodes], val_labels)
    val_acc = accuracy(logits[val_nodes], val_labels)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc:.4f}")

