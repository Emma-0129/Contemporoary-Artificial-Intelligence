# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import os
from utils.config import config
import torch.nn.functional as F


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'LSTM_Alexnet'
        self.train_path = 'data/' + dataset + '/train.txt'  # 训练集
        self.dev_path = 'data/' + dataset + '/dev.txt'  # 验证集
        self.test_path = 'data/' + dataset + '/test.txt'  # 测试集集

        self.class_list = [x.strip() for x in open(
            'data/' + dataset + '/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = config["vocab_path"]  # 词表
        if not os.path.exists('saved_dict/'): os.mkdir('saved_dict/')
        self.save_path = './saved_dict/' + dataset + '-' + self.model_name + '.ckpt'  # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            np.load('./pretrained/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda:' + config["gpu"] if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = config["dropout"]  # 随机失活
        self.patience = config["patience"]
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = config["epochs"]  # epoch数
        self.batch_size = config["batch_size"]  # mini-batch大小
        self.pad_size = config["pad_size"]  # 每句话处理成的长度(短填长切)
        self.learning_rate = config["learning_rate"]  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数

        self.save_result = config["save_result"]


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


# alexnet+lstm
class Model(nn.Module):
    def __init__(self, config, blocks):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size * 2, 1)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048)
        )

        self.fc2 = nn.Linear(2048+1, config.num_classes)

    def forward(self, text, image):
        text, _ = text
        text = self.embedding(text)  # [batch_size, seq_len, embeding]=[128, 64, 300]
        text, _ = self.lstm(text)

        text = self.fc1(text[:, -1, :])  # 句子最后时刻的 hidden state

        image = self.features(image)
        image = image.view(image.size(0), 256 * 6 * 6)
        image = self.classifier(image)

        out = self.fc2(torch.cat((image, text), 1))

        return out
