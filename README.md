# Contemporoary-Artificial-Intelligence

# 任务介绍

基于文本图像的多模态图像识别

## 环境
tqdm==4.62.3
torchvision==0.9.1+cu111
numpy==1.19.5
torch==1.8.1+cu111
transformers==4.15.0
Pillow==9.2.0
scikit_learn==1.1.1

## 训数据预处理

对train数据进行shuffle后按8:2比例切分成训练集和验证集，图片和文本分别处理，图片部分处理成
`编号,label`的txt文件，文本部分处理成`文本+\t+label`的txt文件

> cd data
> python data_process

## 模型介绍

本实验采用基于word2vec+LSTM+Alexnet的多模态模型，对于文本部分采用glove预训练的英文word2vec词向量加上bilstm模型进行特征抽取，对于图像部分采用Alexnet模型进行特征抽取，将两种模态特征进行concat操作后输入MLP进行结果分类

## 训练和预测
训练结束后效果最好的模型保存在`saved_dict`文件夹下，生成的预测文件在`data/实验五数据/test_with_label.txt`路径下
> python run.py 

### 模型保存
超参数在`utils/config.py`处设置，最好模型保存在saved_dict文件夹，训练过程和模型构造早log文件夹

## 消融实验和结果说明
对于该分类任务，本实验采用基于word2vec+LSTM+Alexnet的多模态模型但是在后续消融实验中，若采用Alexnet对纯图像进行分类，效果较好。通过对数据集的分析，我们可以发现当前模型对于处理这样复杂语言语法交融的数据集的处理没有很好，这需要后期的优化和精细化。


