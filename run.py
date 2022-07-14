# coding: UTF-8
import time
import torch
import numpy as np
from utils.train_eval import train
from importlib import import_module
from utils.log import logger
from utils.config import config
from utils.util import build_dataset, get_time_dif, get_iter

if __name__ == '__main__':
    dataset = config["dataset"]  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz  随机初始化:random
    embedding = 'glove/embedding_glove.npz'

    model_name = config["model_name"]

    x = import_module('models.' + model_name)
    logger.info(f'{model_name} model starts training...')

    config = x.Config(dataset, embedding)
    vocab, train_data, dev_data, test_data = build_dataset(config, )
    config.n_vocab = len(vocab)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logger.info("Loading data...")
    train_iter, dev_iter, test_iter = get_iter(train_data, dev_data, test_data, config)

    time_dif = get_time_dif(start_time)
    logger.info(f"Time usage:{time_dif}")

    # train

    model = x.Model(config, [3, 8, 36, 3]).to(config.device)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
    logger.info(f'{model_name} model training end...')
