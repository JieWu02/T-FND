import numpy as np
import pandas as pd

from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def make_dfs(config):
    train_dataframe = pd.read_csv(config.train_text_path)
    train_dataframe.dropna(subset=['text'], inplace=True)
    train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
    train_dataframe.label = train_dataframe.label.apply(lambda x: config.classes.index(x))
    config.class_weights = get_class_weights(train_dataframe.label.values)

    if config.test_text_path is None:
        offset = int(train_dataframe.shape[0] * 0.80)
        test_dataframe = train_dataframe[offset:]
        train_dataframe = train_dataframe[:offset]
    else:
        test_dataframe = pd.read_csv(config.test_text_path)
        test_dataframe.dropna(subset=['text'], inplace=True)
        test_dataframe.label = test_dataframe.label.apply(lambda x: config.classes.index(x))

    if config.validation_text_path is None:
        offset = int(train_dataframe.shape[0] * 0.90)
        validation_dataframe = train_dataframe[offset:]
        train_dataframe = train_dataframe[:offset]
    else:
        validation_dataframe = pd.read_csv(config.validation_text_path)
        validation_dataframe.dropna(subset=['text'], inplace=True)
        validation_dataframe = validation_dataframe.sample(frac=1).reset_index(drop=True)
        validation_dataframe.label = validation_dataframe.label.apply(lambda x: config.classes.index(x))

    return train_dataframe, test_dataframe, validation_dataframe


def build_loaders(config, dataframe, mode, distributed=False, rank=0, world_size=1):
    DatasetLoader = config.get_dataset_loader()
    dataset = DatasetLoader(config, dataframe=dataframe, mode=mode)
    
    # 创建分布式采样器
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(mode == 'train'))
        shuffle = False  # 使用sampler时不能使用shuffle
    
    if mode != 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size // 2,
            num_workers=config.num_workers,
            pin_memory=False,
            shuffle=shuffle and (mode == 'train'),
            sampler=sampler,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
        )
    return dataloader


def get_class_weights(y):
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return class_weights
