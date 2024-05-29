# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from litgpt import Tokenizer, Task2prompt, Task2tokens, modality_tokens_to_string
from litgpt.data import DataModule

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.lines = []
        # 遍历文件夹，获取所有txt文件的路径
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                # 读取文件并按行存储
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.lines.extend(file.readlines())

    def __len__(self):
        # 返回数据集中的行数
        return len(self.lines)

    def __getitem__(self, index):
        # 获取并返回指定索引的行内容
        return self.lines[index].strip()


class CustomDatasetForUnconditionedVideo(Dataset):
    def __init__(self, folder_path):
        self.video_token_paths = [os.path.join(folder_path, video_id, "result.npz") for video_id in os.listdir(folder_path)]

    def __len__(self):
        # 返回数据集中的行数
        return len(self.video_token_paths)

    def __getitem__(self, index):
        # 获取并返回指定索引的行内容
        video_token_path = self.video_token_paths[index]
        with np.load(video_token_path) as data:
            # 获取名为 'data' 的 NumPy 数组
            quantized = np.abs(data['data'].reshape(-1)).astype(np.int32)
            # reconstructed_array = quantized.reshape(array.shape)
        return quantized


@dataclass
class CustomData(DataModule):
    """The CustomData data module for pretraining."""

    data_path: Union[str, Path] = Path("/app/wen/litgpt/data/video_tokens")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.2
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 0
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=4096, repr=False, init=False)

    def __post_init__(self) -> None:
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        if str(self.data_path).startswith("s3://"):
            print(f"The CustomData data path points to an S3 location: {self.data_path}. Skipping preprocessing.")
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"Found CustomData train and val dir: {self.data_path}. Skipping preprocessing.")
            return

        mode="unconditional video"
        if mode == "unconditional video":
            dataset = CustomDatasetForUnconditionedVideo(self.data_path)
        else:
            dataset = CustomDataset(self.data_path)
        # 计算训练集和验证集的大小
        val_size = int(self.val_split_fraction * len(dataset))
        train_size = len(dataset) - val_size
        # 随机分割数据集
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        split_dataset = {"train": train_dataset, "val": val_dataset}

        def tokenize(data: Dataset, model: str, index: int):
            # yield self.tokenizer.encode(data[index], eos=True)
            if mode == "unconditional video":
                video_token = data[index]
                video_token_str = modality_tokens_to_string(video_token)
                prompt = Task2prompt % (Task2tokens["unconditional video"], "", "", "", video_token_str, "")
            else:
                prompt = Task2prompt % (Task2tokens["text-video"], data[index], "aaa"*5*16*16, "bbb"*16*1024, "", "")
            print(prompt)
            # if len(prompt) != self.seq_length:
            #     prompt = (prompt+" aaaa"*self.seq_length)[: self.seq_length]
            yield self.tokenizer.encode(prompt, eos=False)

        optimize(
            fn=partial(tokenize, split_dataset["train"], mode),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=(self.num_workers),
            chunk_bytes="200MB",
        )
        optimize(
            fn=partial(tokenize, split_dataset["val"], mode),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=(self.num_workers),
            chunk_bytes="200MB",
        )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=self.data_path_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader
