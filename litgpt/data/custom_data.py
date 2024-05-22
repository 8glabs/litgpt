# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset, DataLoader, random_split

from litgpt import Tokenizer
from litgpt.data import DataModule

text_prefix = ""
audio_prefix = "👂"
video_prefix = "📹"
task_prefix = "📋"

start_of_seq, end_of_seq = '<sos>', '<eos>'
start_of_text, end_of_text = '<bot_i>', '<eot_i>'
start_of_input_visual, end_of_input_visual = '<bov_i>', '<eov_i>'
start_of_input_audio, end_of_input_audio = '<boa_i>', '<eoa_i>'
source_of_video = '<source>'
resolution_of_video = '<res>'
start_of_output_visual, end_of_output_visual = '<bov_o>', '<eov_o>'
start_of_output_audio, end_of_output_audio = '<boa_o>', '<eoa_o>'
modal_specials = [start_of_seq, end_of_seq, start_of_text, end_of_text, start_of_input_visual, end_of_input_visual, start_of_input_audio, end_of_input_audio, source_of_video, resolution_of_video, start_of_output_visual, end_of_output_visual, start_of_output_audio, end_of_output_audio]

text_vocab_size=32000
audio_codebook_size=1024
audio_codebook_num=16
audio_vocab_size=audio_codebook_size * audio_codebook_num
visual_vocab_size=2**18


task2tokens = {
    "text-video": f"{task_prefix}0",
    "image-vodeo": f"{task_prefix}1",
    "unconditional video": f"{task_prefix}2",
    "Audioavatar talkingheads": f"{task_prefix}3",
    "Audioavatar singingheads": f"{task_prefix}4",
    "text-image": f"{task_prefix}5",
}

modal_special_str = {
    "text":{
        "prefix": text_prefix,
        "sos": start_of_text,
        "eos": end_of_text,
        "vocab_size": text_vocab_size
    },
    "audio":{
        "prefix": audio_prefix,
        "sos": start_of_input_audio,
        "eos": end_of_input_audio,
        "vocab_size": audio_vocab_size
    },
    "visual":{
        "prefix": video_prefix,
        "sos": start_of_input_visual,
        "eos": end_of_input_visual,
        "vocab_size": visual_vocab_size
    },
}

# task_id, text, input_visual, input_audio, output_visual, output_audio
task_prompt = f"{start_of_seq}%s{start_of_text}%s{end_of_text}{start_of_input_visual}%s{end_of_input_visual}{start_of_input_audio}%s{end_of_input_audio}{source_of_video}{resolution_of_video}{start_of_output_visual}%s{end_of_output_visual}{start_of_output_audio}%s{end_of_output_audio}{end_of_seq}"

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


@dataclass
class CustomData(DataModule):
    """The OpenWebText data module for pretraining."""

    data_path: Union[str, Path] = Path("data/custom_data")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 1
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=128, repr=False, init=False)

    def __post_init__(self) -> None:
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def add_custom_tokens(self) -> None:
        for modal_special in modal_specials:
            if modal_special not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([modal_special])
        for modality in modal_special_str.keys():
            if modality == "text":
                continue
            prefix = modal_special_str[modality]["prefix"]
            start = modal_special_str[modality]["sos"]
            end = modal_special_str[modality]["eos"]
            modality_vocab_size = modal_special_str[modality]["vocab_size"]
            if start not in self.tokenizer.get_vocab():
                tokens = [f"<{prefix}{x}>" for x in range(modality_vocab_size)] + [start, end]
                self.tokenizer.add_tokens(tokens)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well
        self.add_custom_tokens()

    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        if str(self.data_path).startswith("s3://"):
            print(f"The OpenWebText data path points to an S3 location: {self.data_path}. Skipping preprocessing.")
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"Found OpenWebText train and val dir: {self.data_path}. Skipping preprocessing.")
            return

        dataset = CustomDataset(self.data_path)
        # 计算训练集和验证集的大小
        val_size = int(self.val_split_fraction * len(dataset))
        train_size = len(dataset) - val_size
        # 随机分割数据集
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        split_dataset = {"train": train_dataset, "val": val_dataset}

        def tokenize(data: Dataset, index: int):
            prompt = task_prompt % (task2tokens["text-video"], data[index]["text"], "", "", "", "")
            yield self.tokenizer.encode(prompt, eos=False)

        optimize(
            fn=partial(tokenize, split_dataset["train"]),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=(os.cpu_count() - 1),
            chunk_bytes="200MB",
        )
        optimize(
            fn=partial(tokenize, split_dataset["val"]),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=(os.cpu_count() - 1),
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
