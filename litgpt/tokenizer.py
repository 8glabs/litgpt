# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os.path
from pathlib import Path
from typing import Optional, Union

import torch

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
modal_specials = [start_of_seq, end_of_seq, start_of_text, end_of_text, source_of_video, resolution_of_video, start_of_output_visual, end_of_output_visual, start_of_output_audio, end_of_output_audio]

text_vocab_size=32000
audio_codebook_size=1024
audio_codebook_num=16
audio_vocab_size=audio_codebook_size * audio_codebook_num
visual_vocab_size=2**18

Task2tokens = {
    "text-video": f"{task_prefix}0",
    "image-vodeo": f"{task_prefix}1",
    "unconditional video": f"{task_prefix}2",
    "Audioavatar talkingheads": f"{task_prefix}3",
    "Audioavatar singingheads": f"{task_prefix}4",
    "text-image": f"{task_prefix}5",
}

# task_id, text, input_visual, input_audio, output_visual, output_audio
Task2prompt = f"{start_of_seq}%s{start_of_text}%s{end_of_text}{start_of_input_visual}%s{end_of_input_visual}{start_of_input_audio}%s{end_of_input_audio}{source_of_video}{resolution_of_video}{start_of_output_visual}%s{end_of_output_visual}{start_of_output_audio}%s{end_of_output_audio}{end_of_seq}"

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

class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # some checkpoints have both files, `.model` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from transformers import LlamaTokenizer
            self.processor = LlamaTokenizer.from_pretrained(
                checkpoint_dir,
                model_max_length=4096,
            )
            self.processor.pad_token = self.processor.eos_token
            self.processor.padding_side = "right"
            self.bos_id = self.processor.bos_token_id
            self.eos_id = self.processor.eos_token_id
            self.backend = "transformers"
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                self.bos_id = self.token_to_id(bos_token["content"]) if bos_token is not None else None
                eos_token = config.get("eos_token")
                self.eos_id = self.token_to_id(eos_token["content"]) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError
        self.checkpoint_dir = checkpoint_dir
        self.add_custom_tokens()

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def vocab_size(self, with_added_tokens=False) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=with_added_tokens)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def get_vocab(self):
        if self.backend == "huggingface":
            return self.processor.get_vocab()
        if self.backend == "sentencepiece":
            pieces = [self.processor.id_to_piece(id) for id in range(self.processor.get_piece_size())]
            return pieces
        raise RuntimeError

    def add_tokens(self, tokens):
        if self.backend == "huggingface":
            self.processor.add_tokens(tokens)
        elif self.backend == "sentencepiece":
            raise NotImplementedError
        else:
            raise RuntimeError

    def add_custom_tokens(self) -> None:
        if "-addtokens" in self.checkpoint_dir.name:
            return
        for modal_special in modal_specials:
            if modal_special not in self.processor.get_vocab():
                self.add_tokens([modal_special])
        for modality in modal_special_str.keys():
            if modality == "text":
                continue
            prefix = modal_special_str[modality]["prefix"]
            start = modal_special_str[modality]["sos"]
            end = modal_special_str[modality]["eos"]
            modality_vocab_size = modal_special_str[modality]["vocab_size"]
            if start not in self.processor.get_vocab():
                tokens = [f"<{prefix}{x}>" for x in range(modality_vocab_size)] + [start, end]
                self.add_tokens(tokens)
        if not os.path.exists(str(self.checkpoint_dir)+"-addtokens"):
            os.makedirs(str(self.checkpoint_dir)+"-addtokens", exist_ok=True)
        self.processor.save_pretrained(str(self.checkpoint_dir)+"-addtokens")

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece" or self.backend == "transformers":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError
        if bos or (bos is None and self.use_bos):
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined a bos token")
            tokens = [bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
