from typing import Optional

import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from classes.configs import TranslatorConfig, DataConfig
from classes.translator import Translator
from classes.dialect_datamodule import DialectDatamodule


__global_seed: Optional[int] = None
tokenizer: Optional[T5TokenizerFast] = None
model: Optional[Translator] = None
datamodule: Optional[DialectDatamodule] = None


def initialize():
    if __global_seed is None:
        configure_seeds()
    if tokenizer is None:
        load_tokenizer()
    if model is None:
        load_model()
    if datamodule is None:
        load_datamodule()


def configure_seeds(seed: int = 1234):
    global __global_seed

    pl.seed_everything(seed)
    __global_seed = seed


def load_model(state_dict_path: str = "./checkpoints/model_checkpoint.pth"):
    global model

    if model is not None:
        return model

    translator_config = TranslatorConfig(
        pretrained_model_name="paust/pko-t5-base",
        lr=0.0001,
    )

    pretrained_model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        translator_config["pretrained_model_name"]
    )  # type: ignore

    if tokenizer is None:
        load_tokenizer()
        assert tokenizer is not None

    model = Translator(
        pretrained_model=pretrained_model,
        tokenizer=tokenizer,
        config=translator_config,
    )

    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path))

    return model


def load_tokenizer():
    global tokenizer

    if tokenizer is not None:
        return tokenizer

    translator_config = TranslatorConfig(
        pretrained_model_name="paust/pko-t5-base",
        lr=0.0001,
    )

    tokenizer = T5TokenizerFast.from_pretrained(translator_config["pretrained_model_name"])
    return tokenizer


def load_datamodule(
    data_dir="./dataset",
    batch_size=8,
    num_workers=0,
    refresh=False,
):
    global datamodule

    if refresh == False and datamodule is not None:
        return datamodule

    data_config = DataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if tokenizer is None:
        load_tokenizer()

    datamodule = DialectDatamodule(config=data_config, tokenizer=tokenizer)
    return datamodule
