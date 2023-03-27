from typing import TypedDict


class TranslatorConfig(TypedDict):
    pretrained_model_name: str
    lr: float


class DataConfig(TypedDict):
    data_dir: str
    batch_size: int
    num_workers: int
