import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

from classes.configs import DataConfig


class DialectDataset(Dataset):
    def __init__(
        self,
        standard_form_texts,
        dialect_form_texts,
    ):
        super().__init__()
        self.standard_form_texts = standard_form_texts
        self.dialect_form_texts = dialect_form_texts

    def __len__(self):
        return len(self.standard_form_texts)

    def __getitem__(self, idx: int):
        standard_form_text = self.standard_form_texts[idx]
        dialect_form_text = self.dialect_form_texts[idx]

        return standard_form_text, dialect_form_text


class DialectDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        config: DataConfig,
        tokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        train_data = pd.read_csv(os.path.join(self.config["data_dir"], "train.csv"))
        test_data = pd.read_csv(os.path.join(self.config["data_dir"], "validation.csv"))

        dataset = DialectDataset(test_data["standard_form_texts"], test_data["dialect_form_texts"])
        size = len(dataset)
        tr_size = int(0.8 * size)
        val_size = (size - tr_size) // 2
        test_size = size - tr_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [tr_size, val_size, test_size])

    def collate_fn(self, batch):
        standard_form_texts, dialect_form_texts = zip(*batch)
        standard_form_tokens = self.tokenizer(standard_form_texts, padding=True, return_tensors="pt")
        dialect_form_tokens = self.tokenizer(dialect_form_texts, padding=True, return_tensors="pt")

        return {
            "input_ids": standard_form_tokens["input_ids"],
            "mask": standard_form_tokens["attention_mask"],
            "labels": dialect_form_tokens["input_ids"],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
        )
