import os
from typing import TypedDict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5ForConditionalGeneration, AutoTokenizer


class TranslatorConfig(TypedDict):
    pretrained_model_name: str
    data_dir: str
    batch_size: int
    num_workers: int
    lr: float


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
        config: TranslatorConfig,
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


class Translator(pl.LightningModule):
    def __init__(
        self,
        pretrained_model,
        config: TranslatorConfig,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = config

    def forward(self, input_ids, mask=None, labels=None):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=mask, labels=labels)
        return {
            "logits": outputs.logits,
            "loss": outputs.loss,
        }

    def step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        mask = batch["mask"]
        labels = batch["labels"]

        outputs = self.forward(input_ids, mask, labels)

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx)
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx)
        self.log("val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self.step(batch, batch_idx)
        self.log("test_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config["lr"])


def main():
    pl.seed_everything(1234)

    config = TranslatorConfig(
        pretrained_model_name="paust/pko-t5-base",
        data_dir="./dataset",
        batch_size=128,
        num_workers=2,
        lr=0.0001,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])
    pretrained_model = T5ForConditionalGeneration.from_pretrained(config["pretrained_model_name"])

    datamodule = DialectDatamodule(config, tokenizer)
    model = Translator.load_from_checkpoint(
        checkpoint_path="./checkpoints/epoch=2-step=19848.ckpt",
        pretrained_model=pretrained_model,
        config=config,
    )

    datamodule.setup()
    batch = next(iter(datamodule.test_dataloader()))
    outputs = model.forward(
        input_ids=batch["input_ids"],
        mask=batch["mask"],
        labels=batch["labels"],
    )
    logits = outputs["logits"]
    predicts = tokenizer.batch_decode(logits.argmax(dim=-1), skip_special_tokens=True)
    inputs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

    pd.DataFrame(
        {
            "predicts": predicts,
            "inputs": inputs,
            "labels": labels,
        }
    ).to_csv("result.csv")


if __name__ == "__main__":
    main()
