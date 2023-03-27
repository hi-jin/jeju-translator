import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from classes.configs import TranslatorConfig


class Translator(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        config: TranslatorConfig,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.config = config

    def generate(self, text: str):
        inputs: torch.Tensor = self.tokenizer.encode(text, return_tensors="pt")  # type: ignore
        outputs = self.pretrained_model.generate(inputs)[0]
        generated = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return generated

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
