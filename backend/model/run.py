import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from core import initialize, load_model, load_datamodule


def train():
    initialize()

    model = load_model()
    datamodule = load_datamodule(num_workers=0)

    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(monitor="val_loss", save_last=True),
        ]
    )

    trainer.fit(model=model, datamodule=datamodule)


def generate():
    initialize()

    model = load_model()

    while True:
        print("표준어 입력 (quit 입력시 종료) : ", end="", flush=True)
        text = sys.stdin.readline().strip()
        if text == "quit":
            break
        print(f"번역 결과 : {model.generate(text)}")
        print()


def main():
    initialize()

    def print_full_line(character="-"):
        print(character * os.get_terminal_size().columns)

    print_full_line()
    print("1. generate (생성 테스트)\n2. train")
    print_full_line()
    print()
    print("cmd : ", end="", flush=True)

    cmd = sys.stdin.readline().strip()
    if cmd == "generate" or cmd == "1":
        generate()
    elif cmd == "train" or cmd == "2":
        train()


if __name__ == "__main__":
    main()
