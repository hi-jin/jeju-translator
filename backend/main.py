import os
import sys
from typing import Optional

sys.path.append(os.path.abspath("./model"))
sys.path.append(os.path.abspath("./model/classes"))

from flask import Flask, jsonify, request

from model.core import load_model, Translator


app = Flask(__name__)
model: Optional[Translator] = None


@app.before_first_request
def initialize():
    global model
    model = load_model("./model/checkpoints/model_checkpoint.pth")


@app.route("/translate", methods=["POST"])
def translate():
    if model is None:
        return {"error": "internal error"}, 500
    text = request.form.get("text")  # type: ignore
    if text is None:
        return {"error": "internal error"}, 500

    translated = model.generate(text)
    return jsonify({"output": translated}), 200


if __name__ == "__main__":
    app.run()
