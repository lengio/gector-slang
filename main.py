from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from gector_helper import load_model, process_batches

app = FastAPI()
MODEL = load_model()


class Sentence(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def read_item(sentence: Sentence):
    probabilities, predicted_labels, error_probabilities = process_batches([sentence.text], MODEL)
    return {"probabilities": probabilities[0],
            "predicted_labels": predicted_labels[0],
            "error_probabilities": error_probabilities[0]}
