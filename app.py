from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from transformers import pipeline
import torch
import os

app = FastAPI()

env = os.environ.get("env", "dev")
torch_device = "cuda" if env == "prod" else "cpu"

model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=torch.device(torch_device),
    hypothesis_template="This example has to do with topic {}.",
    multi_label=True,
)


class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: Union[List[str], List[float]]
    datatype: str


class InputRequest(BaseModel):
    inputs: List[InferenceData]


class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]


@app.get("/")
async def hello_world():
    return "restrict_to_topic"


@app.post("/validate", response_model=OutputResponse)
async def restrict_to_topic(input_request: InputRequest):
    text_vals = None
    candidate_topics = None
    zero_shot_threshold = 0.5

    for inp in input_request.inputs:
        if inp.name == "text":
            text_vals = inp.data
        elif inp.name == "candidate_topics":
            candidate_topics = inp.data
        elif inp.name == "zero_shot_threshold":
            zero_shot_threshold = float(inp.data[0])

    if text_vals is None or candidate_topics is None:
        raise HTTPException(status_code=400, detail="Invalid input format")

    return RestrictToTopic.infer(text_vals, candidate_topics, zero_shot_threshold)


class RestrictToTopic:
    model_name = "facebook/bart-large-mnli"
    device = torch.device(torch_device)
    model = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        hypothesis_template="This example has to do with topic {}.",
        multi_label=True,
    )

    @staticmethod
    def infer(text_vals, candidate_topics, threshold) -> OutputResponse:
        outputs = []
        for idx, text in enumerate(text_vals):
            results = RestrictToTopic.model(text, candidate_topics)
            pred_labels = [
                label for label, score in zip(results["labels"], results["scores"]) if score > threshold
            ]

            if not pred_labels:
                pred_labels = ["No valid topic found."]

            outputs.append(
                InferenceData(
                    name=f"result{idx}",
                    datatype="BYTES",
                    shape=[len(pred_labels)],
                    data=pred_labels,
                )
            )

        output_data = OutputResponse(
            modelname="RestrictToTopicModel", modelversion="1", outputs=outputs
        )

        return output_data


# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
