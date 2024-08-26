from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from transformers import pipeline
import torch
import os

app = FastAPI()

env = os.environ.get("env", "dev")
torch_device = "cuda" if env == "prod" else "cpu"

# Initialize the zero-shot classification pipeline
classifier = pipeline(
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
    text: str
    candidate_topics: List[str]
    zero_shot_threshold: float = 0.5

class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]

@app.post("/validate", response_model=OutputResponse)
def restrict_to_topic(input_request: InputRequest):
    print('make request')
    
    text = input_request.text
    candidate_topics = input_request.candidate_topics
    zero_shot_threshold = input_request.zero_shot_threshold
    
    
    if text is None or candidate_topics is None:
        raise HTTPException(status_code=400, detail="Invalid input format")
    
    # Perform zero-shot classification
    result = classifier(text, candidate_topics)
    print("result: ", result)
    topics = result["labels"]
    scores = result["scores"]
    found_topics = [topic for topic, score in zip(topics, scores) if score > zero_shot_threshold]
    
    if not found_topics:
        found_topics = ["No valid topic found."]
    
    output_data = OutputResponse(
        modelname="RestrictToTopicModel",
        modelversion="1",
        outputs=[
            InferenceData(
                name="results",
                datatype="BYTES",
                shape=[len(found_topics)],
                data=found_topics
            )
        ]
    )
    
    return output_data

# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
