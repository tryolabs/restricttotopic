from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from transformers import pipeline
import torch

app = FastAPI()

# Initialize the zero-shot classification pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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

@app.post("/validate", response_model=OutputResponse)
def restrict_to_topic(input_request: InputRequest):
    print('make request')
    text = None
    candidate_topics = None
    zero_shot_threshold = 0.5
    
    for inp in input_request.inputs:
        if inp.name == "text":
            text = inp.data[0]
        elif inp.name == "candidate_topics":
            candidate_topics = inp.data
        elif inp.name == "zero_shot_threshold":
            zero_shot_threshold = float(inp.data[0])
    
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