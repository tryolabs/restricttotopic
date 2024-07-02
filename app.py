import json
import torch
import nltk
from typing import Any, Dict, List
from transformers import pipeline

class InferlessPythonModel:

    def initialize(self):
        self._classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0,
                hypothesis_template="This example has to do with topic {}.",
                multi_label=True,
            )
        
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = self._classifier(inputs["text"], inputs["candidate_topics"])
        topics = result["labels"]
        scores = result["scores"]
        found_topics = []
        for topic, score in zip(topics, scores):
            if score > self._zero_shot_threshold:
                found_topics.append(topic)
        return {"results": found_topics}
        
    def finalize(self):
        pass
    
    