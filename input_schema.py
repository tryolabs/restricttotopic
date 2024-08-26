INPUT_SCHEMA = {
    "text": {
        "example": ["Let's talk about Iphones made by Apple"],
        "shape": [1],
        "datatype": "STRING",
        "required": True,
    },
    "candidate_topics": {
        "example": ["Apple Iphone", "Samsung Galaxy"],
        "shape": [-1],
        "datatype": "STRING",
        "required": True,
    },
    "zero_shot_threshold": {
        "example": [0.5],
        "shape": [1],
        "datatype": "FP32",
        "required": True,
    },
}
