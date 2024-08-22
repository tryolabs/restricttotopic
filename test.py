# Import Guard and Validator
from guardrails.hub import RelevancyEvaluator
from guardrails import Guard

# Setup Guard
guard = Guard().use(
    RelevancyEvaluator(llm_callable="gpt-3.5-turbo")
)

# Example values
value = {
    "original_prompt": "What is the capital of France?",
    "reference_text": "The capital of France is Paris."
}

guard.validate(metadata=value)  # Validator passes if the text is relevant