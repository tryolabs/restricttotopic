import os

from dotenv import load_dotenv
from guardrails import Guard
from pydantic import BaseModel, Field
from validator import RestrictToTopic

load_dotenv()


class ValidatorTestObject(BaseModel):
    test_val: str = Field(
        validators=[
            RestrictToTopic(
                valid_topics=["sports"],
                invalid_topics=["music"],
                disable_classifier=True,
                disable_llm=False,
                on_fail="exception",
            )
        ],
        api_key=os.getenv("OPENAI_API_KEY"),
    )


TEST_OUTPUT = """
{
  "test_val": "In Super Bowl LVII in 2023, the Chiefs clashed with the Philadelphia Eagles in a fiercely contested battle, ultimately emerging victorious with a score of 38-35."
}
"""


guard = Guard.from_pydantic(output_class=ValidatorTestObject)

try:
    guard.parse(TEST_OUTPUT)
    print("Successfully passed validation when it was supposed to.")
except Exception:
    print("Failed to pass validation when it was supposed to.")


TEST_FAIL_OUTPUT = """
{
"test_val": "The Beatles were a charismatic English pop-rock band of the 1960s."
}
"""

try:
    guard.parse(TEST_FAIL_OUTPUT)
    print("Failed to fail validation when it was supposed to")
except Exception:
    print("Successfully failed validation when it was supposed to.")
