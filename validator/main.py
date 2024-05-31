import contextvars
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests
from guardrails.utils.casting_utils import to_int
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import pipeline


@register_validator(name="tryolabs/restricttotopic", data_type="string")
class RestrictToTopic(Validator):
    """Checks if text's main topic is specified within a list of valid topics
    and ensures that the text is not about any of the invalid topics.

    This validator accepts at least one valid topic and an optional list of
    invalid topics.

    Default behavior first runs a Zero-Shot model, and then falls back to
    ask OpenAI's `gpt-3.5-turbo` if the Zero-Shot model is not confident
    in the topic classification (score < 0.5).

    In our experiments this LLM fallback increases accuracy by 15% but also
    increases latency (more than doubles the latency in the worst case).

    Both the Zero-Shot classification and the GPT classification may be toggled.

    **Key Properties**

    | Property                      | Description                              |
    | ----------------------------- | ---------------------------------------- |
    | Name for `format` attribute   | `tryolabs/restricttotopic`               |
    | Supported data types          | `string`                                 |
    | Programmatic fix              | Removes lines with off-topic information |

    Args:
        valid_topics (List[str]): topics that the text should be about
            (one or many).
        invalid_topics (List[str], Optional, defaults to []): topics that the
            text cannot be about.
        device (int, Optional, defaults to -1): Device ordinal for CPU/GPU
            supports for Zero-Shot classifier. Setting this to -1 will leverage
            CPU, a positive will run the Zero-Shot model on the associated CUDA
            device id.
        model (str, Optional, defaults to 'facebook/bart-large-mnli'): The
            Zero-Shot model that will be used to classify the topic. See a
            list of all models here:
            https://huggingface.co/models?pipeline_tag=zero-shot-classification
        llm_callable (Union[str, Callable, None], Optional, defaults to
            'gpt-3.5-turbo'): Either the name of the OpenAI model, or a callable
            that takes a prompt and returns a response.
        disable_classifier (bool, Optional, defaults to False): controls whether
            to use the Zero-Shot model. At least one of disable_classifier and
            disable_llm must be False.
        classifier_api_endpoint (str, Optional, defaults to None): An API endpoint
            to recieve post requests that will be used when provided. If not provided, a
            local model will be initialized.
        disable_llm (bool, Optional, defaults to False): controls whether to use
            the LLM fallback. At least one of disable_classifier and
            disable_llm must be False.
        zero_shot_threshold (float, Optional, defaults to 0.5): The threshold used to
            determine whether to accept a topic from the Zero-Shot model. Must be
            a number between 0 and 1.
        llm_threshold (int, Optional, defaults to 3): The threshold used to determine
        if a topic exists based on the provided llm api. Must be between 0 and 5.
    """

    def __init__(
        self,
        valid_topics: List[str],
        invalid_topics: Optional[List[str]] = [],
        device: Optional[Union[str, int]] = -1,
        model: Optional[str] = "facebook/bart-large-mnli",
        llm_callable: Union[str, Callable, None] = None,
        disable_classifier: Optional[bool] = False,
        classifier_api_endpoint: Optional[str] = None,
        disable_llm: Optional[bool] = False,
        on_fail: Optional[Callable[..., Any]] = None,
        zero_shot_threshold: Optional[float] = 0.5,
        llm_theshold: Optional[int] = 3,
    ):
        super().__init__(
            valid_topics=valid_topics,
            invalid_topics=invalid_topics,
            device=device,
            model=model,
            disable_classifier=disable_classifier,
            classifier_api_endpoint=classifier_api_endpoint,
            disable_llm=disable_llm,
            llm_callable=llm_callable,
            on_fail=on_fail,
            zero_shot_threshold=zero_shot_threshold,
            llm_theshold=llm_theshold,
        )
        self.accumulation = ""
        self._valid_topics = valid_topics

        if invalid_topics is None:
            self._invalid_topics = []
        else:
            self._invalid_topics = invalid_topics

        self._device = (
            str(device).lower()
            if str(device).lower() in ["cpu", "mps"]
            else int(device)
        )
        self._model = model
        self._disable_classifier = disable_classifier
        self._disable_llm = disable_llm
        self._classifier_api_endpoint = classifier_api_endpoint

        self._zero_shot_threshold = zero_shot_threshold
        if self._zero_shot_threshold < 0 or self._zero_shot_threshold > 1:
            raise ValueError("zero_shot_threshold must be a number between 0 and 1")

        self._llm_threshold = llm_theshold
        if self._llm_threshold < 0 or self._llm_threshold > 5:
            raise ValueError("llm_threshold must be a number between 0 and 5")
        self.set_callable(llm_callable)

        if self._classifier_api_endpoint is None:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self._model,
                device=self._device,
                hypothesis_template="This example has to do with topic {}.",
                multi_label=True,
            )
        else:
            self._classifier = self._inference_endpoint

        self._json_schema, self._tools = self._create_json_schema(
            self._valid_topics, self._invalid_topics
        )

    def _create_json_schema(self, valid_topics: list, invalid_topics: list) -> str:
        """Creates a json schema that an LLM will fill out. The json schema contains
        one of each of the provided topics, as well as a blank 'present' and 'confidence'
        for the llm to fill in.

        Args:
            valid_topics (list): The valid topics to provide as one of the json schema
            invalid_topics (list): Invalid topics to provide as one of the json schema

        Returns:
            str: The resulting json schema with unfilled data types
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "is_topic_relevant",
                    "description": "Determine if the provided text is about a topic, with a confidence score.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Simply the repeated name of the given topic.",
                            },
                            "present": {
                                "type": "boolean",
                                "description": "If the given topic is discussed in the given text.",
                            },
                            "confidence": {
                                "type": "integer",
                                "description": "The confidence level of the topic being present in the text, from 1-5",
                            },
                        },
                        "required": ["name", "present", "confidence"],
                    },
                },
            },
        ]

        json_schema = []
        for topic in set(valid_topics + invalid_topics):
            json_schema.append({"topic": topic})
        return json_schema, tools

    def get_topics_ensemble(self, text: str, candidate_topics: List[str]) -> list[str]:
        """Finds the topics in the input text based on if it is determined by the zero
        shot model or the llm.

        Args:
            text (str): The input text to find categories from
            candidate_topics (List[str]): The topics to search for in the input text

        Returns:
            list[str]: The found topics
        """
        # Find topics based on zero shot model
        zero_shot_topics = self.get_topics_zero_shot(text, candidate_topics)

        # Find topics based on llm
        llm_topics = self.get_topics_llm(text, candidate_topics)

        return list(set(zero_shot_topics + llm_topics))

    def get_topics_llm(self, text: str, candidate_topics: List[str]) -> list[str]:
        """Returns a list of the topics identified in the given text using an LLM
        callable

        Args:
            text (str): The input text to classify topics.
            candidate_topics (List[str]): The topics to identify if present in the text.

        Returns:
            list[str]: The topics found in the input text.
        """
        topics = self.call_llm(text)
        found_topics = []
        for llm_result in topics:
            if llm_result["present"] and llm_result["confidence"] > self._llm_threshold:
                # Verify the llm didn't hallucinate a topic.
                if llm_result["name"] in candidate_topics:
                    found_topics.append(llm_result["name"])
        return found_topics

    def get_client_args(self) -> Tuple[Optional[str], Optional[str]]:
        """Returns neccessary data for api calls.

        Returns:
            Tuple[Optional[str], Optional[str]]: api key and api base values
        """
        kwargs = {}
        context_copy = contextvars.copy_context()
        for key, context_var in context_copy.items():
            if key.name == "kwargs" and isinstance(kwargs, dict):
                kwargs = context_var
                break

        api_key = kwargs.get("api_key")
        api_base = kwargs.get("api_base")

        return (api_key, api_base)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def call_llm(self, text: str) -> str:
        """Call the LLM with the given prompt.

        Expects a function that takes a string and returns a string.
        Args:
            text (str): The input text to classify using the LLM.
            topics (List[str]): The list of candidate topics.
        Returns:
            response (str): String representing the LLM response.
        """
        return self._llm_callable(text)

    def set_callable(self, llm_callable: Union[str, Callable, None]) -> None:
        """Set the LLM callable.

        Args:
            llm_callable: Either the name of the OpenAI model, or a callable that takes
                a prompt and returns a response.
        """

        if llm_callable is None:
            llm_callable = "gpt-3.5-turbo"

        if isinstance(llm_callable, str):
            if llm_callable not in ["gpt-3.5-turbo", "gpt-4"]:
                raise ValueError(
                    "llm_callable must be one of 'gpt-3.5-turbo' or 'gpt-4'."
                    "If you want to use a custom LLM, please provide a callable."
                    "Check out ProvenanceV1 documentation for an example."
                )

            def openai_callable(text: str) -> str:
                api_key, api_base = self.get_client_args()
                client = OpenAI()
                response = client.chat.completions.create(
                    model=llm_callable,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Given a series of topics, determine if the topic is present in the provided text. Return the result as json.
                            
                            Text
                            ----
                            {text}

                            Schema
                            ------
                            {self._json_schema}

                            Complete Schema
                            ---------------

                            """,
                        },
                    ],
                    tools=self._tools,
                )
                tool_calls = []
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append(json.loads(tool_call.function.arguments))
                return tool_calls

            self._llm_callable = openai_callable
        elif isinstance(llm_callable, Callable):
            self._llm_callable = llm_callable
        else:
            raise ValueError("llm_callable must be a string or a Callable")

    def get_topics_zero_shot(self, text: str, candidate_topics: List[str]) -> list[str]:
        """Gets the topics found through the zero shot classifier

        Args:
            text (str): The text to classify
            candidate_topics (List[str]): The potential topics to look for

        Returns:
            list[str]: The resulting topics found that meet the given threshold
        """
        result = self._classifier(text, candidate_topics)
        topics = result["labels"]
        scores = result["scores"]
        found_topics = []
        for topic, score in zip(topics, scores):
            if score > self._zero_shot_threshold:
                found_topics.append(topic)
        return found_topics

    def validate(
        self, value: str, metadata: Optional[Dict[str, Any]] = {}
    ) -> ValidationResult:
        """Validates that a string contains at least one valid topic and no invalid topics.

        Args:
            value (str): The given string to classify
            metadata (Optional[Dict[str, Any]], optional): _description_. Defaults to {}.

        Raises:
            ValueError: If a topic is invalid and valid
            ValueError: If no valid topics are set
            ValueError: If there is no llm or zero shot classifier set

        Returns:
            ValidationResult: PassResult if a topic is restricted and valid,
            FailResult otherwise
        """
        valid_topics = set(self._valid_topics)
        invalid_topics = set(self._invalid_topics)
        all_topics = list(set(valid_topics) | set(invalid_topics))

        # throw if valid and invalid topics are empty
        if not valid_topics:
            raise ValueError(
                "`valid_topics` must be set and contain at least one topic."
            )

        # throw if valid and invalid topics are not disjoint
        if bool(valid_topics.intersection(invalid_topics)):
            raise ValueError("A topic cannot be valid and invalid at the same time.")

        # Ensemble method
        if not self._disable_classifier and not self._disable_llm:
            found_topics = self.get_topics_ensemble(value, all_topics)
        # LLM Classifier Only
        elif self._disable_classifier and not self._disable_llm:
            found_topics = self.get_topics_llm(value, all_topics)
        # Zero Shot Classifier Only
        elif not self._disable_classifier and self._disable_llm:
            found_topics = self.get_topics_zero_shot(value, all_topics)
        else:
            raise ValueError("Either classifier or llm must be enabled.")

        # Determine if valid or invalid topics were found
        invalid_topics_found = []
        valid_topics_found = []
        for topic in found_topics:
            if topic in self._valid_topics:
                valid_topics_found.append(topic)
            elif topic in self._invalid_topics:
                invalid_topics_found.append(topic)

        # Require at least one valid topic and no invalid topics
        if invalid_topics_found:
            return FailResult(
                error_message=f"Invalid topics found: {invalid_topics_found}"
            )
        if not valid_topics_found:
            return FailResult(error_message="No valid topic was found.")

        return PassResult()

    def validate_stream(
        self, value: str, metadata: Optional[Dict[str, Any]] = {}
    ) -> ValidationResult:
        self.accumulation += value

        if self._is_sentence(self.accumulation):
            result = self.validate(self.accumulation)
            self.accumulation = ""
            return result

        return PassResult()

    def _inference_endpoint(self, text: str, candidate_topics: list[str]) -> str:
        data = {
            "inputs": [
                {
                    "data": [text],
                    "name": "text",
                    "shape": [1],
                    "datatype": "BYTES",
                },
                {
                    "data": candidate_topics,
                    "name": "candidate_labels",
                    "shape": [len(candidate_topics)],
                    "datatype": "BYTES",
                },
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 84d51dc484078596e1d023a8dd55dbff5e9c7cb7d861dedb5a72cf0d85a2a514c6d329ad789f2353e9f08cb682f51a8233fdf4c85ccf719b7c2f78f9377ac3cb",
        }
        results = requests.post(
            url=self._classifier_api_endpoint,
            headers=headers,
            data=json.dumps(data),
        )
        results = results.json()
        results = results["outputs"][0]["data"][0]
        results = results.replace(" b", " ")
        results = results.replace("[b", "[")
        results = results.replace("'", '"')
        results = json.loads(results)

        topics = [str(label) for label in results["labels"]]
        scores = results["scores"]
        return {"labels": topics, "scores": scores}

    def _is_sentence(self, value: str) -> bool:
        return value.endswith((".", "?", "!"))
