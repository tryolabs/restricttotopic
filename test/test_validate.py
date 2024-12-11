
from guardrails.validators import PassResult, FailResult

from validator import RestrictToTopic


def test_valid_topics():
    v = RestrictToTopic(valid_topics=["food", "travel"], on_fail="noop")
    assert isinstance(v.validate("I've always wanted to visit Japan."), PassResult)
    assert isinstance(v.validate("My hobbies include cooking and baking."), PassResult)
    assert isinstance(v.validate("Banach-Tarski is an anagram of Banach-Tarski-Banach-Tarski."), FailResult)


def test_invalid_topics():
    v = RestrictToTopic(invalid_topics=["food", "travel"], on_fail="noop")
    assert isinstance(v.validate("I used to tell dad jokes. Sometimes he would laugh."), PassResult)
    assert isinstance(v.validate("The 'B' in 'Benoit B. Mandelbrot' is short for 'Benoit B. Mandelbrot'."), PassResult)
    assert isinstance(v.validate("What did you have for lunch?"), FailResult)
    assert isinstance(v.validate("I live in Spain, but the 's' is silent."), FailResult)


def test_valid_and_invalid_topics():
    v = RestrictToTopic(
        valid_topics=["food",],
        invalid_topics=["travel",],
        on_fail="noop",
    )
    assert isinstance(v.validate("I enjoy eating fish."), PassResult)
    assert isinstance(v.validate("I went to Japan and had sushi."), FailResult)


def test_metadata_override():
    v = RestrictToTopic(
        valid_topics=["food",],
        on_fail="noop",
    )
    assert isinstance(v.validate("I went to Japan and had sushi."), PassResult)
    assert isinstance(
        v.validate(
            "I went to Japan and had sushi.",
            metadata={"invalid_topics": ["travel",]}
        ),
        FailResult
    )
