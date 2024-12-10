import pytest

from validator import RestrictToTopic


def test_init_without_topics():
    """Make sure that if no topics are specified that the validator will fail."""
    _ = RestrictToTopic(valid_topics=["a", "b"])
    _ = RestrictToTopic(invalid_topics=["a", "b"])
    _ = RestrictToTopic(["a", "b"], ["c", "d"])
    with pytest.raises(Exception) as e:
        _ = RestrictToTopic(valid_topics=[], invalid_topics=[])
        assert str(e.value) == "Either valid topics or invalid topics must be specified."
    with pytest.raises(Exception) as e:
        _ = RestrictToTopic()
        assert str(e.value) == "Either valid topics or invalid topics must be specified."
