from guardrails import Guard
from RestrictToTopic import RestrictToTopic

guard = Guard().use(
    RestrictToTopic(
        valid_topics="sports",
        use_local=False,
        validation_endpoint="http://ec2-44-213-73-134.compute-1.amazonaws.com/restrict_to_topic",
    )
)

print(guard.validate("Globally, football is the same as American soccer"))