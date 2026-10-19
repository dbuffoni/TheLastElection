class UnclassifiableDecree(Exception):
    """S1 cannot classify the decree: it goes back to the player to be rephrased, at no cost (13.1)."""


class TextTooLong(ValueError):
    """Decrees, announcements and private messages are limited to 50 characters."""


MAX_TEXT_CHARS = 50


def validate_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """The single place that enforces the player's 50-character constraint (and non-empty text)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    t = " ".join(text.split())  # collapse whitespace; the limit applies to what is published
    if not t:
        raise ValueError("a decree or message cannot be empty")
    if len(t) > limit:
        raise TextTooLong(f"{len(t)} characters; the limit is {limit}")
    return t
