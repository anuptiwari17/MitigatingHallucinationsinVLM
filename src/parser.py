import re


def parse_yes_no(text: str) -> int:
    """
    Deterministic, zero-API-cost Yes/No parser.
    Returns:
        1  -> Yes
        0  -> No
       -1  -> Unclear / cannot parse
    """
    if not text or not text.strip():
        return -1

    cleaned = text.strip().lower()
    # Remove leading punctuation, markdown bolding, or quotes
    cleaned = re.sub(r"^[\*\s\"\'`\(\[]+", "", cleaned)

    # 1. Check leading token (most reliable since our prompt asks for Yes/No first)
    first_match = re.match(r"^(yes|no)\b", cleaned)
    if first_match:
        return 1 if first_match.group(1) == "yes" else 0

    # 2. Check explicit phrasing in first sentence
    first_sentence = re.split(r"[\.\n!?]", cleaned)[0]
    has_yes = bool(re.search(r"\b(yes|correct|true|affirmative)\b", first_sentence))
    has_no = bool(re.search(r"\b(no|incorrect|false|not true|does not|cannot|isn't|aren't|wasn't|weren't)\b", first_sentence))

    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0

    return -1
