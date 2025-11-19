from __future__ import annotations

import re

_NON_DIGIT = re.compile(r"[^0-9\-]")


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip().replace("\n", " ")
    text = text.replace("Answer:", "").replace("answer:", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_numeric(text: str) -> str:
    stripped = _NON_DIGIT.sub("", text)
    return stripped.lstrip("0") or stripped


def answers_match(pred: str, target: str) -> bool:
    if not target:
        return False
    if pred == target:
        return True
    pred_num = normalize_numeric(pred)
    target_num = normalize_numeric(target)
    if pred_num and target_num and pred_num == target_num:
        return True
    pred_first = pred.split(" ")[0]
    target_first = target.split(" ")[0]
    return pred_first == target_first

