"""Decode declared answer fields without penalizing surrounding explanation.

This does not repair malformed JSON, choose among conflicting blocks, or infer
business success. Independent tool/state evidence is still required by oracles.
"""
import json
import re


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate answer key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Non-finite JSON answer")


def _loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def extract_object(answer):
    text = str(answer).strip()
    try:
        value = _loads(text)
    except (ValueError, TypeError):
        fences = list(re.finditer(r"```([^\n`]*)\n(.*?)```", text, flags=re.DOTALL))
        if fences:
            if len(fences) != 1 or fences[0][1].strip().lower() not in {"", "json"}:
                return {}
            outside = text[:fences[0].start()] + text[fences[0].end():]
            if "{" in outside or "}" in outside:
                return {}
            candidate = fences[0][2]
        else:
            # Parse the entire brace envelope, never scan inner objects after
            # a failed decode. Multiple objects or a truncated outer object fail.
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < start or "```" in text:
                return {}
            candidate = text[start:end + 1]
        try:
            value = _loads(candidate)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}
