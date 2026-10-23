"""OpenAI wrapper for text and structured JSON generation."""

import json
from openai import OpenAI

from src.config import LLM_MODEL, OPENAI_API_KEY


def llm_available() -> bool:
    return bool(OPENAI_API_KEY)


def ask(
    prompt: str,
    system: str = "",
    max_tokens: int = 800,
    json_mode: bool = False,
) -> str:

    if not llm_available():
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    kwargs = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system or "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_completion_tokens": max_tokens,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content

    if not content:
        raise ValueError("OpenAI returned an empty response.")

    return content


def ask_json(
    prompt: str,
    system: str = "",
    max_tokens: int = 800,
) -> dict:

    instructions = (
        system
        + "\nReturn exactly one valid JSON object. "
        + "Do not include markdown or explanations."
    )

    result = ask(
        prompt=prompt,
        system=instructions,
        max_tokens=max_tokens,
        json_mode=True,
    )

    parsed = json.loads(result)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from OpenAI.")

    return parsed