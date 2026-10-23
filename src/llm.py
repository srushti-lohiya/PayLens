"""Small wrapper around the Anthropic API. Every LLM feature degrades gracefully
when ANTHROPIC_API_KEY is not set, so the rest of the project still runs."""
import json, os, re
from src.config import LLM_MODEL


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def ask(prompt: str, system: str = "", max_tokens: int = 800) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(model=LLM_MODEL, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text")


def ask_json(prompt: str, system: str = "", max_tokens: int = 800) -> dict:
    text = ask(prompt, system + "\nReply with ONE JSON object only. No markdown, no prose.", max_tokens)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])
