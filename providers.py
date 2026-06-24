"""
Unified call_model() — Gemini only for now.
To add OpenAI or Anthropic later, add their if-blocks here and nowhere else.
"""
import os
from google import genai

_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def call_model(provider: str, model: str, messages: list[dict]) -> tuple[str, dict]:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns (response_text, usage) where usage has "input_tokens"/"output_tokens".
    """
    if provider == "gemini":
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        convo = [m for m in messages if m["role"] != "system"]

        # Gemini uses "user"/"model" roles, not "user"/"assistant"
        contents = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["content"]}],
            }
            for m in convo
        ]

        config = {"system_instruction": system_msg} if system_msg else None

        resp = _gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        text = resp.text
        usage = {
            "input_tokens": resp.usage_metadata.prompt_token_count,
            "output_tokens": resp.usage_metadata.candidates_token_count,
        }
        return text, usage

    raise ValueError(f"Unknown provider: {provider!r}. Currently only 'gemini' is supported.")
