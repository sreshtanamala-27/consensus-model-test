"""
models.py

One function per model, all with the same signature:

    call_x(post_text: str) -> dict:
        {
            "model": "openai" | "deepseek" | "claude" | "qwen",
            "raw_response": "<exact text the model returned, or None on error>",
            "error": "<error message, or None if the call succeeded>",
        }

Every model receives the same initial prompt and is given no definition of
what "Low / Medium / High Risk" means. Debate prompts anonymize responses
from other models so provider names are not exposed during reconsideration.
"""

import os

from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """Read the following text and classify it into exactly one \
of these three categories: Low Risk, Medium Risk, High Risk.

Respond in this format:
Classification: <Low Risk / Medium Risk / High Risk>
Reasoning: <brief explanation>

Text:
\"\"\"{post}\"\"\"
"""


DEBATE_PROMPT_TEMPLATE = """Read the original text and independently reconsider \
its classification. You may keep your previous classification or change it \
after considering the other models' responses. Do not simply follow the \
majority; make your own judgment. No category definitions or external scoring \
rubric are provided.

Original text:
\"\"\"{post}\"\"\"

Your previous response:
{own_previous}

Other models' previous responses:
{other_responses}

Respond in this format:
Classification: <Low Risk / Medium Risk / High Risk>
Reasoning: <brief explanation>
"""


def _build_prompt(post: str) -> str:
    return PROMPT_TEMPLATE.format(post=post)


def _build_debate_prompt(
    post: str, own_previous_response: str | None, other_responses: list[str | None]
) -> str:
    anonymized_responses = []
    for index, response in enumerate(other_responses, start=1):
        displayed_response = response if response else "[No response available]"
        anonymized_responses.append(f"Model {index}:\n{displayed_response}")

    own_previous = own_previous_response or "[No response available]"
    return DEBATE_PROMPT_TEMPLATE.format(
        post=post,
        own_previous=own_previous,
        other_responses="\n\n".join(anonymized_responses),
    )


def _call_openai_prompt(prompt: str) -> dict:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"model": "openai", "raw_response": text, "error": None}
    except Exception as e:  # noqa: BLE001 - we want to record any failure and keep going
        return {"model": "openai", "raw_response": None, "error": str(e)}


def _call_deepseek_prompt(prompt: str) -> dict:
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"model": "deepseek", "raw_response": text, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"model": "deepseek", "raw_response": None, "error": str(e)}


def _call_claude_prompt(prompt: str) -> dict:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()
        return {"model": "claude", "raw_response": text, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"model": "claude", "raw_response": None, "error": str(e)}


def _call_qwen_prompt(prompt: str) -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    try:
        import requests

        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return {"model": "qwen", "raw_response": text, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"model": "qwen", "raw_response": None, "error": str(e)}


def call_openai(post: str) -> dict:
    return _call_openai_prompt(_build_prompt(post))


def debate_openai(
    post: str, own_previous_response: str | None, other_responses: list[str | None]
) -> dict:
    return _call_openai_prompt(
        _build_debate_prompt(post, own_previous_response, other_responses)
    )


def call_deepseek(post: str) -> dict:
    return _call_deepseek_prompt(_build_prompt(post))


def debate_deepseek(
    post: str, own_previous_response: str | None, other_responses: list[str | None]
) -> dict:
    return _call_deepseek_prompt(
        _build_debate_prompt(post, own_previous_response, other_responses)
    )


def call_claude(post: str) -> dict:
    return _call_claude_prompt(_build_prompt(post))


def debate_claude(
    post: str, own_previous_response: str | None, other_responses: list[str | None]
) -> dict:
    return _call_claude_prompt(
        _build_debate_prompt(post, own_previous_response, other_responses)
    )


def call_qwen(post: str) -> dict:
    return _call_qwen_prompt(_build_prompt(post))


def debate_qwen(
    post: str, own_previous_response: str | None, other_responses: list[str | None]
) -> dict:
    return _call_qwen_prompt(
        _build_debate_prompt(post, own_previous_response, other_responses)
    )


# Registry used by run.py so models can be selected/skipped by name from the CLI.
MODEL_FUNCS = {
    "openai": call_openai,
    "deepseek": call_deepseek,
    "claude": call_claude,
    "qwen": call_qwen,
}

DEBATE_FUNCS = {
    "openai": debate_openai,
    "deepseek": debate_deepseek,
    "claude": debate_claude,
    "qwen": debate_qwen,
}
