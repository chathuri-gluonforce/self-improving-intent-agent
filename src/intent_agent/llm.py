from openai import OpenAI

from intent_agent.config import get_settings

_settings = get_settings()
client = OpenAI(
    base_url=_settings.openrouter_base_url,
    api_key=_settings.openrouter_api_key.get_secret_value(),
)


def ask(prompt: str, temperature: float = 0) -> str:
    r = client.chat.completions.create(
        model=_settings.openrouter_model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content.strip()
