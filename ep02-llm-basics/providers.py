"""One small function per provider: same inputs (instructions + message) -> the reply text.

Pick whichever you like; they do the same job. Each reads its API key from an environment variable, so keys never
appear in the code:
    gemini     pip install google-genai   GEMINI_API_KEY     free tier available (aistudio.google.com)
    openai     pip install openai         OPENAI_API_KEY     paid (platform.openai.com)
    anthropic  pip install anthropic      ANTHROPIC_API_KEY  paid (platform.claude.com)

No temperature here: the newest models from all three providers tune it themselves (some reject the setting, some
ignore it). To turn the dial yourself, see 03_temperature.py, which runs an open model on your laptop.

Model names change often. These were current in November 2026; if one stops working, check the provider's models
page and change the name (or pass --model to 04_five_levers.py).
"""

MODELS = {
    "gemini": "gemini-3.5-flash-lite", # Google's fast, low-cost model; on the free tier (also try gemini-4-hydrogen)
    "openai": "gpt-6-luna",            # OpenAI's small, low-cost model
    "anthropic": "claude-haiku-5-5",   # Anthropic's small, fast model
}


def gemini(instructions, message, model=MODELS["gemini"]):
    from google import genai
    client = genai.Client()                       # reads GEMINI_API_KEY
    reply = client.interactions.create(
        model=model,
        system_instruction=instructions,          # role + rules
        input=message,                            # facts + examples + the email
    )
    return reply.output_text


def openai(instructions, message, model=MODELS["openai"]):
    from openai import OpenAI
    client = OpenAI()                             # reads OPENAI_API_KEY
    reply = client.responses.create(
        model=model,
        instructions=instructions,                # role + rules
        input=message,                            # facts + examples + the email
    )
    return reply.output_text


def anthropic(instructions, message, model=MODELS["anthropic"]):
    import anthropic as sdk
    client = sdk.Anthropic()                      # reads ANTHROPIC_API_KEY
    reply = client.messages.create(
        model=model,
        max_tokens=1000,
        system=instructions,                      # role + rules
        messages=[{"role": "user", "content": message}],   # facts + examples + the email
    )
    return "".join(block.text for block in reply.content if block.type == "text")


PROVIDERS = {"gemini": gemini, "openai": openai, "anthropic": anthropic}


def ask(provider, instructions, message, model=None):
    fn = PROVIDERS[provider]
    try:
        return fn(instructions, message, model) if model else fn(instructions, message)
    except ImportError:
        raise SystemExit(f"The {provider} library isn't installed. Run: pip install -r requirements.txt")
    except Exception as e:                        # keep the common problems short and readable
        msg = str(e)
        if any(s in msg for s in ("503", "529", "high demand", "overloaded", "UNAVAILABLE")):
            hint = "The model is busy right now. Wait a minute and try again, or pick another with --model."
        elif any(s in msg for s in ("429", "quota", "rate limit", "RESOURCE_EXHAUSTED")):
            hint = "Rate limit or quota reached (free tiers have limits). Wait a bit and try again."
        elif any(s in msg for s in ("401", "403", "API key", "api_key", "authentication", "PERMISSION_DENIED")):
            hint = f"The API key was missing or rejected. Check the environment variable for {provider} (see README)."
        elif any(s in msg for s in ("404", "not found", "NOT_FOUND")):
            hint = "That model name wasn't found. Model names change: check the provider's models page, use --model."
        else:
            raise
        raise SystemExit(f"\n[{provider}] {hint}\n(details: {msg[:200]})")
