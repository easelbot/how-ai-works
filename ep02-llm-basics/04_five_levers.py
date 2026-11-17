"""04 - The 5 levers, applied (How AI Works, Ep 2, chapters 4 + 6).

Nina answers support tickets for a small home-goods shop. She asks an LLM to draft a reply to a late-delivery
email, twice:
    LAZY     "Reply to this customer."  -> the generic "Dear valued customer" reply
    LEVERS   all five levers pulled:
        1 PROMPT       role + task + format             (the instructions)
        2 CONTEXT      the real order details + refund policy, so it doesn't guess
        3 EXAMPLES     two of Nina's past replies, to copy her tone   (few-shot prompting)
        4 TEMPERATURE  not set: the newest models tune it themselves (try the dial in 03_temperature.py)
        5 CHECK        "quote the policy, say I don't know if unsure" + a small code check of the reply

Run it:
    python 04_five_levers.py --dry-run                 # no key: prints exactly what would be sent
    python 04_five_levers.py                           # Gemini (default, free tier): needs GEMINI_API_KEY
    python 04_five_levers.py --provider openai         # needs OPENAI_API_KEY
    python 04_five_levers.py --provider anthropic      # needs ANTHROPIC_API_KEY
    python 04_five_levers.py --model <name>            # try another model from the same provider

The shop, order, customer and policy are made up (example data). On a free tier, don't send private data: it may be
used to improve the provider's products.
"""
import argparse
import re

from providers import MODELS, PROVIDERS, ask

# ─────────────────────────── the (made-up) facts ───────────────────────────
EMAIL = ("Hi, I ordered a blender two weeks ago (order B-3141) and it still hasn't arrived. "
         "This is really frustrating. Where is it? - Leo")

ORDER = "Order B-3141 | customer: Leo | item: blender | promised: 12 November | new delivery date: 20 November"

POLICY = ("Refund policy: if an order arrives more than 5 days late, we refund the $8 shipping fee. "
          "If it hasn't arrived within 30 days, the customer can cancel for a full refund.")

EXAMPLES = """\
Example reply 1:
Hi Priya, thanks for telling us, and sorry about the mix-up! Your lamp is on its way and arrives Friday.
I've added a note to your order so it's handled first. Anything else, just reply to this email. - Nina

Example reply 2:
Hi Sam, sorry, that's not the experience we want for you. The replacement mug ships today and should reach you
by Tuesday. You don't need to send the broken one back. - Nina"""

# ─────────────────────────── LAZY: one vague line ───────────────────────────
LAZY_INSTRUCTIONS = "Reply to this customer."
LAZY_MESSAGE = EMAIL

# ─────────────────────────── LEVERS: all five ───────────────────────────
INSTRUCTIONS = (                                                   # lever 1: PROMPT (role, task, format)
    "You are Nina, a friendly support agent for a small home-goods shop. "
    "Reply to the customer's late-delivery email. "
    "Under 100 words. Use the customer's name. End with one clear next step. "
    "Use only the facts given. Quote the refund policy. "         # lever 5: CHECK (ground it, quote it)
    "If something isn't in the facts, say you don't know instead of guessing."
)
MESSAGE = (
    f"Facts:\n{ORDER}\n{POLICY}\n\n"                              # lever 2: CONTEXT
    f"My tone (match it):\n{EXAMPLES}\n\n"                        # lever 3: EXAMPLES
    f"Customer email:\n{EMAIL}"
)


def check(reply):
    """Lever 5, in code: a few quick checks a human would do before sending."""
    text = reply.lower()
    results = [
        ("uses the customer's name (Leo)", "leo" in text),
        ("gives the real new date (20 November)", "20" in text and "nov" in text),
        ("mentions the $8 shipping refund", "$8" in text or "8 dollar" in text),
        ("under 100 words", len(reply.split()) <= 100),
    ]
    made_up = sorted(set(re.findall(r"\$\d+", reply)) - {"$8"})   # any money amount that isn't in our facts?
    results.append((f"no invented amounts{' (found ' + ', '.join(made_up) + ')' if made_up else ''}", not made_up))
    for label, ok in results:
        print(f"   [{'ok' if ok else '!!'}] {label}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    ap.add_argument("--model", help=f"override the model (defaults: {MODELS})")
    ap.add_argument("--dry-run", action="store_true", help="print what would be sent; no API call, no key needed")
    args = ap.parse_args()

    for name, instructions, message in [("LAZY", LAZY_INSTRUCTIONS, LAZY_MESSAGE),
                                        ("5 LEVERS", INSTRUCTIONS, MESSAGE)]:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        if args.dry_run:
            print(f"[instructions]\n{instructions}\n\n[message]\n{message}")
            continue
        reply = ask(args.provider, instructions, message, args.model)
        print(f"({args.provider}, {args.model or MODELS[args.provider]})\n\n{reply.strip()}\n\nCheck:")
        check(reply)


if __name__ == "__main__":
    main()
