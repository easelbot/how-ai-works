"""02 - Next-token prediction, for real (How AI Works, Ep 2, chapter 1).

The core trick of every LLM: look at the text so far, SCORE EVERY possible next token, pick one, add it, repeat.
This script runs a small open model (GPT-2, 124M parameters, ~0.5 GB download the first time) on your own CPU and
prints the REAL probabilities it gives to the next token. Then it writes a few tokens, one at a time, showing the
loop.

    pip install transformers torch
    python 02_next_token.py
    python 02_next_token.py "Once upon a time, a robot"

No API key needed. GPT-2 is old and small (2019), so its guesses are rougher than today's models, but the mechanism
is exactly the same one inside ChatGPT, Gemini and Claude.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
PROMPT = " ".join(sys.argv[1:]) or "The cat sat on the"
STEPS = 8          # how many tokens to write in the loop
TOP = 8            # how many candidates to print

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()


def next_token_probs(text):
    """Probabilities for the token after `text`."""
    # 1. text -> tokens (numbers)
    ids = tokenizer(text, return_tensors="pt")
    ids = ids.input_ids
    # 2. a score for EVERY possible next token
    with torch.no_grad():
        scores = model(ids).logits[0, -1]
    # 3. scores -> percentages (they add to 100%)
    return torch.softmax(scores, dim=-1)


def bar(p, width=40):
    return "#" * max(1, round(p * width)) if p >= 0.005 else "."


# 1) One step: every option gets a score
probs = next_token_probs(PROMPT)
top = torch.topk(probs, TOP)
print(f'\n"{PROMPT} ___"   the model scores all {probs.numel():,} tokens. The top {TOP}:\n')
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"   {tokenizer.decode([i])!r:<14} {p:6.1%}  {bar(p)}")

# 2) The loop: pick the most likely token, add it, score again
print(f"\nThe loop (always picking the most likely token, {STEPS} times):\n")
text = PROMPT
for step in range(1, STEPS + 1):
    probs = next_token_probs(text)
    top3 = torch.topk(probs, 3)
    picked = tokenizer.decode([top3.indices[0].item()])
    options = ", ".join(f"{tokenizer.decode([i])!r} {p:.0%}"
                        for p, i in zip(top3.values.tolist(), top3.indices.tolist()))
    text += picked
    print(f"   step {step}: {options:<48} -> picks {picked!r}")
print(f"\nResult: {text!r}")
print("\nThat's how an LLM writes everything: one token at a time.")
