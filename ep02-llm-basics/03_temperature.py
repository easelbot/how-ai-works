"""03 - The temperature dial (How AI Works, Ep 2, chapter 4: lever four).

Temperature changes how adventurous the pick is. The model's scores are divided by the temperature before they
become probabilities:
    low  (0.2)  -> the top token gets almost all the probability: steady, predictable
    1.0         -> the model's own probabilities
    high (1.8)  -> the bars flatten, unlikely tokens get a real chance: creative, surprising, more mistakes

This script shows the same probability bars at three temperatures, then writes three short endings at each one.
It runs a small open model (GPT-2) on your CPU, so you control the dial directly. (Many of the newest commercial
models lock this dial and tune it for you; with open models you run yourself, you can still turn it.)

    pip install transformers torch
    python 03_temperature.py
    python 03_temperature.py "My new coffee shop is called"

No API key needed.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
PROMPT = " ".join(sys.argv[1:]) or "The best name for a new coffee shop is"
TEMPERATURES = [0.2, 1.0, 1.8]
TOP = 6
ENDINGS = 3        # sampled endings per temperature
LENGTH = 10        # tokens per ending

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

ids = tokenizer(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1]

# Show the top tokens at the model's own temperature (1.0), then see how the dial reshapes their bars.
top_ids = torch.topk(logits, TOP).indices.tolist()
print(f'\n"{PROMPT} ___"')
for t in TEMPERATURES:
    probs = torch.softmax(logits / t, dim=-1)     # <- the whole dial is this one division
    print(f"\n  temperature {t}:")
    for i in top_ids:
        p = probs[i].item()
        print(f"     {tokenizer.decode([i])!r:<14} {p:6.1%}  {'#' * max(1, round(p * 40)) if p >= 0.005 else '.'}")
    rest = 1 - probs[top_ids].sum().item()       # all the other ~50,000 tokens together
    print(f"     {'everything else':<14} {rest:6.1%}  (shared by {probs.numel() - TOP:,} other tokens)")

print(f"\nNow {ENDINGS} endings per temperature (sampled, so a new run gives new endings):")
for t in TEMPERATURES:
    print(f"\n  temperature {t}:")
    for _ in range(ENDINGS):
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=t, top_k=0,
                             max_new_tokens=LENGTH, pad_token_id=tokenizer.eos_token_id)
        ending = tokenizer.decode(out[0, ids.shape[1]:]).replace("\n", " ").strip()
        print(f"     ...{ending}")

print("\nLow: almost the same every time. High: more variety, and more nonsense.")
