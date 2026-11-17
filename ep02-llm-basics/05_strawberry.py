"""05 - Why "how many R's in strawberry?" was hard (How AI Works, Ep 2, chapter 1 + homework).

You see letters. The model sees tokens (chunks, each one a number). Counting letters inside a chunk it has never
seen spelled out is like counting the letters in a word you've only ever seen as two stickers.
Spelling the word out first turns every letter into its own token, and then counting is easy.

    pip install tiktoken
    python 05_strawberry.py
    python 05_strawberry.py --ask gemini       # optional: ask a real model both ways (needs that provider's key)

Newer models usually get this right now (often by spelling the word out first, on their own), but the reason it was
hard is still a good look at how they read.
"""
import argparse

import tiktoken

enc = tiktoken.get_encoding("o200k_base")   # OpenAI's tokenizer; other models cut words differently
WORD = "strawberry"


def tokens(text):
    return [enc.decode([i]) for i in enc.encode(text)]


ap = argparse.ArgumentParser()
ap.add_argument("--ask", choices=["gemini", "openai", "anthropic"], help="also ask a real model")
args = ap.parse_args()

print(f"\nYou see letters:      {' '.join(WORD)}   ->  {WORD.count('r')} R's")
print(f"The model sees tokens: {tokens(WORD)}   (ids {enc.encode(WORD)})")
spelled = " ".join(WORD)
print(f"\nSpelled out first:    {tokens(spelled)}")
print(f"                      -> {len(enc.encode(spelled))} tokens, one per letter: now each R is its own piece.")

if args.ask:
    from providers import ask
    for question in [f"How many R's are in the word {WORD}? Answer with just the number.",
                     f"Spell the word {WORD} letter by letter, then count the R's."]:
        print(f"\nQ: {question}\nA: {ask(args.ask, 'Be brief.', question).strip()}")
