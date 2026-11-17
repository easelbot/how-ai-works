"""01 - Tokens: what an LLM actually reads (How AI Works, Ep 2, chapter 1).

An LLM never sees letters or words. Text is cut into chunks called TOKENS, and every token is just a number (its ID).
This script shows the real tokens and IDs for a few words, then for any text you type.

    pip install tiktoken
    python 01_tokens.py
    python 01_tokens.py "Your own sentence here"

No API key needed. tiktoken is OpenAI's tokenizer ("o200k_base", used by GPT-4o and later). Every model family has
its own tokenizer, so the exact chunks and IDs differ between models, but the idea is the same everywhere.
"""
import sys

import tiktoken

enc = tiktoken.get_encoding("o200k_base")


def show(text):
    ids = enc.encode(text)
    chunks = [enc.decode([i]) for i in ids]
    print(f"\n{text!r}  ->  {len(ids)} token(s)")
    for chunk, token_id in zip(chunks, ids):
        print(f"   {chunk!r:<18} id {token_id}")


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    for word in ["cat", "unbelievable", "strawberry", "The cat sat on the", "Easelbot"]:
        show(word)

    # Rule of thumb: in English, one token is a bit less than one word on average (roughly 0.75 to 1).
    sample = ("Large language models read text as tokens. A short word is usually one token, "
              "while a long or rare word gets split into several pieces.")
    words, tokens = len(sample.split()), len(enc.encode(sample))
    print(f"\nA sample sentence: {words} words -> {tokens} tokens ({words / tokens:.2f} words per token)")
