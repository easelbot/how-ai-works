# Ep 2: LLMs, explained and applied (How AI Works, Easelbot)

Small, runnable experiments for every idea in the video: what an LLM actually reads, how it picks the next word, the
temperature dial, and Nina's support reply with **the 5 levers** (prompt, context, examples, temperature, check).
Four of them need **no API key**. They run on a normal laptop CPU.

| File | Shows | Chapter | Needs |
|---|---|---|---|
| [`01_tokens.py`](01_tokens.py) | the real tokens + IDs a model reads ("unbelievable", "strawberry", your own text) | 1 | `tiktoken`, no key |
| [`02_next_token.py`](02_next_token.py) | **real** next-token probabilities from a small open model, then the loop: pick, add, repeat | 1 | `transformers` + `torch`, no key |
| [`03_temperature.py`](03_temperature.py) | the same bars at temperature 0.2 / 1.0 / 1.8, plus sampled endings at each | 4 | `transformers` + `torch`, no key |
| [`04_five_levers.py`](04_five_levers.py) | Nina's reply two ways: a lazy prompt vs all 5 levers, plus an automatic check of the reply | 4 + 6 | one provider (below), or `--dry-run` with no key |
| [`05_strawberry.py`](05_strawberry.py) | why counting the R's in "strawberry" was hard, and why spelling it out fixes it | 1 | `tiktoken`, no key (`--ask` optional) |
| [`06_on_screen_card.py`](06_on_screen_card.py) | the exact 14-line code card from the video (Gemini) | 6 | `google-genai` + `GEMINI_API_KEY` |
| [`providers.py`](providers.py) | one small function per provider: Gemini, OpenAI, Claude, side by side | 6 | — |

## Setup
Python 3.10 or newer.
```
git clone https://github.com/easelbot/how-ai-works
cd how-ai-works/ep02-llm-basics
pip install -r requirements.txt
```
The first run of `02` or `03` downloads GPT-2 (about 0.5 GB) once. GPT-2 is old and small (2019), so its guesses
are rougher than today's chatbots, but the mechanism is exactly the same.
On Windows you may see a warning about "symlinks" from `huggingface_hub`: it's harmless.

## No key needed
```
python 01_tokens.py
python 01_tokens.py "Try your own sentence"
python 02_next_token.py
python 02_next_token.py "Once upon a time, a robot"
python 03_temperature.py
python 05_strawberry.py
python 04_five_levers.py --dry-run        # shows exactly what would be sent to the model
```

## With an API key: pick one provider
| Provider | Install | Key (environment variable) | Cost |
|---|---|---|---|
| **Gemini** (default) | `pip install google-genai` | `GEMINI_API_KEY`, from [Google AI Studio](https://aistudio.google.com) | **free tier** (rate limits) |
| OpenAI | `pip install openai` | `OPENAI_API_KEY`, from [platform.openai.com](https://platform.openai.com) | paid (a few cents for these runs) |
| Claude | `pip install anthropic` | `ANTHROPIC_API_KEY`, from [platform.claude.com](https://platform.claude.com) | paid (a few cents for these runs) |

Set the key as an environment variable, never in the code:
```
# macOS / Linux
export GEMINI_API_KEY="your-key"
# Windows PowerShell (this window only)
$env:GEMINI_API_KEY="your-key"
# Windows, permanently: Start > "Edit environment variables for your account" > New, then restart your terminal
```
Then:
```
python 04_five_levers.py                          # Gemini
python 04_five_levers.py --provider openai
python 04_five_levers.py --provider anthropic
python 04_five_levers.py --model gemini-4-hydrogen   # any other model from that provider
python 05_strawberry.py --ask gemini
python 06_on_screen_card.py
```

**Free tier note:** on Gemini's free tier, what you send may be used to improve Google's products. All the data here
is made up; don't send private or customer data on a free tier.
**Never commit a key.** If one leaks, delete it in the provider's dashboard and create a new one.
**Model names change often.** These were current in November 2026. If you get "model not found", check the
provider's models page and pass `--model`.

## Why there's no temperature in the API code
In the video, temperature is lever four. The newest models from all three companies tune it themselves: Claude's
newest models reject the setting, OpenAI's newest reject it unless reasoning is switched off, and Gemini's newest
ignore it. Older and open models still let you turn the dial, which is why `03_temperature.py` runs an open model
on your own laptop: there you control it directly. With the big chatbots, steer with words instead: "be precise",
or "give me ten wild ideas".

## Example output
(from a run on 2026-11-17; sampled text and API replies will differ on your machine)

`python 01_tokens.py`
```
'strawberry'  ->  3 token(s)
   'st'               id 302
   'raw'              id 1618
   'berry'            id 19772
```
`python 02_next_token.py`: no single "right" answer, every option gets a probability
```
"The cat sat on the ___"   the model scores all 50,257 tokens. The top 8:
   ' floor'         7.6%  ###
   ' bed'           6.5%  ###
   ' couch'         5.4%  ##
   ' ground'        5.2%  ##
   ...
Result: 'The cat sat on the floor, and the cat was still asleep'
```
`python 03_temperature.py`: the same top token at three temperatures
```
  temperature 0.2:   ' "'  91.1%   everything else   0.0%
  temperature 1.0:   ' "'  12.6%   everything else  69.8%
  temperature 1.8:   ' "'   0.7%   everything else  97.7%
```
At 1.8 the endings turn into nonsense ("Omega pu Pak 92 Cook92 Nutrition…").

`python 04_five_levers.py` (Gemini, the 5-levers reply)
```
Hi Leo, sorry, that's not the experience we want for you. Your blender (order B-3141) has a new delivery date
of 20 November, which is 8 days past the promised 12 November date.
Per our refund policy: "If an order arrives more than 5 days late, we refund the $8 shipping fee." ...
Check:
   [ok] uses the customer's name (Leo)
   [ok] gives the real new date (20 November)
   [ok] mentions the $8 shipping refund
   [ok] under 100 words
   [ok] no invented amounts
```
The lazy prompt's reply, from the same model, failed three of those checks, signed off "[Your Name]" and promised a
follow-up "within the next 30 minutes" that nobody had asked for.

**Tested:** every file was run on a Windows laptop CPU (no GPU) with Python 3.11, and `04`/`05 --ask`/`06` against
Gemini's free tier. The OpenAI and Claude functions in `providers.py` follow the official docs (November 2026) but
were not run for this release; if something there breaks, please open an issue.

## Try next
- `02`: try a prompt in your language, or a famous line ("To be or not to"). How sure is the model?
- `03`: push the temperature to 3.0 in the file. When does it turn into nonsense?
- `04`: delete one lever at a time from `INSTRUCTIONS` / `MESSAGE` and watch which check fails.
- `providers.py`: compare the three functions. Same job, slightly different words.

Example data only: the shop, orders, customers and policy are made up. Tokens come from OpenAI's `o200k_base`
tokenizer; every model family cuts text a little differently, but the idea is the same.
