# Ep 1: a tiny AI agent (How AI Works, Easelbot)

The whole agent from the video in one Python file: **brain** (an LLM), **hands** (tools), **memory** (context + a
small RAG search), **the loop** (think, act, observe) and **guardrails** (only listed tools, a $100 refund limit with
a human approval, a step limit, a log line for every step).

## Run it without an API key
```
python agent.py --offline
```
A scripted "brain" walks through Leo's broken-blender ticket so you can watch the loop.

## Run it with Claude
```
pip install anthropic
export ANTHROPIC_API_KEY=your-key        # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key"
python agent.py
python agent.py --ticket "Hi, my $180 coffee machine (order A-2077) arrived cracked. Refund please!"
```
The second ticket is over the $100 limit, so the guardrail asks **you** to approve the refund.

## Try next
- Add a tool (e.g. `replace_item`) to the `TOOLS` menu and `RUN` table.
- Change the rules (the agent's "job description") and see how its behaviour changes.
- Swap the keyword search for real embeddings + a vector database.

Example data only: the shop, orders and policy are made up.
