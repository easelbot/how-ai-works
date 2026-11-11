"""A tiny AI agent: the 5 pillars from Easelbot's "How AI Works, Ep 1" in one file.

    BRAIN       an LLM decides the next step            (Claude, via the Anthropic API)
    HANDS       tools it can ask for                    (find_order, search_policy, issue_refund, send_email)
    MEMORY      context + a small RAG search            (the conversation + search_policy over policy pages)
    LOOP        think -> act -> observe, until done     (the while-loop below)
    GUARDRAILS  only these tools, a $100 refund limit,  (too_big() -> ask_human(), MAX_STEPS, a log line per step)
                a step limit, and a log of every step

Run it:
    python agent.py --offline     # no API key needed: a scripted "brain" shows the loop step by step
    pip install anthropic         # then, with an API key (export ANTHROPIC_API_KEY=...):
    python agent.py               # Claude makes the decisions
    python agent.py --ticket "Hi, my $180 coffee machine (order A-2077) arrived cracked. Refund please!"

The shop, orders and policy are made up (example data). Real agents use real systems behind the same tools, and a
real RAG search uses embeddings + a vector database instead of the keyword search below.
"""
import argparse
import json

MODEL = "claude-opus-5-5"
MAX_STEPS = 10            # guardrail: never loop forever
REFUND_LIMIT = 100        # guardrail: bigger refunds need a human

# ─────────────────────────── the (pretend) shop ───────────────────────────
ORDERS = {
    "A-1042": {"customer": "Leo", "item": "blender", "price": 49, "delivered_days_ago": 3},
    "A-2077": {"customer": "Maya", "item": "coffee machine", "price": 180, "delivered_days_ago": 5},
}
POLICY_PAGES = [  # MEMORY (long-term): the "filing cabinet", already cut into small chunks
    "Refunds: customers can get a full refund within 30 days of delivery.",
    "Damaged items: ask the customer for a photo of the damage before refunding.",
    "Shipping: standard delivery takes 3 to 5 business days.",
    "Warranty: electrical items have a 1-year warranty for faults.",
]
TICKET = "Hi, my blender (order A-1042) arrived broken. Photo attached. I want my money back. - Leo"
RULES = ("You are Brightbox customer support. Be kind and brief. Always check the order and the refund policy "
         "before you act. Use the tools to do the work, then reply to the customer with send_email. "
         "When everything is done, give a one-line summary.")   # the agent's "job description"


# ─────────────────────────── HANDS: the tools ───────────────────────────
def find_order(order_id):
    order = ORDERS.get(order_id)
    return json.dumps(order) if order else f"Error: no order {order_id}"


def search_policy(question, top=2):
    """MEMORY: a toy RAG search - score each chunk by shared words, return the best ones."""
    words = set(question.lower().split())
    ranked = sorted(POLICY_PAGES, key=lambda page: -len(words & set(page.lower().split())))
    return "\n".join(ranked[:top])


def issue_refund(order_id, amount):
    return f"Refund of ${amount} for order {order_id} sent."


def send_email(to, text):
    print(f"\n    [email to {to}] {text}\n")
    return "Email sent."


RUN = {"find_order": find_order, "search_policy": search_policy, "issue_refund": issue_refund,
       "send_email": send_email}


def schema(props, required):
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [  # the "menu" the brain can pick from: a name, one line of description, what it needs
    {"name": "find_order", "description": "Look up an order by its ID.", "strict": True,
     "input_schema": schema({"order_id": {"type": "string"}}, ["order_id"])},
    {"name": "search_policy", "description": "Search the shop's policy pages.", "strict": True,
     "input_schema": schema({"question": {"type": "string"}}, ["question"])},
    {"name": "issue_refund", "description": "Refund an order (amount in dollars).", "strict": True,
     "input_schema": schema({"order_id": {"type": "string"}, "amount": {"type": "number"}},
                            ["order_id", "amount"])},
    {"name": "send_email", "description": "Email the customer.", "strict": True,
     "input_schema": schema({"to": {"type": "string"}, "text": {"type": "string"}}, ["to", "text"])},
]


# ─────────────────────────── GUARDRAILS ───────────────────────────
def too_big(name, args):
    return name == "issue_refund" and args.get("amount", 0) > REFUND_LIMIT


def ask_human(name, args):
    """Human in the loop: a person approves (or blocks) the risky step."""
    answer = input(f"    GUARDRAIL: the agent wants {name}({args}). Approve? [y/N] ").strip().lower()
    return answer == "y"


def run_tool(name, args):
    if name not in RUN:                                   # least privilege: only tools on the menu exist
        return f"Error: unknown tool {name}", True
    if too_big(name, args) and not ask_human(name, args):
        return "A human declined this refund. Tell the customer a manager will review it.", True
    return RUN[name](**args), False


# ─────────────────────────── BRAIN: Claude (or a scripted stand-in) ───────────────────────────
def claude_brain():
    import anthropic
    client = anthropic.Anthropic()        # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)

    def step(history):
        return client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=RULES,
            tools=TOOLS,
            messages=history,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",          # if the model declines a request, the API retries on a fallback model
        )
    return step


class _Block:                             # minimal stand-ins for the SDK's content blocks (offline mode only)
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


class _Reply:
    def __init__(self, stop_reason, content):
        self.stop_reason, self.content = stop_reason, content


def offline_brain():
    """A scripted 'brain' that makes the decisions a real model would make for Leo's ticket - so you can watch
    the loop without an API key. (A real LLM picks these steps itself; nobody writes them in advance.)"""
    plan = [
        ("I need the order.", "find_order", {"order_id": "A-1042"}),
        ("What does the policy say about damaged items?", "search_policy", {"question": "refund damaged items photo"}),
        ("Delivered 3 days ago, photo attached, under 30 days: refund $49.", "issue_refund",
         {"order_id": "A-1042", "amount": 49}),
        ("Tell Leo it's done.", "send_email", {"to": "Leo", "text": "Sorry about the blender! Your $49 refund is on its way."}),
    ]
    state = {"i": 0}

    def step(history):
        i = state["i"]
        state["i"] += 1
        if i < len(plan):
            thought, name, args = plan[i]
            return _Reply("tool_use", [_Block("text", text=thought),
                                       _Block("tool_use", id=f"step{i}", name=name, input=args)])
        return _Reply("end_turn", [_Block("text", text="Done: order checked, policy followed, $49 refunded, Leo emailed.")])
    return step


# ─────────────────────────── LOOP ───────────────────────────
def run_agent(ticket, brain):
    history = [{"role": "user", "content": ticket}]       # MEMORY (short-term): everything the brain can see
    for n in range(1, MAX_STEPS + 1):
        reply = brain(history)                            # THINK: ask the brain for the next step
        for block in reply.content:
            if block.type == "text" and block.text.strip():
                print(f"  {n:>2}. think    {block.text.strip()}")
        if reply.stop_reason == "end_turn":               # it says it's done -> stop
            return
        if reply.stop_reason == "refusal":
            print("  the model declined this request.")
            return
        if reply.stop_reason == "max_tokens":
            print("  the reply was cut off (max_tokens) - stopping.")
            return
        history.append({"role": "assistant", "content": reply.content})   # keep its turn exactly as it came
        if reply.stop_reason == "pause_turn":
            continue
        results = []
        for call in (b for b in reply.content if b.type == "tool_use"):
            print(f"      act      {call.name}({json.dumps(call.input)})")   # GUARDRAIL: log every step
            output, failed = run_tool(call.name, call.input)               # ACT: our code presses the button
            print(f"      observe  {output.splitlines()[0]}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output,
                            **({"is_error": True} if failed else {})})
        history.append({"role": "user", "content": results})               # OBSERVE: results go into memory
    print(f"  stopped after {MAX_STEPS} steps (guardrail).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A tiny AI agent (Easelbot - How AI Works, Ep 1)")
    ap.add_argument("--offline", action="store_true", help="no API key: a scripted brain walks Leo's ticket")
    ap.add_argument("--ticket", default=TICKET, help="the customer message to handle")
    a = ap.parse_args()
    print(f"\nTICKET: {a.ticket if not a.offline else TICKET}\n")
    run_agent(TICKET if a.offline else a.ticket, offline_brain() if a.offline else claude_brain())
