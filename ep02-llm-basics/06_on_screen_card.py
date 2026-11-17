"""06 - The code card from the video, exactly as shown on screen (How AI Works, Ep 2, chapter 6).

Nina's setup with all the levers, for Gemini (free tier). The full version for all three providers, with checks,
is 04_five_levers.py.

    pip install google-genai
    python 06_on_screen_card.py          # needs GEMINI_API_KEY (free from aistudio.google.com)

Example data only: the shop, order, customer and policy are made up.
"""
ORDER = "Order B-3141 | customer: Leo | item: blender | promised: 12 November | new delivery date: 20 November"
POLICY = ("Refund policy: if an order arrives more than 5 days late, we refund the $8 shipping fee. "
          "If it hasn't arrived within 30 days, the customer can cancel for a full refund.")
EXAMPLES = ("Hi Priya, thanks for telling us, and sorry about the mix-up! Your lamp arrives Friday. - Nina\n"
            "Hi Sam, sorry, that's not the experience we want for you. The new mug ships today. - Nina")
EMAIL = "Hi, my blender (order B-3141) still hasn't arrived after two weeks. Where is it? - Leo"

# ───────────── the card ─────────────
from google import genai
client = genai.Client()                            # reads GEMINI_API_KEY

reply = client.interactions.create(
    model="gemini-3.5-flash-lite",
    system_instruction=(                           # 1 PROMPT: role, task, format
        "You are Nina, a friendly support agent. Reply in under 100 words. "
        "Use only the facts given and quote the refund policy. "      # 5 CHECK
        "If something isn't in the facts, say you don't know."),
    input=(f"Facts:\n{ORDER}\n{POLICY}\n\n"          # 2 CONTEXT
           f"My tone:\n{EXAMPLES}\n\n"               # 3 EXAMPLES
           f"Customer email:\n{EMAIL}"),
)
print(reply.output_text)
