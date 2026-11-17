"""06 - The code card from the video, exactly as shown on screen (How AI Works, Ep 2, chapter 6).

Nina's setup with the levers, for Gemini (free tier). The full version for all three providers, with the automatic
checks, is 04_five_levers.py (+ providers.py for the OpenAI and Claude versions).

    pip install google-genai
    python 06_on_screen_card.py          # needs GEMINI_API_KEY (free from aistudio.google.com)

Example data only: the shop, order, customer and policy are made up.
"""
# The facts and examples (lever 2: CONTEXT, lever 3: EXAMPLES) and the customer's email
FACTS = ("Facts:\n"
         "Order B-3141 | customer: Leo | item: blender | promised: 12 November | new delivery date: 20 November\n"
         "Refund policy: if an order arrives more than 5 days late, we refund the $8 shipping fee. "
         "If it hasn't arrived within 30 days, the customer can cancel for a full refund.\n\n")
EXAMPLES = ("My tone (match it):\n"
            "Hi Priya, thanks for telling us, and sorry about the mix-up! Your lamp arrives Friday. - Nina\n"
            "Hi Sam, sorry, that's not the experience we want for you. The new mug ships today. - Nina\n\n")
EMAIL = "Customer email:\nHi, my blender (order B-3141) still hasn't arrived after two weeks. Where is it? - Leo"

# ───────────── the card (14 lines) ─────────────
from google import genai
client = genai.Client()

RULES = ("You are Nina, a friendly "
  "support agent. Under 100 words. "
  "Quote the refund policy. If "
  "unsure, say you don't know.")

reply = client.interactions.create(
  model="gemini-3.5-flash-lite",
  system_instruction=RULES,
  input=FACTS + EXAMPLES + EMAIL,
)
print(reply.output_text)
