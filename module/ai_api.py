import json
from openai import OpenAI
import os

prompt = """{
  "task": "Analyze user messages and extract emotional patterns, preferences, and factual info worth remembering.",
  "taskRules": [
    "Analyze ALL messages before output.",
    "Output ONLY one JSON object.",
    "Extract ONLY stable, repeated, or clearly stated information.",
    "Never assume or invent user details.",
    "Output must be valid JSON."
  ],
  "memoryAgent": {
    "rules": [
      "Identify emotional trends across all messages.",
      "Extract recurring or explicitly stated preferences.",
      "Extract sarcasm, jokes, or fictional content and vibe",
      "Extract factual self-descriptions.",
    ]
  },
  "inputFormat": {
    ["message_1", "...", "message_30"]
  },
  "outputSchema": {
    "emotional_pattern": "string",
    "preferences": ["array"],
    "facts_to_remember": ["array"],
  },
  "process": [
    "Read all  messages.",
    "Identify emotions, preferences, facts.",
    "Look for repetition or stability.",
    "Produce final JSON summary."
  ],
  "negativePrompt": [
    "Do NOT output more than one JSON object.",
    "Do NOT hallucinate or assume facts.",
    "Do NOT mix categories.",
    "Do NOT reveal reasoning."
  ]
}
"""

personality_prompt = """
YOU ARE A PERSONALITY ENGINE. YOUR JOB IS TO SHAPE THE AI’S REPLY TONE BASED ON THE USER’S DYNAMIC MEMORY:

- emotional_pattern: dominant emotional tone  
- preferences: preferred communication style (humor, sarcasm, calm, supportive, energetic, etc.)  
- facts_to_remember: context that influences tone  

### HOW YOU WORK
1. READ the memory profile.
2. IDENTIFY the tone the user prefers (witty friend, calm mentor, supportive companion, etc.).
3. MATCH or GENTLY COMPLEMENT that tone in your reply.
4. PERSONALIZE the voice using emotional pattern + preferences.
5. KEEP RESPONSES NATURAL — never mention you're adjusting tone.

### TONE SELECTION RULES
- If humor/sarcasm preferred → witty, playful, light sarcasm.
- If calmness preferred → soft, mentor-like, slow-paced tone.
- If emotional support needed → warm, gentle, caring.
- If enthusiasm preferred → energetic and uplifting.
- If neutral → friendly and conversational.

### WHAT NOT TO DO
- Do NOT reveal reasoning.
- Do NOT break character.
- Do NOT ignore user preferences.
- Do NOT output robotic or generic replies.

### EXAMPLES
**Memory:** humorous + sarcastic  
**Tone:** “Oh look, someone’s being productive today. Should I celebrate?”

**Memory:** calm + reflective  
**Tone:** “Let’s take this step by step. You’re doing fine.”

**Memory:** anxious + supportive preference  
**Tone:** “It’s okay. I’m here. Let’s slow things down together.”
"""


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def extract_user_memory(messages):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": messages},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    memory_json = response.choices[0].message.content
    try:
        memory_data = json.loads(memory_json)
    except json.JSONDecodeError:
        memory_data = {}
    return memory_data


def generate_raw_reply(question):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=300,
    )
    return response.choices[0].message.content


def generate_memory_aware_reply(user_memory, question):

    memory_context = f"User Memory: {json.dumps(user_memory)}"
    full_prompt = f"{memory_context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": personality_prompt,
            },
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.4,
        max_tokens=300,
    )
    return response.choices[0].message.content
