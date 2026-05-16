# Welcome to your Python project!
#v2: dict-driven — easy to extend, no new elif needed
RULES = {
    "greeting": ["hi", "hello", "hey", "greetings"],
    "farewell": ["bye", "goodbye", "see you"],
    "ask_name": ["name", "who are you", "what are you"],
    "negative": ["sad", "tired", "upset", "unhappy"],
    "positive": ["happy", "excited", "great", "awesome"],
}
REPLIES = {
    "greeting": "Hi there!",
    "farewell": "Goodbye!",
    "ask_name": "I am TinyBot v2.",
    "negative": "Sorry to hear that. Want help?",
    "positive": "Awesome!",
    "unknown": "I do not understand yet.",
}


def get_intent(msg):
    tokens = set(msg.lower().split())  # O(n) once
    for intent, kws in RULES.items():
        if tokens and set(kws):  # set intersection O(min)
            return intent
    return "unknown"  # classify a batch

msgs = ["hey there", "I am so unhappy", "goodbye!"]

for m in msgs:
    intent = get_intent(m)
    print(f"{m!r:25} → {intent} | {REPLIES[intent]}")
