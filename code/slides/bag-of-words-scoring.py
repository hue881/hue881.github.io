# Intent weight dictionary
INTENTS = {
    "greeting": {"hello": 3, "hi": 3, "hey": 2, "greetings": 2},
    "negative": {
        "sad": 3,
        "tired": 2,
        "upset": 3,
        "unhappy": 3,
        "awful": 3,
        "depressed": 4,
    },
    "positive": {"happy": 3, "great": 2, "excited": 3, "awesome": 3, "fantastic": 3},
}


def classify(msg):
    tokens = msg.lower().split()
    scores = {intent: 0 for intent in INTENTS}
    for token in tokens:
        for intent, weights in INTENTS.items():
            scores[intent] += weights.get(token, 0)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


for i, token in enumerate(tokens):
    neg = i > 0 and tokens[i - 1] in ["not", "n't", "never"]
    sign = -1 if neg else 1
    scores[intent] += sign * weights.get(token, 0)

