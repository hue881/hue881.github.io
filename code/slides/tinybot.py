# === TinyBot v1 — Original Day 17 Code ===
# DETERMINISTIC: same input → same output. Always.

print('Welcome to TinyBot! Type "bye" to quit.')
while True:  # infinite event loop
    msg = input("You: ").lower().strip()  # normalise case O(n) # remove whitespace
    # === INTENT DISPATCH: O(k) where k = # elif branches ===
    if msg == "bye":
        # exact match sentinel print('Bot: Goodbye!')
        break  # exits while loop
    elif "hello" in msg or "hi" in msg:
        print("Bot: Hi there!")  # BUG: 'hi' matches 'this', 'white', 'hi-five'
    elif "name" in msg:
        print("Bot: I am TinyBot, a simple Python chatbot.")
    elif "help" in msg:
        print("Bot: Ask about my name or say hello.")
    elif "sad" in msg or "tired" in msg:
        print(
            "Bot: Breaks and friends can really help."
        )  # BUG: 'not sad' → still triggers this branch
    elif "happy" in msg or "excited" in msg:
        print("Bot: That is awesome news!")
    else:  # fallback — covers ~∞ inputs
        print("Bot: I do not understand yet.")
