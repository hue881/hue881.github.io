# Starter — fill in the blanks:
RULES = { 'greeting': ['hi', 'hello'], _____ }
REPLIES = { 'greeting': 'Hi there!', _____ }
def get_intent(msg):
	tokens = _____
	for intent, kws in RULES.items():
		if _____:
			return intent
	return 'unknown'
