RULES = {
	'greeting': ['hi', 'hello', 'hey'],
	'farewell': ['bye', 'goodbye'],
	'negative': ['sad', 'tired', 'upset', 'unhappy'],
	'positive': ['happy', 'excited', 'great', 'awesome'],
}

REPLIES = {
	'greeting': 'Hi there!',
	'farewell': 'Goodbye!',
	'negative': 'Sorry to hear that. Want help?',
	'positive': 'Awesome!',
	'unknown': 'I do not understand yet.',
}

def get_intent(msg):
	tokens = set(msg.lower().split())
	# O(n) once
	for intent, kws in RULES.items():
		if tokens & set(kws):
		# set intersection O(min)
		return intent
	return 'unknown'
