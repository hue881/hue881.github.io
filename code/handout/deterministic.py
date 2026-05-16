print('Welcome to TinyBot! Type "bye" to quit.')
while True:
	# infinite event loop
	msg = input('You: ').lower().strip()
	# normalise + clean
	# INTENT DISPATCH: O(k) where k = number of elif branches
	if msg == 'bye':
		print('Bot: Goodbye!')
		break
	elif 'hello' in msg or 'hi' in msg:
		# BUG: 'hi' matches 'this', 'white'
		print('Bot: Hi there!')
	elif 'name' in msg:
		print('Bot: I am TinyBot.')
	elif 'sad' in msg or 'tired' in msg:
	# BUG: 'not sad' still triggers this
		print('Bot: Breaks and friends can really help.')
	elif 'happy' in msg or 'excited' in msg:
		print('Bot: That is awesome news!')
	else:
		print('Bot: I do not understand yet.')
