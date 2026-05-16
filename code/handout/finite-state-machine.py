state = 'start'
name = None
while True:
	msg = input('You: ').lower().strip()
	if state == 'start':
		if any(w in msg.split() for w in ['hi', 'hello', 'hey']):
			print("Bot: Hi! What's your name?")
			state = 'ask_name'
			# transition
		else:
			print('Bot: Say hello first!')
	elif state == 'ask_name':
		name = msg.title()
		print(f'Bot: Nice to meet you, {name}!')
		state = 'know_name'
	elif state == 'know_name':
		if msg == 'bye':
			print(f'Bot: Goodbye, {name}!'); break
		else:
			print(f'Bot: What else, {name}?')
