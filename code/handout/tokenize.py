def tokenize(sentence):
	import re
	clean = re.sub(r'[^a-z\s]', '', sentence.lower())
	return clean.split()

tokens = tokenize("I'm NOT sad!")
# ['im', 'not', 'sad']
