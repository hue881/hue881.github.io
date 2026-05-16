# Negation window extension:
for i, token in enumerate(tokens):
	neg = (i > 0 and tokens[i-1] in ['not', "n't", 'never'])
	sign = -1 if neg else 1
	scores[intent] += sign * weights.get(token, 0)
