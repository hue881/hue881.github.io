def test():
	cases = [
		("hi there", "greeting"),
		("I'm not sad", "unknown"),
		# should fail TinyBot v1
		# add 6 more...
	]
	for msg, expected in cases:
		result = get_intent(msg)
		status = " PASS"✓ if result == expected else " FAIL"✗
		print(f"{status} | {msg!r} → {result}")
