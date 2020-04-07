
def chain(*iterables):
	for it in iterables:
		for element in it:
			yield element

list(chain(['I','love'],['python']))
