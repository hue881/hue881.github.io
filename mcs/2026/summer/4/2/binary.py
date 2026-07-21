def binary_search(lst, value):
	low = 0
	high = len(list) - 1
	while low <= high:
		mid = low + (high - low) // 2
		if lst[mid] == value:
			return mid
		elif lst[mid] < value:
			low = mid + 1
		else:
			high = mid - 1
	return -1


lst = [2, 3, 5, 10, 20]
value = 10

res = binary_search(lst, value)
if res != -1:
	print(f'{value} is at index {res}')
else:
	print(f'{value} is not in lst')
