# 选择排序
def findSmallest(arr):
	smallest =arr[0]
	smallest_index=0 
	for i in range(1,len(arr)):
		if arr[i]<smallest:
			smallest = arr[i]
			smallest_index = i 

	return smallest_index 

def selectionSort(arr):
	newArr=[] 
	for i in range(len(arr)):
		# 找出数组最小元素并将其加入到新数组中
		smallest=findSmallest(arr)
		newArr.append(arr.pop(smallest))

	return newArr  

print(selectionSort([5,3,4,6,2,10]))

