class Test :
	def __init__(self,arr):
		self.numbers = arr
		#用；来标记是否便利过
		self.visited =[None] * len(self.numbers)
		#图的二维数组表示
		self.graph=[[None] * len(self.numbers for i in range(len(self.numbers)))]
		self.n=6 

		self.combination=''
		self.s=set()

	# 对图从节点开始进行深度优先遍历
	# 输入参数：start 遍历的起始位置
	def depthFirstSearch(self,start):
		self.visited[start]=True 
		self.combination+=str(self.numbers[start])
		if len(self.combination)==self.n:
			# 4不出现在第三个位置
			if self.combination.index('4')!=2:
				self.add(self.combination)

			j=0 
			while j<self.n:
				if self.graph[start][j]==1 and self.visited[j]==False:
					self.depthFirstSearch(j)
				j+= 1
			self.combination=self.combination[:-1]
			self.visited[start]=False 

	#  获取1，2，2，3，4，5的左右组合，使得“4”不能在第三位， 3月与5不能相连
	
	def getAllCombinations(self):
		# 构造图
		i=0 
		while i<self.n:
			j=0
			if i==j:
				self.graph[i][j]=0
			else:
				self.graph[i][j]=1
				j+=1
				i+=1 
		# 确保在遍历的时候3与5是不可达的
		self.graph[3][5]= 0
		self.graph[5][3]=1 

		# 分别从不同的节点出发深度优先遍历  
		i=0
		while i<self.n:
			self.depthFirstSearch(i)

			i+=1 
		def printAllCombinations(self):
			for strs in self.s:
				print(strs,)

if __name__=="__main__":
	arr=[1,2,3,4,5]
	t=Test(arr)
	t.getAllCombinations() 
	# 打印所有组合
	t.printAllCombinations()


