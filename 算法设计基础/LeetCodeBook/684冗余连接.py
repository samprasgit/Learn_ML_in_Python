class UF:
    def __init__(self, M):
        self.parent = {}
        self.cnt = 0
        # 初始化 parent，size 和 cnt
        for i in range(M):
            self.parent[i] = i
            self.cnt += 1

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        return x
    def union(self, p, q):
        if self.connected(p, q): return
        leader_p = self.find(p)
        leader_q = self.find(q)
        self.parent[leader_p] = leader_q
        self.cnt -= 1
    def connected(self, p, q):
        return self.find(p) == self.find(q)
        
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        uf = UF(1001)
        for fr, to in edges:
            if uf.connected(fr, to): return [fr, to]
            uf.union(fr, to)
