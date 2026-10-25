import numpy as np
from numba import jit

@jit
def _fill_table(align_mod, s1, s2, m,):
	n = len(s2)
	dp = np.zeros((m+1, n+1), dtype=np.int32) 

	cost = np.zeros(m, dtype=np.int32)
	bottleneck = np.full(m, n, dtype=np.int32)  

	if align_mod == "global":
		for i in range(m + 1):
			dp[i][0] = i  # Delete all chars from s1
		for j in range(n + 1):
			dp[0][j] = j  # Insert all chars of s2
	elif align_mod == "semi_global":
		for j in range(n + 1):
			dp[0][j] = j  # Insert all chars of s2
		

	# Fill the rest of the table
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if s1[i - 1] == s2[j - 1]:
				dp[i][j] = dp[i - 1][j - 1]      # No operation needed
			else:
				dp[i][j] = 1 + min(
					dp[i - 1][j],      # Delete from s1
					dp[i][j - 1],      # Insert into s1
					dp[i - 1][j - 1]   # Replace
				)
		cost[i-1] = dp[i][-1] 
		bottleneck[i-1] = min([cost[i-1], bottleneck[i-2]])
	return dp, bottleneck

@jit
def _fill_table_cost(align_mod, s1, s2, m,):
	n = len(s2)
	dp = np.zeros((m+1, n+1), dtype=np.int32) 

	cost = np.zeros(m, dtype=np.int32)

	if align_mod == "global":
		for i in range(m + 1):
			dp[i][0] = i  # Delete all chars from s1
		for j in range(n + 1):
			dp[0][j] = j  # Insert all chars of s2
	elif align_mod == "semi_global":
		for j in range(n + 1):
			dp[0][j] = j  # Insert all chars of s2
		

	# Fill the rest of the table
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if s1[i - 1] == s2[j - 1]:
				dp[i][j] = dp[i - 1][j - 1]      # No operation needed
			else:
				dp[i][j] = 1 + min(
					dp[i - 1][j],      # Delete from s1
					dp[i][j - 1],      # Insert into s1
					dp[i - 1][j - 1]   # Replace
				)
		cost[i-1] = dp[i][-1] 
	return dp, cost

class EditDistance:
	"""
	Dynamic Programming solution for Edit Distance (Levenshtein Distance).
	Calculates the minimum number of operations (insert, delete, replace)
	required to transform string s1 into string s2.
	"""

	def __init__(self, s1: str, align_mod="local"):
		self.s1 = s1
		self.m = len(s1)
		self.align_mod = align_mod
		

	def fill_table(self, s2) -> None:
		"""Fill the DP table bottom-up."""
		_, bottleneck = _fill_table(self.align_mod,self.s1,s2,self.m)
		# self.print_table(dp,s2)
		return bottleneck

	def fill_table_cost(self, s2) -> None:
		"""Fill the DP table bottom-up."""
		_, cost = _fill_table_cost(self.align_mod,self.s1,s2,self.m)
		# self.print_table(dp,s2)
		return cost

	def print_table(self, dp, s2) -> None:
		"""Pretty-print the DP table for debugging/visualisation."""
		n = len(s2)
		header = "    " + "  ".join(["_"] + list(s2))
		print(header)
		for i in range(self.m + 1):
			row_label = "_" if i == 0 else self.s1[i - 1]
			print(f"{row_label}  ", end="")
			print("  ".join(str(dp[i][j]) for j in range(n + 1)))


# --- Example usage ---
if __name__ == "__main__":
	s1, s2 = "ACTCTATTTT", "ACTGCTTTT"
	# oligo = "ACTGCTTTT"

	# all_paths = gt.topology.all_paths(graph, 0, 15)
	# for p in all_paths:
	#     print(''.join([v_label[i] for i in p]))
	#     print("----------------------------")
	# print(top_order)
	# print([v_label[i] for i in top_order])
	# # print([graph.vertex_index[i] for i in graph.vertices()])
	# # print([v_label[i] for i in graph.vertices()])
	
	dp_obj = EditDistance(s1=s1, align_mod="global")
	bn = dp_obj.fill_table(s2)

	s1, s2 = "GGGACGTTACGTCCC", "ACGTACGT"

	dp_obj = EditDistance(s1=s1, align_mod="semi_global")
	cost = dp_obj.fill_table_cost(s2)
	bn = dp_obj.fill_table(s2)
	print(cost)
	print(bn)
