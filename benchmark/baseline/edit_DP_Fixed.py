import numpy as np
from numba import jit

@jit
def _fill_table(s1, s2, m,):
	n = len(s2)
	dp = np.zeros((m+1, n+1, n+1), dtype=np.int32) 

	cost = np.zeros(m, dtype=np.int32)
	bottleneck = np.full(m, n, dtype=np.int32)  
	for j in range(n + 1): #double check
		dp[:,j,0] = j  # Insert all chars of s2
		dp[0,j,:] = j  # Insert all chars of s2
		dp[1:,0,j] = j  # Insert all chars of s2
	temp_k = 1
	while temp_k <= n:
		dp[temp_k,0,temp_k:] = temp_k
		temp_k += 1

	# Fill the rest of the table
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if s1[i - 1] == s2[j - 1]:
				for k in range(1, n+1):
					dp[i,j,k] = dp[i-1, j-1, k-1]      # No operation needed
			else:
				for k in range(1, n+1):
					dp[i,j,k] = 1 + min(
						dp[i-1, j, k-1],      # Delete from s1
						dp[i, j-1, k],      # Insert into s1
						dp[i-1, j-1, k-1]   # Replace
					)
		cost[i-1] = dp[i][-1][-1] 
		bottleneck[i-1] = min([cost[i-1], bottleneck[i-2]])
	return dp, bottleneck

@jit
def _fill_table_cost(s1, s2, m,):
	n = len(s2)
	dp = np.zeros((m+1, n+1, n+1), dtype=np.int32) 

	cost = np.zeros(m, dtype=np.int32)

	for j in range(n + 1): #double check
		dp[:,j,0] = j  # Insert all chars of s2
		dp[0,j,:] = j  # Insert all chars of s2
		dp[1:,0,j] = j  # Insert all chars of s2
	temp_k = 1
	while temp_k <= n:
		dp[temp_k,0,temp_k:] = temp_k
		temp_k += 1
		

	# Fill the rest of the table
	for i in range(1, m + 1):
		for j in range(1, n + 1):
			if s1[i - 1] == s2[j - 1]:
				for k in range(1, n+1):
					dp[i,j,k] = dp[i-1, j-1, k-1]      # No operation needed
			else:
				for k in range(1, n+1):
					dp[i,j,k] = 1 + min(
						dp[i-1, j, k-1],      # Delete from s1
						dp[i, j-1, k],      # Insert into s1
						dp[i-1, j-1, k-1]   # Replace
					)
		cost[i-1] = dp[i][-1][-1] 
	return dp, cost

class EditDistance_Fixed:
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
		_, bottleneck = _fill_table(self.s1,s2,self.m)
		# self.print_table(dp,s2)
		return bottleneck

	def fill_table_cost(self, s2) -> None:
		"""Fill the DP table bottom-up."""
		dp, cost = _fill_table_cost(self.s1,s2,self.m)
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
	s1, s2 = "GGGACGTTACGTCCC", "ACGTACGT"

	dp_obj = EditDistance_Fixed(s1=s1, align_mod="semi_global")
	# cost = dp_obj.fill_table_cost(s2)
	bn = dp_obj.fill_table(s2)
	# print(cost)
	print(bn)


	s1, s2 = "GTGCTAC", "ACT" 
	dp_obj = EditDistance_Fixed(s1=s1, align_mod="semi_global")
	cost = dp_obj.fill_table_cost(s2)
	bn = dp_obj.fill_table(s2)
	print(cost)
	print(bn)