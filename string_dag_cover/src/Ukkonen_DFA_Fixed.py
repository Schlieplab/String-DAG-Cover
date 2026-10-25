from collections import deque
import sys

class Ukk_DFA_Fixed:
	def __init__(self, word, max_edits, length_restricted=False, sigma = "ACGT"):
		self.word = word
		self.alphabet = list(sigma) 
		self.max_edits = max_edits

		self.states = []          # state id -> DP row
		self.delta = []          # matrix of states x symbols, value is the new state
		self.alpha_map = {"A":0, "C": 1, "G":2, "T": 3}
		self.final_states = set()
		self.start_state = None
		if length_restricted:
			self._build_unanchored_fixed_length_dfa()
			# self._build_equal_length()
		else:
			self._build()

	def _build(self):
		"""This function builds an unanchored variable length Levenshtein automaton
		"""
		cap = self.max_edits + 1
		ids = {}
		queue = deque()
		def _step(row, symbol):
			"""One DP step: the row for the input extended by `symbol`."""
			new = [0]
			for i, ch in enumerate(self.word):
				cost = 0 if ch == symbol else 1       
				new.append(min(new[i] + 1,            # insertion
							row[i] + cost,         # match / substitution
							row[i + 1] + 1))       # deletion
			return tuple(min(x, cap) for x in new)

		def get_id(row):
			if row not in ids:
				ids[row] = len(self.states)
				self.states.append(row)
				self.delta.append([0,0,0,0])
				if row[-1] <= self.max_edits:
					self.final_states.add(ids[row])
				queue.append(row)
			return ids[row]

		start = tuple(min(i, cap) for i in range(len(self.word) + 1))
		self.start_state = get_id(start)
		while queue:
			row = queue.popleft()
			src = ids[row]
			for sym in self.alphabet:
				self.delta[src][self.alpha_map[sym]] = get_id(_step(row, sym))

	def _build_equal_length(self):
		"""This function builds an anchored fixed length Levenshtein automaton
		"""
		max_len, cap = len(self.word), self.max_edits +1
		ids = {}
		queue = deque()
		
		def prune(row, L):
			return tuple(x if x + abs(i - L) <= self.max_edits else cap for i, x in enumerate(row))

		def get_id(key):
			if key not in ids:
				ids[key] = len(self.states)
				self.states.append(key)
				self.delta.append([0,0,0,0])
				row, L = key
				if L == max_len and row[-1] <= self.max_edits:
					self.final_states.add(ids[key])
				queue.append(key)
			return ids[key]

		def step_equal_length(row, symbol, l):
			"""One DP step: the row for the input extended by `symbol`."""
			new = [row[0] + 1]
			for i, ch in enumerate(self.word):
				new.append(min(new[i] + 1,            # insertion
							row[i] + (ch != symbol),         # match / substitution
							row[i + 1] + 1))       # deletion
			return prune(tuple(min(x, cap) for x in new), l)

		start = tuple(min(i, cap) for i in range(max_len + 1))
		self.start_state = get_id((prune(start, 0), 0))
		while queue:
			row, L = queue.popleft()
			if L == max_len:
				continue
			src = ids[(row, L)]
			for sym in self.alphabet:
				r = step_equal_length(row, sym, L+1)
				if min(r) <= self.max_edits:
					self.delta[src][self.alpha_map[sym]] = get_id((r, L +1))
	
	def _build_unanchored_fixed_length_dfa(self):
		"""This function builds an unanchored fixed length Levenshtein automaton
		"""
		n, cap = len(self.word), self.max_edits + 1

		def prune(row, L):
			return tuple(x if x + abs(i - L) <= self.max_edits else cap for i, x in enumerate(row))

		def step(row, symbol, L):
			new = [row[0] + 1]
			for i, ch in enumerate(self.word):
				new.append(min(new[i] + 1, row[i] + (ch != symbol), row[i + 1] + 1))
			return prune(tuple(min(x, cap) for x in new), L)

		fresh = (prune(tuple(min(i, cap) for i in range(n + 1)), 0), 0) # form (start_row,L),L))

		def advance(S, a):
			nxt = set()
			for row, L in S | {fresh}:              # start (row,l)-tuple added to each state therefore unanchored
				if L == n:
					continue                         # fixed length reached
				r = step(row, a, L + 1)
				if min(r) <= self.max_edits:         # state alive, dead state is all values above max edit
					nxt.add((r, L + 1))
			return frozenset(nxt)

		def is_final(S):
			return any(L == n and row[-1] <= self.max_edits for row, L in S) #length + edits must match

		ids = {}
		start = frozenset()
		ids[start] = 0
		self.states.append(start)
		self.delta.append([None] * len(self.alphabet))
		queue = deque([start])
		while queue:
			S = queue.popleft()
			for j, a in enumerate(self.alphabet):
				T = advance(S, a)
				if T not in ids:
					ids[T] = len(self.states)
					self.states.append(T)
					self.delta.append([None] * len(self.alphabet))
					if is_final(T):
						self.final_states.add(ids[T])
					queue.append(T)
				self.delta[ids[S]][j] = ids[T]

	def __repr__(self):
		n_trans = sum(len(t) for t in self.delta)
		return (f"LevenshteinDFA(word={self.word!r}, max_edits={self.max_edits}, "
				f"states={len(self.states)}, "
				f"transitions={n_trans}, final={len(self.final_states)})")

	def _check(self, symbol):
		if symbol not in self.alphabet:
			raise ValueError(f"symbol {symbol!r} is not in the alphabet")

	def transition(self, state, symbol):
		"""Next state id via the full delta, or None for the implicit dead state.
		This function doesn't work for unanchored, fixed length automaton
		"""
		self._check(symbol)
		if state is None:
			return None
		return self.delta[state][self.alpha_map[symbol]]

	def accepts(self, text):
		"""For Anchored mode: is the whole text within max_edits of the word?
		This function doesn't return correct results for unanchored, fixed length automaton
		"""
		state = self.start_state
		for symbol in text:
			state = self.transition(state, symbol)
			if state is None:
				return False
		return state in self.final_states

	def write_delta_table(self, file_path: str) -> None:
		"""Write the delta transition function to a tab-separated table.

		Rows are states, columns are alphabet symbols (sorted), and cell
		values are the target states.

		Args:
			file_path: Path to the output file
		"""
		sorted_sigma = sorted(self.alphabet)
		with open(file_path, 'w') as f:
			f.write("State\t" + "\t".join(sorted_sigma) + "\n")
			for idx, trans in enumerate(self.delta):
				row = [f"*{idx}" if idx in self.final_states else str(idx)] + [str(t) for t in trans]
				f.write("\t".join(row) + "\n")


if __name__ == "__main__":
	dfa = Ukk_DFA_Fixed("CGG", 1)
	print(dfa, "\n")

	print("start state:", dfa.start_state)
	print("final states:", sorted(dfa.final_states), "\n")

	print(dfa.accepts("CGG"))
	print(dfa.accepts("CGGA"))
	print(dfa.accepts("CG"))
	print(dfa.accepts("CGA"))
	print(dfa.accepts("ACGGT"))
	print(dfa.accepts("ACGG"))

	dfa = Ukk_DFA_Fixed("CGG", 1, True)
	print(dfa, "\n")

	print("start state:", dfa.start_state)
	print("final states:", sorted(dfa.final_states), "\n")

	print(dfa.accepts("CGG"))
	print(dfa.accepts("CGGA"))
	print(dfa.accepts("CG"))
	print(dfa.accepts("CGGT"))
	print(dfa.accepts("ACGT"))

	# CHECK 2
	for e in range(4):
		dfa = Ukk_DFA_Fixed("AACAAGATACTTAACTCTTG", e)
		print(dfa, "\n")


	
