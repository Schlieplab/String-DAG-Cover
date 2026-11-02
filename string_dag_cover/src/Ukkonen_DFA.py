from collections import deque


class Ukkonen_DFA:
	def __init__(self, word, max_edits,sigma = "ACGT"):
		self.word = word
		self.alphabet = list(sigma) 
		self.max_edits = max_edits

		self.states = []          # state id -> DP row
		self.delta = []          # matrix of states x symbols, value is the new state
		self.alpha_map = {"A":0, "C": 1, "G":2, "T": 3}
		self.final_states = set()
		self.start_state = None
		self._build()

	# --- construction -------------------------------------------------------

	def _step(self, row, symbol):
		"""One DP step: the row for the input extended by `symbol`."""
		new = [0]
		for i, ch in enumerate(self.word):
			cost = 0 if ch == symbol else 1       
			new.append(min(new[i] + 1,            # insertion
						row[i] + cost,         # match / substitution
						row[i + 1] + 1))       # deletion
		cap = self.max_edits + 1                  # values above k are equivalent
		return tuple(min(x, cap) for x in new)

	def _build(self):
		k = self.max_edits
		ids = {}
		queue = deque()

		def get_id(row):
			if row not in ids:
				ids[row] = len(self.states)
				self.states.append(row)
				self.delta.append([0,0,0,0])
				if row[-1] <= k:
					self.final_states.add(ids[row])
				queue.append(row)
			return ids[row]

		start = tuple(min(i, k + 1) for i in range(len(self.word) + 1))
		self.start_state = get_id(start)
		while queue:
			row = queue.popleft()
			src = ids[row]
			for sym in self.alphabet:
				self.delta[src][self.alpha_map[sym]] = get_id(self._step(row, sym))

	def __repr__(self):
		n_trans = sum(len(t) for t in self.delta)
		return (f"LevenshteinDFA(word={self.word!r}, max_edits={self.max_edits}, "
				f"states={len(self.states)}, "
				f"transitions={n_trans}, final={len(self.final_states)})")

	def _check(self, symbol):
		if symbol not in self.alphabet:
			raise ValueError(f"symbol {symbol!r} is not in the alphabet")

	def transition(self, state, symbol):
		"""Next state id via the full delta, or None for the implicit dead state."""
		self._check(symbol)
		if state is None:
			return None
		return self.delta[state][self.alpha_map[symbol]]

	def accepts(self, text):
		"""Anchored mode: is the whole text within max_edits of the word?"""
		state = self.start_state
		for symbol in text:
			state = self.transition(state, symbol)
			if state is None:
				return False
		return state in self.final_states


if __name__ == "__main__":
	dfa = Ukkonen_DFA("CTT", 0)
	print(dfa, "\n")

	print("start state:", dfa.start_state)
	print("final states:", sorted(dfa.final_states), "\n")

	# print("delta:")
	# for s in range(len(dfa.states)):
	#     mark = "*" if s in dfa.final_states else " "
	#     print(f" {mark}{s:>2} {dfa.states[s]}")
	#     for sym, t in dfa.delta[s].items():
	#         print(f"        --{sym!r}--> {t}")

	# print()
	# text = "ACTTCTTGCTT"
	# print(f"\nmatches of 'CTT' ending at: {list(dfa.find_ends(text, use_fail_links=True))}"
	#       f"  in {text!r}")

	# AC_view(dfa.delta, dfa.states,dfa.start_state, dfa.final_states,
	#         file_name="test_ls_dfa_full_delta",
	#         node_attr={"fontsize": "20"},
	#         edge_attr={"fontsize": "20pt"},
	#     )
	

	# checks 2
	for e in range(4):
		dfa = Ukkonen_DFA("AACAAGATACTTAACTCTTG", e)
		print(dfa, "\n")
		# pprint(dfa.goto)
		# for s in range(len(dfa.states)):
		#     mark = "*" if s in dfa.final_states else " "
		#     print(f" {mark}{s:>2} {dfa.states[s]}")
		#     for sym, t in dfa.delta[s].items():
		#         print(f"        --{sym!r}--> {t}")

	
