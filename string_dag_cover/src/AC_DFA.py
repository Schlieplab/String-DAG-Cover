import collections, time
from graphviz import Digraph
from quickmers import fixed_length_levenshtein_ball_iterator


def _iter_words(items):
	"""Yield strings from a str, an iterable of str, or an iterable of iterables of str."""
	if items is None:
		return
	if isinstance(items, str):
		yield items
		return
	for item in items:
		if isinstance(item, str):
			yield item
		else:
			yield from item  # e.g. a levenshtein-ball iterator


class AhoCorasickDFA:
	def __init__(self, patterns=None, word=None, sigma="ACGT"):
		self.sigma = set(sigma)

		# goto function; state 0 is the root/initial state. Completed in place
		# into the full transition function by _build_fail_delta.
		self.alpha_map = {ch: idx for idx, ch in enumerate(sorted(self.sigma))}
		self.trie_trans = [[-1] * len(self.alpha_map)]
		self.depth = [0]
		self.final_states = set()
		self.start_state = 0

		for w in _iter_words(word):
			self._add_pattern(w)
		for pattern in _iter_words(patterns):
			self._add_pattern(pattern)

		# states where a pattern ends exactly (before propagation along fail links)
		self.pattern_ends = set(self.final_states)
		self.delta, self.fail = self._build_fail_delta()
		self.states = list(range(len(self.trie_trans)))

	def _add_pattern(self, pattern: str) -> None:
		node = 0
		for ch in pattern:
			if self.trie_trans[node][self.alpha_map[ch]] == -1:
				self.trie_trans.append([-1] * len(self.alpha_map))
				self.depth.append(self.depth[node] + 1)
				self.trie_trans[node][self.alpha_map[ch]] = len(self.trie_trans) - 1
			node = self.trie_trans[node][self.alpha_map[ch]]
		self.final_states.add(node)


	def _build_fail_delta(self):
		n_sigma = len(self.sigma)
		delta = self.trie_trans
		root = 0
		fail = [root] * len(self.trie_trans)
		queue = collections.deque()

		for idx in range(n_sigma):
			if self.trie_trans[root][idx] == -1:
				delta[root][idx] = root
			else:
				fail[delta[root][idx]] = root
				queue.append(delta[root][idx])				
				
		while queue:
			curr = queue.popleft()
			f = fail[curr]
			for ch_idx in range(n_sigma):
				nxt = delta[curr][ch_idx]
				if nxt == -1:
					delta[curr][ch_idx] = delta[f][ch_idx]
				else:
					fail[nxt] = delta[f][ch_idx]
					# a state accepts if any of its suffixes is a pattern
					if fail[nxt] in self.final_states:
						self.final_states.add(nxt)
					queue.append(nxt)
		return delta, fail

	def transition(self, state: int, letter: str) -> int:
		"""Return the next state for ``state`` on ``letter``.

		Falls back to the initial state if ``letter`` is outside the
		alphabet the automaton was built with.
		"""
		try:
			return self.delta[state][self.alpha_map[letter]]
		except KeyError:
			return self.start_state

	def num_prefixes(self, include_empty: bool = True) -> int:
		"""Return the number of distinct prefixes of the patterns.

		Each state corresponds to exactly one distinct prefix shared
		by one or more patterns (``_add_pattern`` only creates a new node
		when that prefix has not been seen before), so this is simply the
		number of states in the trie. The root state represents the empty
		prefix; set ``include_empty=False`` to exclude it.
		"""
		n_prefixes = len(self.trie_trans)
		return n_prefixes if include_empty else n_prefixes - 1

	def search(self, text: str):
		"""Returns list of (end_index, pattern) for all matches."""
		results = []
		state = self.start_state
		for i, ch in enumerate(text):
			state = self.transition(state, ch)
			s = state
			while s != self.start_state:
				if s in self.pattern_ends:
					results.append((i, text[i - self.depth[s] + 1 : i + 1]))
				s = self.fail[s]
		return results

	def view(self, file_name, file_format = "png",node_attr = None, edge_attr = None, show_full_delta = False, include_fail_links = True,) -> None:
		dot = self._create_base_graph(file_name, file_format, node_attr, edge_attr)

		letters = sorted(self.alpha_map, key=self.alpha_map.get)
		if show_full_delta:
			for state, transitions in enumerate(self.delta):
				for letter, nxt in zip(letters, transitions):
					dot.edge(str(state), str(nxt), label=letter)
		else:
			# delta[i][c] is a trie edge iff it leads one level deeper
			for i, d in enumerate(self.delta):
				for letter, s in zip(letters, d):
					if self.depth[s] == self.depth[i] + 1:
						dot.edge(str(i), str(s), label=letter)

			if include_fail_links:
				for idx, fail in enumerate(self.fail):
					if idx != self.start_state:
						dot.edge(f"{idx}", f"{fail}", style="dashed")

		dot.render()

	def write_delta_table(self, file_path: str) -> None:
		"""Write the delta transition function to a tab-separated table.

		Rows are states, columns are alphabet symbols (sorted), and cell
		values are the target states.

		Args:
			file_path: Path to the output file
		"""
		sorted_sigma = sorted(self.sigma)
		with open(file_path, 'w') as f:
			# Write header
			f.write("State\t" + "\t".join(sorted_sigma) + "\n")
			# Write rows
			for idx, trans in enumerate(self.delta):
				row = [f"*{idx}" if idx in self.final_states else str(idx)] + [str(t) for t in trans]
				# print(row)
				f.write("\t".join(row) + "\n")

	def _create_base_graph(self,file_name: str,file_format= "png",node_attr = None,edge_attr = None) -> Digraph:
		"""Create a base graph for visualization.

		Args:
			file_name: Name of the output file
			file_format: Format of the output file (svg or png)
			node_attr: Attributes for nodes in the visualization
			edge_attr: Attributes for edges in the visualization

		Returns:
			A configured Digraph object
		"""
		dot = Digraph(
			filename=file_name,
			format=file_format,
			node_attr=node_attr,
			edge_attr=edge_attr,
		)
		dot.graph_attr["rankdir"] = "LR"
		dot.node("", "", shape="plaintext")

		accepting = {str(state) for state in self.final_states}
		for q in self.states:
			if str(q) in accepting:
				dot.node(str(q), str(q), shape="doublecircle")
			else:
				dot.node(str(q), str(q), shape="circle")

		dot.edge("", str(self.start_state), label="")
		return dot


if __name__ == "__main__":
	# small manual tests / usage examples, not part of the library code
	patterns = ["CCT", "TGC"]
	text = "ACGATCTTGATTTCGTAGC"

	ac = AhoCorasickDFA({"CTT"}, sigma=["A","C","G","T"])

	print("sigma:", ac.sigma)
	print("output states:", ac.final_states)
	print("states:", ac.states)
	for s in ac.states:
		print(f"delta[{s}]:", ac.delta[s])

	word = "CTTGACCTCCGACCAGGCTC"
	editn = fixed_length_levenshtein_ball_iterator(word, 3,0)
	st_time = time.perf_counter_ns()
	ac = AhoCorasickDFA(editn, word=word, sigma=["A","C","G","T"])
	e_time = time.perf_counter_ns()
	print(f"time: {(e_time-st_time)/1_000_000_000:.3f} [s]")
