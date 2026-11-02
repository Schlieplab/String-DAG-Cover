from quickmers import fixed_length_levenshtein_ball_iterator

from run_common import run_product_graph
from src.AC_DFA import AhoCorasickDFA


if __name__ == '__main__':
	run_product_graph(
		"AC_PG_fixed",
		lambda ts, max_edit: AhoCorasickDFA(fixed_length_levenshtein_ball_iterator(ts, max_edit, 0), word=ts),
		"Evaluate the target site path cover on the product graph "
		"of a bubble subgraph and the Aho-Corasick automaton of the fixed-length edit ball",
	)
