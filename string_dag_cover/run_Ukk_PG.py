from run_common import run_product_graph
from src.Ukkonen_DFA import Ukkonen_DFA


if __name__ == '__main__':
	run_product_graph(
		"Ukk_PG",
		lambda ts, max_edit: Ukkonen_DFA(ts, max_edit),
		"Evaluate the target site path cover on the product graph "
		"of a bubble subgraph and the Ukkonen Levenshtein automaton",
	)
