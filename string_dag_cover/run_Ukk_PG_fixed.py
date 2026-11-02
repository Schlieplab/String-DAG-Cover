from run_common import run_product_graph
from src.Ukkonen_DFA_Fixed import Ukk_DFA_Fixed


if __name__ == '__main__':
	run_product_graph(
		"Ukk_PG_fixed",
		lambda ts, max_edit: Ukk_DFA_Fixed(ts, max_edit, True),
		"Evaluate the target site path cover on the product graph "
		"of a bubble subgraph and the fixed-length Ukkonen Levenshtein automaton",
	)
