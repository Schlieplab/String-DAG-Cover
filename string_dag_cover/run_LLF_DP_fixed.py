import logging
import sys

import graph_tool as gt
import pandas as pd

from run_common import load_subgraph, parse_args, setup_logging, write_result
from src.LossLessFilter_Fixed import LosslessFilter_DAG_Cover_Fixed

logger = logging.getLogger(__name__)


if __name__ == '__main__':
	args = parse_args("Compute the bottleneck value per target site of a bubble "
					  "subgraph with the fixed-length lossless filter DP", with_max_edit=False)
	setup_logging()
	logger.info(f"{sys.argv[0]} starting up")
	gene_tag, subgraph, ts_candidates, _, _ = load_subgraph(args, add_sink=False)

	top_order = gt.topology.topological_sort(subgraph)
	dp_obj = LosslessFilter_DAG_Cover_Fixed(subgraph, None, top_order)
	bn_values = []
	for ts in ts_candidates:
		bottleneck = dp_obj.fill_table(ts)
		bn_values.append(bottleneck[top_order[-1]])

	df = pd.DataFrame({"ts_candidates": ts_candidates, "bn_value": bn_values})
	write_result(df.set_index("ts_candidates"), gene_tag, "LLF_DP_fixed", f"result_{args.b_start}.csv")
	logger.info("Script Done.")
