import logging, time, sys
from utils.utils import extract_subgraph, get_targetsites, uncompress_nodes
logger = logging.getLogger(__name__)

def preprocess(graph, start_ids_bubb, b_s, b_e, bubbles, ts_len, add_sink=True):
	logger.info(f"Extract Bubble as subgraph {b_s}, {b_e} (prefix: {start_ids_bubb[b_s]})")  
	start_ts = time.perf_counter_ns()
	subgraph = extract_subgraph(graph, start_ids_bubb[b_s], b_e, ts_len)
	extract_time = time.perf_counter_ns() - start_ts

	logger.info("Get Target sites")
	start_ts = time.perf_counter_ns()
	ts_candidates, num_paths = get_targetsites(subgraph, start_ids_bubb[b_s], b_e, bubbles, ts_len)
	ts_time = time.perf_counter_ns() - start_ts
	logger.info(f" # target site candidates: {len(ts_candidates)}")  
	logger.info(f" # paths: {num_paths}")  
	logger.info("Decompress nodes")  
	start_ts = time.perf_counter_ns()
	subgraph, b_s_idx = uncompress_nodes(subgraph, start_ids_bubb[b_s], b_e, b_s, draw=False )
	logger.info(f"Original bubble source node in uncompressed graph: {b_s_idx}")  
	h_sink_node = None
	if add_sink:
		sink_nodes = [v for v in subgraph.vertices() if v.out_degree() == 0]
		if len(sink_nodes) > 1:
			logger.error("Subgraph has more than one sink node")
			sys.exit(1)
		node_label = subgraph.vp['v_label']
		n = subgraph.add_vertex()
		node_label[n] = "$"
		subgraph.add_edge(sink_nodes[0],n)
		h_sink_node = int(n)
	uncompress_time = time.perf_counter_ns() - start_ts
	return subgraph, ts_candidates, h_sink_node, num_paths,[extract_time, ts_time, uncompress_time]


