import argparse
import datetime
import logging
import os
import sys
import time

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)                                     # for string_dag_cover.*
sys.path.insert(0, os.path.join(project_root, "string_dag_cover"))  # for src.*, utils.* inside the package
sys.path.insert(0, benchmark_dir)                                    # for baseline.*
from quickmers import levenshtein_ball_iterator

from string_dag_cover.preprocess_subgraphs import preprocess
from string_dag_cover.src.AC_DFA import AhoCorasickDFA
from string_dag_cover.src.product_graph import ProductGraph
from string_dag_cover.utils.logging import LevelFormatter
from string_dag_cover.utils.utils import get_bubbles, load_gfa

logger = logging.getLogger(__name__)

def process_subgraph( subgraph,ts_candidates,h_sink_node, max_edit):
	logger.info(f"Process {len(ts_candidates)} target sites with DAG Cover.")  
	data = {"string" : [], "sg_covered": []}
	st_time = time.perf_counter_ns()
	eb_time = 0
	ac_time = 0
	pg_time = 0
	cover_check_time = 0
	for ts in ts_candidates:
		timer_start = time.perf_counter_ns()
		ts_edball = levenshtein_ball_iterator(ts, max_edit, 0)
		eb_time += (time.perf_counter_ns() - timer_start)
		timer_start = time.perf_counter_ns()
		ac_ls = AhoCorasickDFA(ts_edball, word=ts)
		ac_time += (time.perf_counter_ns() - timer_start)
		timer_start = time.perf_counter_ns()
		prod_graph = ProductGraph(subgraph,ac_ls, h_sink_node, full_numba=True)
		pg_time += (time.perf_counter_ns() - timer_start)
		timer_start = time.perf_counter_ns()
		path_covered = prod_graph.check_path_cover_faster((int(subgraph.vertices().next()), 0))
		cover_check_time += (time.perf_counter_ns() - timer_start)
		if not path_covered:
			data['string'].append(ts)
			data['sg_covered'].append(None)
		else:
			data['string'].append(ts)
			data['sg_covered'].append(max_edit)
	st_end = time.perf_counter_ns() - st_time
	return data, [st_end, eb_time, ac_time, pg_time, cover_check_time]
	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description="Evaluate the target site path cover on the product graph "
					"of a bubble subgraph and its Aho-Corasick automaton"
	)
	parser.add_argument(dest="graph",
						type=str,
						help="Graph file in GFA format (reindexed)")
	parser.add_argument(dest="max_edit",
						type=int,
						help="max edit operations for Levenshtein automaton")
	parser.add_argument(dest="sg_file",
						type=str,
						help="Defined sub graphs to process in csv format (start and end id)")
	parser.add_argument(dest="bubble_file",
						type=str,
						help="Extracted bubbles to process in csv format (start and end id)")
	parser.add_argument('--kmer_length', "-k",
						type=int,
						default=20,
						help="length of k-mer")
	parser.add_argument('--b_start', '-s',
						type=int,
						help="Start node id of the bubble to evaluate")
	parser.add_argument('--single', action='store_true',
						help="only stated max_edit, not for <= max edit")
	args = parser.parse_args()


	graph_file = os.path.expanduser(args.graph)
	max_edit = args.max_edit
	k = args.kmer_length
	sg_file = os.path.expanduser(args.sg_file)
	bubble_file = os.path.expanduser(args.bubble_file)
	b_s = args.b_start
	handler = logging.StreamHandler(sys.stderr)
	singular_run = args.single
	numba_f = False
	handler.setFormatter(LevelFormatter())
	handler.flush()

	
	logger.setLevel(logging.INFO)
	logger.addHandler(handler)
	sys.stdout.flush()
	sys.stderr.flush()

	logger.info(f"{sys.argv[0]} starting up")
	gene_tag = os.path.basename(os.path.dirname(graph_file))
	logger.info(f"Gene tag folder : {gene_tag}")

	bm_file = f"{project_root}/benchmark/results/{gene_tag}/AC_PG.csv"
	os.makedirs(os.path.dirname(bm_file), exist_ok=True)

	load_start = time.perf_counter_ns()
	graph = load_gfa(graph_file, sort=False)
	load_time = time.perf_counter_ns() - load_start
	load_bubble_start = time.perf_counter_ns()
	bubbles = get_bubbles(bubble_file)
	subgraphs_df = pd.read_csv(sg_file)
	new_bubbl = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["bubble_end"]))
	start_ids_bubb = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["prefix_node"]))
	
	load_bubble_time = time.perf_counter_ns() - load_bubble_start
	logger.info(f"Graph loading: {load_time/1_000_000_000:6f} s, Bubble loading: {load_bubble_time/1_000_000_000:6f}")

	b_e = new_bubbl.get(b_s, None)
	
	subgraph, ts_candidates, h_sink_node, num_paths, sg_times = preprocess(graph, start_ids_bubb ,b_s,b_e,bubbles, k)
	graph_length = subgraph.num_vertices()
	num_ts = len(ts_candidates)
	edit_range = [max_edit] if singular_run else range(max_edit+1)
	res_dfs = []
	for m_e in edit_range:
		logger.info(f"GFA ID: {b_s},# max_edit: {m_e}, graph length: {graph_length}, # candidate: {len(ts_candidates)}, # paths: {num_paths}")

		res, times = process_subgraph(subgraph, ts_candidates, h_sink_node, m_e)

		res['editball'] = m_e
		res_dfs.append(pd.DataFrame(res))
	
		if os.path.exists(bm_file):
			with open(bm_file, "a") as file:
				file.write(f"{datetime.datetime.now(tz=datetime.timezone.utc)},{graph_length},{b_s},{num_ts},{num_paths},{m_e},{sg_times[0]},{sg_times[1]},{sg_times[2]},{times[0]},{times[1]},{times[2]},{times[3]},{times[4]}\n")
		else:
			with open(bm_file, "w") as file:
				file.write("ts_utc,graph_length,gfs_start,num_ts,num_paths,max_edit,extract_time,ts_time,uncompress_time,dag_cover_time,eb_time,ac_time,pg_time,check_cover_time\n")
				file.write(f"{datetime.datetime.now(tz=datetime.timezone.utc)},{graph_length},{b_s},{num_ts},{num_paths},{m_e},{sg_times[0]},{sg_times[1]},{sg_times[2]},{times[0]},{times[1]},{times[2]},{times[3]},{times[4]}\n")

	df = pd.concat(res_dfs, ignore_index=True)
	df = df.pivot(columns="editball", index="string", values="sg_covered")
	suffix = "_results_singular" if singular_run else "_results"
	out_path = f"{project_root}/results/{gene_tag}/AC_PG/{b_s}{suffix}.csv"
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	df.to_csv(out_path)
	logger.info("Script Done.")
