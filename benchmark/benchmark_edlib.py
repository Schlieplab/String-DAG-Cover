import argparse
import datetime
import logging
import os
import sys
import time

import pandas as pd

package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir) 
import edlib
import graph_tool as gt
from graph_tool.all import *

from string_dag_cover.preprocess_subgraphs import preprocess
from string_dag_cover.utils.logging import LevelFormatter
from string_dag_cover.utils.utils import get_bubbles, load_gfa


def process_bubble(subgraph,ts_candidates, b_s, r_code, out_file, sg_times):

	logger.info(f"Process {len(ts_candidates)} target sites with DP.")  
	top_order = gt.topology.topological_sort(subgraph)

	logger.info("Get path iterator")
	v_label = subgraph.vp['v_label']
	st_time = time.perf_counter_ns()
	bns = []
	for ts in ts_candidates:
		num_paths = 0
		max_bn = -1
		for path in  gt.topology.all_paths(subgraph,top_order[0], top_order[-1]):
			num_paths +=1 
			if num_paths % 100_000 == 0:
				logger.info(f"Node: {b_s} Still collecting... {num_paths} paths so far")
			if num_paths > 1_000_000:
				logger.info(f"Path count exploded ({num_paths}), stopping")
				break
			s1 = "".join([v_label[v] for v in path])
			edlib_obj = edlib.align(ts, s1, mode="HW")
			max_bn = max(max_bn, edlib_obj['editDistance'])
		bns.append(max_bn)
	dp_time = time.perf_counter_ns() - st_time
	
	b_path = f"benchmarks/{r_code}/edlib.csv"

	os.makedirs(os.path.dirname(b_path), exist_ok=True)
	if os.path.exists(b_path):
		with open(b_path, "a") as file:
			file.write(f"{datetime.datetime.now(tz=datetime.timezone.utc)},{len(top_order)},{b_s},{len(ts_candidates)},{num_paths},{sg_times[0]},{sg_times[1]},{sg_times[2]},{dp_time}\n")
	else:
		with open(b_path, "w") as file:
			file.write("ts_utc,graph_length,gfs_start,num_ts,num_paths,extract_time,ts_time,uncompress_time,dp_time\n")
			file.write(f"{datetime.datetime.now(tz=datetime.timezone.utc)},{len(top_order)},{b_s},{len(ts_candidates)},{num_paths},{sg_times[0]},{sg_times[1]},{sg_times[2]},{dp_time}\n")

	logger.info(f"Writing output: {out_file}")

	ts_bn_df = pd.DataFrame({
		"ts_candidates": ts_candidates,
		"bn_value": bns,
	})
	ts_bn_df.to_csv(out_file, index=False)
	
	logger.info("Bubble done.")


if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description="Compute target site suitability per bubble or for all bubbles"
	)

	parser.add_argument('--graph', "-g",
						type=str,
						help="Graph file in GFA format")
	parser.add_argument('--bubbles', "-b",
						type=str,
						help="Merged Bubble from bubble finder")
	parser.add_argument('--bubbles_origin', "-r",
						type=str,
						help="Bubble from bubble finder")
	parser.add_argument('--seq_len', '-k',
						type=int,
						help="File to tore meta data")
	parser.add_argument('--b_start', '-s',
						type=int,
						default=-1,
						help="File to tore meta data")
	parser.add_argument('--candidates_out', "-c",
						type=str,
						help="Output file with Overview of candidates for subgraph")

		
	args = parser.parse_args()
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	handler = logging.StreamHandler()
	handler.setFormatter(LevelFormatter())
	logger = logging.getLogger()
	logger.info(f"{sys.argv[0]} starting up")

	graph_file = os.path.expanduser(args.graph)
	sg_file = os.path.expanduser(args.bubbles)
	bubble_raw_file = os.path.expanduser(args.bubbles_origin)
	ts_len = args.seq_len
	candidate_file = args.candidates_out
	b_s = args.b_start
	debug = False

	run = graph_file.split("/")[1]
	load_start = time.perf_counter_ns()
	graph = load_gfa(graph_file, sort=False)
	load_time = time.perf_counter_ns() - load_start
	load_bubble_start = time.perf_counter_ns()
	bubbles = get_bubbles(bubble_raw_file)

	subgraphs_df = pd.read_csv(sg_file)
	new_bubbl = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["bubble_end"]))
	start_ids_bubb = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["prefix_node"]))
	
	load_bubble_time = time.perf_counter_ns() - load_bubble_start
	logger.info(f"Graph loading: {load_time/1_000_000_000:6f} s, Bubble loading: {load_bubble_time/1_000_000_000:6f}")
	if b_s == -1:
		b_s = None # GFA Id
	b_e = new_bubbl.get(b_s, None)

	subgraph, ts_candidates, h_sink_node,num_paths, sg_times = preprocess(graph, start_ids_bubb, b_s, b_e, bubbles,ts_len)

	# res = process_bubble(graph, b_s, b_e, start_ids_bubb, bubbles,ts_len,run,out_file = candidate_file)
	res = process_bubble(subgraph,ts_candidates, b_s, run, candidate_file, sg_times)
	logger.info("THE END.")




