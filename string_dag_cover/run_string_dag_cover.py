import argparse
import itertools
import logging
import os
import random
import sys
import time, pickle, datetime

import graph_tool
import graph_tool.all as gt
import pandas as pd
from quickmers import levenshtein_ball_iterator
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  # for workflow.scripts.*
sys.path.insert(0, scripts_dir)   # for src.*, utils.*
from src.AC_DFA import AhoCorasickDFA
from src.Ukkonen_DFA import Ukkonen_DFA
from src.product_graph import ProductGraph

from utils.logging import LevelFormatter
from utils.utils import load_gfa, get_bubbles
from preprocess_subgraphs import preprocess

logger = logging.getLogger(__name__)	


def build_DFA(method, ts, max_edit):
	if method == "Ukkonen":
		dfa = Ukkonen_DFA(ts, max_edit)
	elif method == "AC":
		edit_nbh = levenshtein_ball_iterator(ts, max_edit, 0)
		dfa = AhoCorasickDFA(edit_nbh, word=ts)
	return dfa

def process_subgraph(subgraph, ts_candidates, h_sink_node, max_edit):
	data = {"string" : [], "sg_covered": []}
	for ts in ts_candidates:
		ac_ls = Ukkonen_DFA(ts, max_edit)
		prod_graph = ProductGraph(subgraph,ac_ls, h_sink_node)
		path_covered = prod_graph.check_path_cover_faster((int(subgraph.vertices().next()), 0))
		if not path_covered:
			data['string'].append(ts)
			data['sg_covered'].append(None)
		else:
			data['string'].append(ts)
			data['sg_covered'].append(max_edit)		
	return data
	

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
	parser.add_argument('--debug', action='store_true',
						help="Enable debug logging")
	args = parser.parse_args()


	graph_file = os.path.expanduser(args.graph)
	max_edit = args.max_edit
	k = args.kmer_length
	sg_file = os.path.expanduser(args.sg_file)
	bubble_file = os.path.expanduser(args.bubble_file)
	b_s = args.b_start
	handler = logging.StreamHandler(sys.stderr)
	debug = args.debug

	handler.setFormatter(LevelFormatter())
	handler.flush()

	
	if debug:
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.INFO)
	logger.addHandler(handler)
	sys.stdout.flush()
	sys.stderr.flush()

	logger.info(f"{sys.argv[0]} starting up")
	gene_tag = graph_file.split("/")[1]

	load_start = time.perf_counter_ns()
	graph = load_gfa(graph_file, sort=False)
	load_time = time.perf_counter_ns() - load_start
	load_bubble_start = time.perf_counter_ns()
	bubbles = get_bubbles(bubble_file)
	utc_plus_2 = datetime.timezone(datetime.timedelta(hours=2))
	
	subgraphs_df = pd.read_csv(sg_file)
	new_bubbl = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["bubble_end"]))
	start_ids_bubb = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["prefix_node"]))
		
	load_bubble_time = time.perf_counter_ns() - load_bubble_start
	logger.info(f"Graph loading: {load_time/1_000_000_000:6f} s, Bubble loading: {load_bubble_time/1_000_000_000:6f}")

	b_e = new_bubbl.get(b_s, None)
	subgraph, ts_candidates, h_sink_node, num_paths, sg_times = preprocess(graph, start_ids_bubb ,b_s,b_e,bubbles, k)
	logger.info(f"# GFA ID: {b_s}, # candidate: {len(ts_candidates)}, # paths: {num_paths}, # max_edit: {max_edit}")
	data = process_subgraph(subgraph, ts_candidates, h_sink_node, max_edit)
	data['editball'] = max_edit
	df = pd.DataFrame(data)
	df = df.pivot(columns="editball", index="string", values="sg_covered")
	df.to_csv(f"results/{gene_tag}/LSA/result_{b_s}_{max_edit}.csv")
	logger.info("Script Done.")
