"""Shared setup for the run_<method>.py scripts: argument parsing, logging,
loading the graph/bubbles and writing results to results/<gene_tag>/<method>/."""
import argparse
import logging
import os
import sys
import time

import pandas as pd

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, scripts_dir)   # for src.*, utils.*
from preprocess_subgraphs import preprocess
from utils.logging import LevelFormatter
from utils.utils import get_bubbles, load_gfa

logger = logging.getLogger(__name__)


def parse_args(description, with_max_edit=True):
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument(dest="graph",
						type=str,
						help="Graph file in GFA format (reindexed)")
	if with_max_edit:
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
	return parser.parse_args()


def setup_logging():
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(LevelFormatter())
	root = logging.getLogger()
	root.setLevel(logging.INFO)
	root.addHandler(handler)


def load_subgraph(args, add_sink=True):
	"""Load graph and bubbles and extract the subgraph of bubble ``args.b_start``.

	Returns gene_tag, subgraph, ts_candidates, h_sink_node, num_paths.
	"""
	graph_file = os.path.expanduser(args.graph)
	sg_file = os.path.expanduser(args.sg_file)
	bubble_file = os.path.expanduser(args.bubble_file)
	b_s = args.b_start
	gene_tag = os.path.basename(os.path.dirname(graph_file))

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
	subgraph, ts_candidates, h_sink_node, num_paths, _ = preprocess(graph, start_ids_bubb, b_s, b_e, bubbles, args.kmer_length, add_sink=add_sink)
	logger.info(f"# GFA ID: {b_s}, # candidate: {len(ts_candidates)}, # paths: {num_paths}")
	return gene_tag, subgraph, ts_candidates, h_sink_node, num_paths


def write_result(df, gene_tag, method, file_name):
	"""Write ``df`` to <project_root>/results/<gene_tag>/<method>/<file_name>, creating missing folders."""
	out_path = f"{project_root}/results/{gene_tag}/{method}/{file_name}"
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	df.to_csv(out_path)
	logger.info(f"Written: {out_path}")


def run_product_graph(method, build_dfa, description):
	"""Run the product graph DAG cover check with the automaton from ``build_dfa(ts, max_edit)``."""
	from src.product_graph import ProductGraph

	args = parse_args(description)
	setup_logging()
	logger.info(f"{sys.argv[0]} starting up")
	gene_tag, subgraph, ts_candidates, h_sink_node, _ = load_subgraph(args)
	max_edit = args.max_edit
	logger.info(f"# max_edit: {max_edit}")

	data = {"string" : [], "sg_covered": []}
	for ts in ts_candidates:
		prod_graph = ProductGraph(subgraph, build_dfa(ts, max_edit), h_sink_node)
		path_covered = prod_graph.check_path_cover_faster((int(subgraph.vertices().next()), 0))
		data['string'].append(ts)
		data['sg_covered'].append(max_edit if path_covered else None)
	data['editball'] = max_edit
	df = pd.DataFrame(data)
	df = df.pivot(columns="editball", index="string", values="sg_covered")
	write_result(df, gene_tag, method, f"result_{args.b_start}_{max_edit}.csv")
	logger.info("Script Done.")
