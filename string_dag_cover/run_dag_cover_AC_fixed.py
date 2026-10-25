import argparse
import itertools
import logging
import os
import pickle
import sys
import time, datetime

import graph_tool
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  # for workflow.scripts.*
sys.path.insert(0, scripts_dir)   # for src.*, utils.*
from line_profiler import profile
from preprocess_subgraphs import preprocess, preprocess
from src.AC_DFA import AhoCorasickDFA
from src.product_graph import ProductGraph
from utils.logging import LevelFormatter
from utils.utils import get_bubbles, load_gfa
from quickmers import fixed_length_levenshtein_ball_iterator
logger = logging.getLogger(__name__)

@profile
def eval_product_graph(set_can, g, sink_node):
	ac = AhoCorasickDFA(set_can, sigma=["A","C","G","T"])
	prod_graph = ProductGraph(g,ac, sink_node)
	path_cover_complete, res_path = prod_graph.check_path_cover((int(g.vertices().next()), 0), sink_node)
	return path_cover_complete, prod_graph, ac, res_path


def expected_graph_cover(g):
	l = g.vp['v_label']
	path_kmers = {}
	path_iter = graph_tool.topology.all_paths(g, 0, 39)
	c = 1
	for p in path_iter:
		path_seq = "".join([l[v] for v in p])
		path_kmers[c] = [path_seq[i:i+20] for i in range(len(path_seq)-20+1)]
		print(len(path_kmers[c]))
		c+=1
	unique_pattern_sets = list(itertools.product(path_kmers[1], path_kmers[2]))
	logger.info(unique_pattern_sets[:10])
	logger.info(f"Expected number of sets of patterns covering SNP: {len(unique_pattern_sets)}")
	all = path_kmers[1] + path_kmers[2]
	all_combs = list(itertools.combinations(all, 2))
	logger.info(f"All combinations: {len(all_combs)}")
	for p in unique_pattern_sets:
		if p in all_combs:
			all_combs.remove(p)
		else:
			all_combs.remove((p[1],p[0]))
	logger.info(len(all_combs))
			
	return unique_pattern_sets, all_combs
		

@profile
def process_subgraph(subgraph, ts_candidates, h_sink_node, max_edit):
	data = {"string" : [], "sg_covered": []}
	for ts in ts_candidates:
		edit_neighbourhood = fixed_length_levenshtein_ball_iterator(ts, max_edit, 0)
		ac_ls = AhoCorasickDFA(edit_neighbourhood,word=ts)
		prod_graph = ProductGraph(subgraph,ac_ls, h_sink_node)
		path_covered = prod_graph.check_path_cover_faster((subgraph.get_vertices()[0], 0))
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
 
	utc_plus_2 = datetime.timezone(datetime.timedelta(hours=2))
	profile.output_prefix = f"profiler_out/AC_fixed/{b_s}_{datetime.datetime.now(tz=utc_plus_2).strftime('%Y%m%d_%H%M%S')}"

	
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

	subgraphs_df = pd.read_csv(sg_file)
	new_bubbl = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["bubble_end"]))
	start_ids_bubb = dict(zip(subgraphs_df["bubble_start"], subgraphs_df["prefix_node"]))
		
	load_bubble_time = time.perf_counter_ns() - load_bubble_start
	logger.info(f"Graph loading: {load_time/1_000_000_000:6f} s, Bubble loading: {load_bubble_time/1_000_000_000:6f}")

	b_e = new_bubbl.get(b_s, None)
	subgraph, ts_candidates, h_sink_node, num_paths, sg_times = preprocess(graph, start_ids_bubb ,b_s,b_e,bubbles, k)
	# subgraph, ts_candidates, h_sink_node, num_paths, sg_times = preprocess(graph, start_ids_bubb ,b_s,b_e,bubbles, k)
	logger.info(f"# GFA ID: {b_s}")
	logger.info(f"# candidate: {len(ts_candidates)}")
	logger.info(f"# paths: {num_paths}")
	logger.info(f"# max_edit: {max_edit}")
	data = process_subgraph(subgraph, ts_candidates, h_sink_node, max_edit)
	data['editball'] = max_edit
	df = pd.DataFrame(data)
	df = df.pivot(columns="editball", index="string", values="sg_covered")
	df.to_csv(f"results/{gene_tag}/AC_fixed/result_{b_s}_{max_edit}.csv")
	logger.info("Script Done.")
