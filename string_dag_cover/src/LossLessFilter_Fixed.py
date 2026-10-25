import argparse
import logging
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prototypes_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
import graph_tool as gt
from graph_tool.all import *

# from dynamic_programming.workflow.scripts.load_GFA import load_gfa
from numba import jit


@jit
def _fill_table(a_indices, a_indptr, top_order, v_label, oligo, out_indices, out_indptr,):
	n = len(oligo)
	m = len(top_order)
	dp = np.zeros((m+1, n+1, n+1), dtype=np.int32) 
	cost = np.zeros(m, dtype=np.int32)
	bottleneck = np.full(m, n, dtype=np.int32)  

	for k in range(n+1):
		dp[:,k,0] = k # init first column of all nodes
		dp[0,k,:] = k # init first row of all letters in oligo and empty letter in subgraph
		dp[1:,0,k] = k # init all paths in empty letters 

	# for i in range(1,n):
	# 	dp[top_order[i-1]+1,0,i:] = i
	temp_k = 1
	nodes = [top_order[0]]
	seen = set()
	while temp_k <= n:
		temp_nodes = set()
		for node in nodes:
			if node not in seen:
				dp[node+1,0,temp_k:] = temp_k
				out_neighbours = out_indices[out_indptr[node]:out_indptr[node + 1]]
				temp_nodes.update(out_neighbours)
				seen.add(node)
		temp_k += 1
		nodes = list(temp_nodes)

	# fill dp
	for v in top_order:
		node_label = v_label[v]
		in_neighbours = a_indices[a_indptr[v]: a_indptr[v+1]]
		if len(in_neighbours) == 0:
			in_neighbours = np.array([-1],dtype=np.int32)
			dp_neighbours = dp[0,:,:].copy().reshape((1,-1,n+1))
		else:
			if len(in_neighbours) == 1:
				dp_neighbours = dp[in_neighbours+1,:,:].copy().reshape((1,-1,n+1))
			else:
				dp_neighbours = dp[in_neighbours+1,:,:]

		for i in range(1, n+1):
			if node_label == oligo[i-1]:
				for k in range(1, n+1):
					max_nb_val, max_nb_idx= 0,0
					for idx in range(len(dp_neighbours)):
						if dp_neighbours[idx,i-1, k-1] > max_nb_val:
							max_nb_val = dp_neighbours[idx,i-1, k-1]
							max_nb_idx = idx
					dp[v+1,i,k] = dp[in_neighbours[max_nb_idx]+1,i-1,k-1]                                # Match
			else:
				for k in range(1, n+1):
					max_del, max_del_neigh = 0,0
					max_mm, max_mm_neigh= 0,0
					for idx in range(len(dp_neighbours)):
						if dp_neighbours[idx,i,k-1] > max_del:
							max_del = dp_neighbours[idx,i,k-1]
							max_del_neigh = idx
						if dp_neighbours[idx,i-1,k-1] > max_mm:
							max_mm = dp_neighbours[idx,i-1,k-1]
							max_mm_neigh = idx

					dp[v+1][i][k] = 1 + min(
						dp[in_neighbours[max_del_neigh]+1][i][k-1],    # Delete from s1
						dp[v+1][i-1][k],                               # Insert into s1
						dp[in_neighbours[max_mm_neigh]+1][i-1][k-1]    # Mismatch
					)
		cost[v] = dp[v+1][-1][-1]
		if len(in_neighbours) != 0:
			bottleneck[v] = min([cost[v], max([bottleneck[vi] for vi in in_neighbours])])
	return bottleneck


@jit
def _fill_table_cost(a_indices, a_indptr, top_order, v_label, oligo, out_indices, out_indptr,):
	n = len(oligo)
	m = len(top_order)
	dp = np.zeros((m+1, n+1, n+1), dtype=np.int32) 
	cost = np.zeros(m, dtype=np.int32)
	bottleneck = np.full(m, n, dtype=np.int32)  

	for k in range(n+1):
		dp[:,k,0] = k # init first column of all nodes
		dp[0,k,:] = k # init first row of all letters in oligo and empty letter in subgraph
		dp[1:,0,k] = k # init all paths in empty letters 

	# for i in range(1,n):
	# 	dp[top_order[i-1]+1,0,i:] = i
	temp_k = 1
	nodes = [top_order[0]]
	seen = set()
	while temp_k <= n:
		temp_nodes = set()
		for node in nodes:
			if node not in seen:
				dp[node+1,0,temp_k:] = temp_k
				out_neighbours = out_indices[out_indptr[node]:out_indptr[node + 1]]
				temp_nodes.update(out_neighbours)
				seen.add(node)
		temp_k += 1
		nodes = list(temp_nodes)

	# fill dp
	for v in top_order:
		node_label = v_label[v]
		in_neighbours = a_indices[a_indptr[v]: a_indptr[v+1]]
		if len(in_neighbours) == 0:
			in_neighbours = np.array([-1],dtype=np.int32)
			dp_neighbours = dp[0,:,:].copy().reshape((1,-1,n+1))
		else:
			if len(in_neighbours) == 1:
				dp_neighbours = dp[in_neighbours+1,:,:].copy().reshape((1,-1,n+1))
			else:
				dp_neighbours = dp[in_neighbours+1,:,:]

		for i in range(1, n+1):
			if node_label == oligo[i-1]:
				for k in range(1, n+1):
					max_nb_val, max_nb_idx= 0,0
					for idx in range(len(dp_neighbours)):
						if dp_neighbours[idx,i-1, k-1] > max_nb_val:
							max_nb_val = dp_neighbours[idx,i-1, k-1]
							max_nb_idx = idx
					dp[v+1,i,k] = dp[in_neighbours[max_nb_idx]+1,i-1,k-1]                                # Match
			else:
				for k in range(1, n+1):
					max_del, max_del_neigh = 0,0
					max_mm, max_mm_neigh= 0,0
					for idx in range(len(dp_neighbours)):
						if dp_neighbours[idx,i,k-1] > max_del:
							max_del = dp_neighbours[idx,i,k-1]
							max_del_neigh = idx
						if dp_neighbours[idx,i-1,k] > max_mm:
							max_mm = dp_neighbours[idx,i-1,k]
							max_mm_neigh = idx

					dp[v+1][i][k] = 1 + min(
						dp[in_neighbours[max_del_neigh]+1][i][k-1],    # Delete from s1
						dp[v+1][i-1][k],                               # Insert into s1
						dp[in_neighbours[max_mm_neigh]+1][i-1][k-1]    # Mismatch
					)
		cost[v] = dp[v+1][-1][-1]
		if len(in_neighbours) != 0:
			bottleneck[v] = min([cost[v], max([bottleneck[vi] for vi in in_neighbours])])
	return bottleneck, cost, dp


class LosslessFilter_DAG_Cover_Fixed():
	def __init__(self, g, origin_b_s, top_order):
		self.m = len(top_order)
		self.g = g
		self.adjacency = gt.spectral.adjacency(g)
		self.a_data = self.adjacency.data
		self.a_indices = self.adjacency.indices
		self.a_indptr = self.adjacency.indptr
		self.v_label = g.vp['v_label']
		self.labels = [self.v_label[n] for n in range(len(top_order))]
		self.top_order = top_order
		self.out_neighbours = self.adjacency.tocsc()
	

	def fill_table(self, oligo) -> None:
		bottleneck = _fill_table(self.a_indices,self.a_indptr , self.top_order, self.labels, oligo, self.out_neighbours.indices, self.out_neighbours.indptr)
		return bottleneck

	def fill_table_cost(self, oligo) -> None:
		bottleneck, cost, dp = _fill_table_cost(self.a_indices,self.a_indptr , self.top_order, self.labels, oligo, self.out_neighbours.indices, self.out_neighbours.indptr)
		return bottleneck, cost, dp

	def print_table(self, dp, oligo, v_labels, filen=None) -> None:
		"""Pretty-print the DP table for debugging/visualisation."""

		header = "       " + "           ".join(["_"] + list(oligo))
		m, n, _ = dp.shape
		if filen:
			header = ",".join(["_"] + list(oligo))
			with open(filen, "w") as f:
				f.write(header)
				f.write(f"_: _, ")
				f.write(",".join(str(dp[0][j].tolist()) for j in range(n))+ "\n")
				for i in self.top_order:
					# row_label = "_" if i == 0 else self.nodes_labels[i - 1]
					f.write(f"{i}: {v_labels[i]}, ")
					f.write(",".join(str(dp[i+1][j].tolist()) for j in range(n))+ "\n")
		else:
			print(header)
			for i in range(m):
				row_label = "_" if i == 0 else self.nodes_labels[i - 1]
				print(f"{row_label}  ", end="")
				print("  ".join(str(dp[i][j]) for j in range(n)))


if __name__ == '__main__':
	

	logging.basicConfig(
		# force=True,
		level=logging.INFO,
		stream=sys.stdout,
		format='### INFO - %(asctime)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	logging.info(f"{sys.argv[0]} starting up")

	# # DEBUG COMPELX NODE
	# from workflow.scripts.load_GFA import load_gfa
	# from quickmers import levenshtein_distance

	# graph = gt.load_graph("../prototypes/22_subgraph.gt")

	# oligo = "AAAGTGAACGGTGTGGGGAA"
	# # oligo = "AGGGAAGGGAAGCCATTAAA"

	# dp_obj = BestWorstTargetSite_SL_DP(graph)
	# dp =dp_obj.compute_forward_backward(oligo)


	# print("###########################")
	# print(oligo)
	# print(dp_obj.cost[dp_obj.top_order].tolist())
	# print(dp_obj.bottleneck[dp_obj.top_order].tolist())
	# print(dp_obj.top_order.tolist())
	# print(np.min(dp_obj.bottleneck[dp_obj.top_order]), dp_obj.top_order[np.argmin(dp_obj.bottleneck[dp_obj.top_order])])
	# dp_obj.print_table(dp, oligo, filen=f"22_{oligo}_subgraph_dp_table.csv")
	# paths = []
	# # nodes_interest = set()
	# # res = get_nodes_until_distance(graph, node_i, 20, set(), nodes_interest)
	# # print(nodes_interest)
	# # for n in nodes_interest:
	# for p in gt.topology.all_paths(graph, 49, dp_obj.top_order[np.argmin(dp_obj.bottleneck[dp_obj.top_order])]):
	#     paths.append(p)
	# v_label = graph.vp['v_label']
	# print([v_label[i] for i in dp_obj.top_order])
	# options = set()
	# for p in paths:
	#     options.add(''.join([v_label[i] for i in p])[-20:])
	#     # if len(''.join([v_label[i] for i in p])) != 20:
	#     #     print("PATH WITH LENGTH NOT 20: ", ''.join([v_label[i] for i in p]))
	#     # print(''.join([v_label[i] for i in p])[-20:])
	#     # print("----------------------------")
	# print(options)
	# dists = []
	# for opt in options:
	#     if levenshtein_distance(opt, oligo) < 15:
	#         dists.append(levenshtein_distance(opt, oligo))
	# print(max(dists))
	#EXAMPLE DEBUG TOY

	# graph, top_order = load_gfa("../../prototypes/dummy_graph.gfa", sort=True)
	# label = graph.vp['v_label']

	# graph_draw(graph, vertex_text=label, output=f"dummy_graph_nt.svg", ink_scale=0.7)

	# oligo = "ACT" # developed with this example
	# # oligo = "ACTGCTTTT"

	# # all_paths = gt.topology.all_paths(graph, 0, 15)
	# # for p in all_paths:
	# #     print(''.join([v_label[i] for i in p]))
	# #     print("----------------------------")
	# # print(top_order)
	# # print([v_label[i] for i in top_order])
	# # # print([graph.vertex_index[i] for i in graph.vertices()])
	# # # print([v_label[i] for i in graph.vertices()])
	
	# dp_obj = BestWorstTargetSite_SL_DP(graph, 0, top_order)
	# # exit()
	# # dp_obj.print_table()
	# print(dp_obj.top_order)
	# cost, bn, dp = dp_obj.fill_table(oligo)
	# dp_obj.print_table(dp, oligo)
	# # print(dp_obj.dp)