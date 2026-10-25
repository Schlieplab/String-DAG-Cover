import os, sys, argparse, logging, random, time
import numpy as np
# from dynamic_programming.workflow.scripts.load_GFA import load_gfa
import graph_tool as gt
from graphviz import Digraph
from numba import jit

@jit
def _fill_table(top_order, oligo, a_indices, a_indptr, node_labels, align_mode):
	n = len(oligo)
	m = len(top_order)
	dp = np.zeros((m+1, n+1), dtype=np.int32) 
	cost = np.zeros(m, dtype=np.int32)
	bn = np.full(m, n, dtype=np.int32)
	# Fill the rest of the table
	if align_mode == "global":
		for i in range(m+1):
			dp[i][0] = i
		for j in range(n+1):
			dp[0][j] = j
	elif align_mode == "semi_global":
		for j in range(n+1):
					dp[0][j] = j
	for v in top_order:
		in_neighbours = a_indices[a_indptr[v]: a_indptr[v+1]]

		if len(in_neighbours) == 0:
			for i in range(1, n+1):
				if node_labels[v] == oligo[i-1]:
					dp[v+1][i] = dp[0][i-1]     # Match
				else:
					dp[v+1][i] = 1 + min(
						dp[0][i],      # Delete from s1
						dp[v+1][i-1],      # Insert into s1
						dp[0][i-1]  # Mismatch
					)
			cost[v] = dp[v+1][-1]
		else:
			if len(in_neighbours) == 1:
				dp_neighbours = dp[in_neighbours+1,:].reshape((1,-1))
			else:
				dp_neighbours = dp[in_neighbours+1,:]
			for i in range(1, n+1):
				max_mm, nb_mm_ind = 0, 0
				max_ins, nb_ins_ind = 0, 0
				for idx in range(len(dp_neighbours)):
					if dp_neighbours[idx,i-1] > max_mm:
						max_mm = dp_neighbours[idx,i-1]
						nb_mm_ind = idx
					if dp_neighbours[idx,i] > max_ins:
						max_ins = dp_neighbours[idx,i]
						nb_ins_ind = idx
				if node_labels[v] == oligo[i-1]:
					dp[v+1][i] = dp[in_neighbours[nb_mm_ind]+1][i-1]     # Match
				else:
					dp[v+1][i] = 1 + min(
						dp[in_neighbours[nb_ins_ind]+1][i],      # Delete from s1
						dp[v+1][i-1],      # Insert into s1
						dp[in_neighbours[nb_mm_ind]+1][i-1]  # Mismatch
					)
			cost[v] = dp[v+1][-1]
		if len(in_neighbours) != 0:
			bn[v] = min([cost[v], max([bn[vi] for vi in in_neighbours])])
	return cost, bn


class LosslessFilter_DAG_Cover():
	def __init__(self, top_order: list, g):
		self.top_order = top_order
		self.m = len(top_order)
		self.g = g
		self.adjacency = gt.spectral.adjacency(g)
		self.a_data = self.adjacency.data
		self.a_indices = self.adjacency.indices
		self.a_indptr = self.adjacency.indptr
		self.v_label = g.vp['v_label']
		self.labels = [self.v_label[n] for n in range(len(top_order))]


	def fill_table(self, oligo, align_mode="semi_global") -> None:
		cost, bn = _fill_table(self.top_order, oligo, self.a_indices,self.a_indptr , self.labels, align_mode=align_mode)
		# self.print_table(dp, len(oligo), oligo)
		return cost, bn

	def print_table(self, dp, n, oligo) -> None:
		"""Pretty-print the DP table for debugging/visualisation."""

		# self.fill_table()
		header = "\t" + "\t".join(["_"] + list(oligo))
		print(header)
		for i in range(self.m + 1):
			row_label = "_: _" if i == 0 else f"{self.top_order[i-1]}: {self.v_label[self.top_order[i-1]]}"
			print(f"{row_label}\t", end="")
			print("\t".join(str(dp[i][j]) for j in range(n + 1)))


def view_graph(graph, until_node: int, focus_n: int, top_o, file_name: str,
		file_format,
		node_attr,
		edge_attr):
		if np.where(top_o == focus_n)[0][0] <= np.where(top_o == until_node)[0][0]:
			raise ValueError("focus_n must come after until_node in top_o to visualize the graph correctly.")
		dot = Digraph(
			name=file_name,
			format=file_format,
			node_attr=node_attr,
			edge_attr=edge_attr,
		)
		dot.graph_attr["rankdir"] = "LR"
		v_label = graph.vp['v_label']
		idx , v = 0, top_order[0]
		while True:
			if v == until_node:
				break
			dot.node(str(v), v_label[v], shape="circle", color="#147899", style="filled", fontcolor="white")
			idx +=1
			v = top_order[idx]
			
		while True:
			if v == focus_n:
				break
			dot.node(str(v), v_label[v], shape="circle", color="#FA9E19", style="filled", fontcolor="white")
			idx +=1
			v = top_order[idx]

		while True:
			dot.node(str(v), v_label[v], shape="circle")
			idx +=1
			if idx >= len(top_order):
				break
			v = top_order[idx]
			
		# for v in top_o:
		#     if int(v) <= until_node:
		#         dot.node(str(v), v_label[v], shape="circle", color="#147899", style="filled", fontcolor="white")
		#     elif int(v) == focus_n:
		#         dot.node(str(v), v_label[v], shape="circle", color="#FA9E19", style="filled", fontcolor="white")
		#     else:
		#         dot.node(str(v), v_label[v], shape="circle")

		for e in graph.edges():
			# for l, s in d.items():
			# dot.edge(str(v_label[e.source()]), str(v_label[e.target()]))
			dot.edge(str(e.source()), str(e.target()))
		
		dot.render()


def view_graph2(graph, node_colors, top_o,  file_name: str,cyclic=False,
		file_format = "pdf",
		node_attr = None,
		edge_attr = None):
		dot = Digraph(
			name=file_name,
			format=file_format,
			node_attr=node_attr,
			edge_attr=edge_attr,
		)
		style_mapping = dict()
		for col, d in node_colors.items():
			for n in d['nodes']:
				style_mapping[n] = {"col": col, "font_col": d.get("font_col", "black")}
		dot.graph_attr["rankdir"] = "LR"
		v_label = graph.vp['v_label']
		if cyclic:
			for v in graph.vertices():
				if int(v) in style_mapping:
					dot.node(str(v), f"{int(v)}: {v_label[v]}", shape="circle", color=style_mapping[v]['col'], style="filled", fontcolor=style_mapping[v]['font_col'])
				else:
					dot.node(str(v), f"{int(v)}: {v_label[v]}", shape="circle")
		else:       
			for v in top_o:
				if v in style_mapping:
					dot.node(str(v), f"{int(v)}: {v_label[v]}", shape="circle", color=style_mapping[v]['col'], style="filled", fontcolor=style_mapping[v]['font_col'])
				else:
					dot.node(str(v), f"{int(v)}: {v_label[v]}", shape="circle")

		for e in graph.edges():
			# for l, s in d.items():
			# dot.edge(str(v_label[e.source()]), str(v_label[e.target()]))
			dot.edge(str(e.source()), str(e.target()))
		
		dot.render()


if __name__ == '__main__':
	

	logging.basicConfig(
		# force=True,
		level=logging.INFO,
		stream=sys.stdout,
		format='### INFO - %(asctime)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	logging.info(f"{sys.argv[0]} starting up")

	# graph, top_order = load_gfa("../../prototypes/dummy_graph.gfa", sort=True)

	# oligo = "ACT" # developed with this example
	# oligo = "ACTGCTTTT"
	# v_label = graph.vp['v_label']

	# all_paths = gt.topology.all_paths(graph, 0, 11)
	# for p in all_paths:
	# 	print(''.join([v_label[i] for i in p]))
	# 	print("----------------------------")
	# print(top_order)
	# print([v_label[i] for i in top_order])
	# # print([graph.vertex_index[i] for i in graph.vertices()])
	# # print([v_label[i] for i in graph.vertices()])
	
	# dp_obj = BestWorstTargetSite_DP(top_order, graph)
	# # exit()
	# c, bn = dp_obj.fill_table(oligo, "global")
	# print(c[top_order])
	# print(bn[top_order])
	# # print(dp_obj.dp)
	# unt_n = 3
	# focus_n = 4
	# # view_graph2(prod_graph.graph, f"GHC_star_dummy", nodes_color= res[2], node_attr={'fontsize': '25'}, edge_attr={'fontsize': '9pt'})

	# for i in range(len(top_order)-1):
	# 	view_graph(graph, top_order[i],top_order[i+1],top_order, f"dummy_graph_{top_order[i]}_{top_order[i+1]}", file_format="png",edge_attr={"color": "black", "len": ".4"}, node_attr={"color": "lightblue", "style": "filled"})