import os, sys, logging, time
import graph_tool as gt
import itertools
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prototypes_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)   # for src.*
sys.path.insert(0, prototypes_dir) # for load_GFA
print(project_root)
print(prototypes_dir)
from graphviz import Digraph
from line_profiler import profile
from numba import njit,types
from utils.utils import load_gfa
from numba.typed import Dict
from collections import defaultdict, deque

@njit
def build_csr(src, dst, n_nodes):
	n_edges = len(src)

	# 1. Count out-degree per node
	indptr = np.zeros(n_nodes + 1, dtype=np.int64)
	for i in range(n_edges):
		indptr[src[i] + 1] += 1

	# 2. Prefix sum turns counts into start offsets
	for u in range(n_nodes):
		indptr[u + 1] += indptr[u]

	# 3. Place each edge's target in its source's slot
	indices = np.empty(n_edges, dtype=np.int64)
	fill = indptr[:-1].copy()               # next free position per node
	for i in range(n_edges):
		u = src[i]
		indices[fill[u]] = dst[i]
		fill[u] += 1

	return indptr, indices

@njit
def _build_with_jit(delta, edges, node_labels):
	pg_nodes = set()
	pg_edges = []
	base_to_int = {"A":0, "C": 1, "G":2, "T": 3}
	for h_edge in edges: # |E| x |O|
		for state in range(len(delta)):
			pg_edges.append(((h_edge[0],state), (h_edge[1], delta[state][base_to_int[node_labels[h_edge[0]]]])))
			pg_nodes.add((h_edge[0],state))
			pg_nodes.add((h_edge[1], delta[state][base_to_int[node_labels[h_edge[0]]]]))
	return list(pg_nodes), pg_edges


@njit
def _build_csr_with_jit(src_arr, dst_arr, n_nodes):
	n_edges = len(src_arr)
	# 1. Count out-degree per node
	indptr = np.zeros(n_nodes + 1, dtype=np.int64)
	for i in range(n_edges):
		indptr[src_arr[i] + 1] += 1

	# 2. Prefix sum turns counts into start offsets
	for u in range(n_nodes):
		indptr[u + 1] += indptr[u]

	# 3. Place each edge's target in its source's slot
	indices = np.empty(n_edges, dtype=np.int64)
	fill = indptr[:-1].copy()               # next free position per node
	for i in range(n_edges):
		u = src_arr[i]
		indices[fill[u]] = dst_arr[i]
		fill[u] += 1
	return indptr, indices

@njit(cache=True)
def _lazy_build_jit(start_id, delta, label_ids, is_final, h_indptr, h_indices):
	"""BFS over product nodes encoded as ``u * n_states + state``.

	Returns ``(order, indptr, indices)``: ``order[i]`` is the encoded product
	node with compact id ``i`` (BFS discovery order) and ``indptr``/``indices``
	is the CSR over compact ids. Nodes are popped in compact-id order, so the
	CSR is filled directly while traversing.
	"""
	n_states = delta.shape[0]
	# encoded node -> compact id; replaces the visited set. A typed Dict beats a
	# dense n_h_nodes * n_states array here, since usually only a small part of
	# the product graph is reachable (dense array: page faults dominate).
	compact = Dict.empty(key_type=types.int64, value_type=types.int64)
	# every node is enqueued at most once -> no ring buffer wrap-around needed;
	# queue/indptr/indices grow by doubling, bounded by the reachable part
	q_cap = 16
	queue = np.empty(q_cap, dtype=np.int64)
	indptr = np.zeros(q_cap + 1, dtype=np.int64)
	e_cap = max(16, h_indices.shape[0])
	indices = np.empty(e_cap, dtype=np.int64)
	n_edges = 0

	compact[start_id] = 0
	queue[0] = start_id
	head = 0
	tail = 1
	while head < tail:
		pid = queue[head]
		u = pid // n_states
		next_state = delta[pid % n_states, label_ids[u]]
		if not is_final[next_state]:
			for k in range(h_indptr[u], h_indptr[u + 1]):
				target = h_indices[k] * n_states + next_state
				c = compact.get(target, -1)
				if c == -1:
					if tail == q_cap:
						q_cap *= 2
						grown_q = np.empty(q_cap, dtype=np.int64)
						grown_q[:tail] = queue[:tail]
						queue = grown_q
						grown_p = np.zeros(q_cap + 1, dtype=np.int64)
						grown_p[:head + 1] = indptr[:head + 1]
						indptr = grown_p
					c = tail
					compact[target] = c
					queue[tail] = target
					tail += 1
				if n_edges == e_cap:
					e_cap *= 2
					grown_e = np.empty(e_cap, dtype=np.int64)
					grown_e[:n_edges] = indices[:n_edges]
					indices = grown_e
				indices[n_edges] = c
				n_edges += 1
		head += 1
		indptr[head] = n_edges
	return queue[:tail].copy(), indptr[:tail + 1].copy(), indices[:n_edges].copy()


@njit(cache=True)
def _full_build_jit(delta, label_ids, is_final, h_indptr, h_indices):
	"""Full product graph without final-state nodes, encoded as ``u * n_states + state``.

	Same node/edge set as _build_with_jit() + remove_finite_states_edges().
	Returns ``(order, indptr, indices)`` like _lazy_build_jit(); compact ids
	follow the encoded order (host node major, state minor). The whole product
	is touched here, so a dense array is used instead of a dict.
	"""
	n_states = delta.shape[0]
	n_h = label_ids.shape[0]
	present = np.zeros(n_h * n_states, dtype=np.bool_)
	n_edges = 0
	# pass 1: mark nodes and count edges. Like the original, a target is a
	# node even when its edge is dropped because the source state is final.
	for u in range(n_h):
		start, end = h_indptr[u], h_indptr[u + 1]
		if start == end:
			continue
		for state in range(n_states):
			next_state = delta[state, label_ids[u]]
			if not is_final[state]:
				present[u * n_states + state] = True
			if is_final[next_state]:
				continue
			for k in range(start, end):
				present[h_indices[k] * n_states + next_state] = True
			if not is_final[state]:
				n_edges += end - start

	# compact ids in encoded order
	compact = np.full(n_h * n_states, -1, dtype=np.int64)
	n_nodes = 0
	for pid in range(n_h * n_states):
		if present[pid]:
			compact[pid] = n_nodes
			n_nodes += 1
	order = np.empty(n_nodes, dtype=np.int64)
	for pid in range(n_h * n_states):
		if present[pid]:
			order[compact[pid]] = pid

	# pass 2: fill CSR; sources come in compact-id order (u major, state minor)
	indptr = np.zeros(n_nodes + 1, dtype=np.int64)
	indices = np.empty(n_edges, dtype=np.int64)
	n_edges = 0
	for u in range(n_h):
		start, end = h_indptr[u], h_indptr[u + 1]
		for state in range(n_states):
			pid = u * n_states + state
			if not present[pid]:
				continue
			next_state = delta[state, label_ids[u]]
			if start != end and not is_final[next_state]:
				for k in range(start, end):
					indices[n_edges] = compact[h_indices[k] * n_states + next_state]
					n_edges += 1
			indptr[compact[pid] + 1] = n_edges
	return order, indptr, indices


class ProductGraph():
	@profile
	def __init__(self, graph, automaton, sink_node, lazy_build=False, full_numba=False):
		self.h_nodes = graph.get_vertices()
		self.h_edges = graph.get_edges()
		self.h_node_label = graph.vp['v_label']
		self.p_nodes = []
		self.p_edges = []
		self.output_states = automaton.final_states
		self.num_states = len(automaton.delta)
		self.delta = np.array(automaton.delta, dtype=np.int64)
		self.sink_node = sink_node
		self.alpha_map = {"A":0, "C": 1, "G":2, "T": 3}
		
		if lazy_build:
			self.h_adjacency = gt.spectral.adjacency(graph).T.tocsr()
			self.h_indptr = self.h_adjacency.indptr
			self.h_indices = self.h_adjacency.indices
			if lazy_build == "numba":
				
				self._lazy_build_numba()  # also sets node_to_id and the CSR
				return
			self._lazy_build()
		elif full_numba:
			self.h_adjacency = gt.spectral.adjacency(graph).T.tocsr()
			self.h_indptr = self.h_adjacency.indptr
			self.h_indices = self.h_adjacency.indices
			self._build_numba()  # also sets node_to_id and the CSR
			return
		else:
			self._build_fast()
			self.remove_finite_states_edges()
		self.create_csr_numba()

	# @profile
	def _build(self):
		self.p_nodes = list(itertools.product(self.h_nodes, list(range(self.num_states))))
		for h_edge in self.h_edges: # |E| x |O|
			for state in range(len(self.delta)):
				self.p_edges.append(((h_edge[0],state), (h_edge[1], self.delta[state][self.alpha_map[self.h_node_label[h_edge[0]]]])))

	# @profile
	def _build_fast(self):
		node_labels = [self.h_node_label[v] for v in self.h_nodes]
		self.p_nodes, self.p_edges = _build_with_jit(delta=self.delta,edges=self.h_edges,node_labels=node_labels)

	# @profile
	def _lazy_build(self, start=(0, 0)):
		"""Build only the part of the product graph reachable from ``start``.

		BFS over (host node, state) pairs; nodes in a final state are not
		added, so the result needs no remove_finite_states_edges().
		"""
		label_ids = {int(v): self.alpha_map.get(self.h_node_label[v], -1) for v in self.h_nodes}

		start = (int(start[0]), int(start[1]))
		if start[1] in self.output_states:
			self.p_nodes, self.p_edges = [], []
			return
		visited = {start}
		p_edges = []
		queue = deque([start])
		while queue:
			u, state = queue.popleft()
			next_state = int(self.delta[state][label_ids[u]])
			if next_state in self.output_states:
				continue
			for v in self.h_indices[self.h_indptr[u]:self.h_indptr[u + 1]]:
				target = (int(v), next_state)
				p_edges.append(((u, state), target))
				if target not in visited:
					visited.add(target)
					queue.append(target)
		self.p_nodes = list(visited)
		self.p_edges = p_edges

	def _numba_inputs(self):
		# the "$" sink is not in alpha_map; its label is never used (no out-edges)
		label_ids = np.array([self.alpha_map.get(self.h_node_label[v], -1) for v in self.h_nodes], dtype=np.int64)
		is_final = np.zeros(self.num_states, dtype=np.bool_)
		is_final[list(self.output_states)] = True
		return label_ids, is_final, self.h_indptr.astype(np.int64), self.h_indices.astype(np.int64)

	def _set_from_csr(self, order, indptr, indices, edge_list):
		"""Set p_nodes, node_to_id, CSR (and p_edges) from a numba kernel result."""
		n_states = self.num_states
		self.csr_indptr, self.csr_indices = indptr, indices
		self.p_nodes = list(zip((order // n_states).tolist(), (order % n_states).tolist()))
		self.node_to_id = {node: idx for idx, node in enumerate(self.p_nodes)}
		if not edge_list:
			self.p_edges = []
			return
		src = np.repeat(np.arange(len(self.p_nodes)), np.diff(indptr))
		self.p_edges = [(self.p_nodes[s], self.p_nodes[t]) for s, t in zip(src.tolist(), indices.tolist())]

	# @profile
	def _lazy_build_numba(self, start=(0, 0), edge_list=True):
		"""Numba version of _lazy_build().

		Also sets node_to_id, csr_indptr and csr_indices, so no
		create_csr_numba() is needed afterwards. p_edges is only needed for
		plotting; edge_list=False skips building it (the costly Python part).
		"""
		label_ids, is_final, h_indptr, h_indices = self._numba_inputs()
		start = (int(start[0]), int(start[1]))
		if is_final[start[1]]:
			self._set_from_csr(np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64),
							   np.zeros(0, dtype=np.int64), edge_list)
			return
		result = _lazy_build_jit(start[0] * self.num_states + start[1], self.delta,
								 label_ids, is_final, h_indptr, h_indices)
		self._set_from_csr(*result, edge_list)

	# @profile
	def _build_numba(self, edge_list=True):
		"""Numba version of _build_fast() + remove_finite_states_edges().

		Also sets node_to_id, csr_indptr and csr_indices, see _lazy_build_numba().
		"""
		result = _full_build_jit(self.delta, *self._numba_inputs())
		self._set_from_csr(*result, edge_list)


	# @profile
	def remove_finite_states_edges(self):
		# nodes_to_del = {node for node in self.p_nodes if node[1] in self.output_states}
		self.p_nodes = [node for node in self.p_nodes if node[1] not in self.output_states]
		self.p_edges = [edge for edge in self.p_edges if edge[0][1] not in self.output_states and edge[1][1] not in self.output_states]

		
	# @profile
	def create_csr(self):
		"""Return CSR adjacency arrays and mappings for tuple-valued nodes.

		Node order is ``self.nodes``. The return value is
		``(indptr, indices, node_to_id, id_to_node)``.
		"""
		node_to_id = {node: idx for idx, node in enumerate(self.p_nodes)}
		adjacency = [[] for _ in range(len(self.p_nodes))]
		for source, target in self.p_edges:
			if source in node_to_id and target in node_to_id:
				adjacency[node_to_id[source]].append(node_to_id[target])

		indptr = np.zeros(len(self.p_nodes) + 1, dtype=np.int64)
		for idx, neighbors in enumerate(adjacency):
			indptr[idx + 1] = indptr[idx] + len(neighbors)
		indices = np.fromiter(
			(neighbor for neighbors in adjacency for neighbor in neighbors),
			dtype=np.int64,
			count=int(indptr[-1]),
		)
		self.node_to_id = node_to_id
		self.csr_indptr = indptr
		self.csr_indices = indices

	# @profile
	def create_csr_numba(self):
		self.node_to_id = {node: idx for idx, node in enumerate(self.p_nodes)}
		src_arr = np.array([self.node_to_id[u] for u, _ in self.p_edges], dtype=np.int64)  # shape (E,)
		dst_arr = np.array([self.node_to_id[v] for _, v in self.p_edges], dtype=np.int64)  # shape (E,)
		self.csr_indptr, self.csr_indices = _build_csr_with_jit(src_arr, dst_arr, len(self.p_nodes))


	# @profile
	def check_path_cover_faster(self, s):
		inds = [self.node_to_id[s]]
		while len(inds) !=0:
			temp_edges = []
			for ind in inds:
				edges = self.csr_indices[self.csr_indptr[ind]:self.csr_indptr[ind + 1]]
				for sink_n_ind in edges:
					if self.p_nodes[sink_n_ind][0] ==self.sink_node:
						return False
				temp_edges.extend(edges)
			inds = list(set(temp_edges))
		return True

	def view(self, file_name: str, file_format = "png", node_attr= None, edge_attr= None):
		"""Visualize product graph 

		Args:
			file_name (str): output filename
			file_format (Literal[&quot;svg&quot;, &quot;png&quot;], optional): _description_. Defaults to "png".
			node_attr (Optional[Dict[str, str]], optional): according to graphviz. Defaults to None.
			edge_attr (Optional[Dict[str, str]], optional): according to graphviz. Defaults to None.
		"""

		dot = self._create_base_graph(file_name, file_format, node_attr, edge_attr)
		for node_pair in self.p_edges:
			dot.edge(f"{int(node_pair[0][0])}_{node_pair[0][1]}", f"{int(node_pair[1][0])}_{node_pair[1][1]}", label="")

		dot.render()

	def _create_base_graph(self,  file_name, file_format="png", node_attr = None, edge_attr= None) -> Digraph:
		"""This is a helper function to visualize the product graph

		Args:
			file_name (str): output filename
			file_format (Literal[&quot;svg&quot;, &quot;png&quot;], optional): _description_. Defaults to "png".
			node_attr (Optional[Dict[str, str]], optional): according to graphviz. Defaults to None.
			edge_attr (Optional[Dict[str, str]], optional): according to graphviz. Defaults to None.

		Returns:
			Digraph: graphviz object
		"""
		dot = Digraph(
			name=file_name,
			format=file_format,
			node_attr=node_attr,
			edge_attr=edge_attr,
		)
		dot.graph_attr["rankdir"] = "LR"
		dot.node("", "", shape="plaintext")
		v_label = self.h_node_label
		for idx, n in enumerate(self.p_nodes):
			if n[1] in self.output_states and self.alive[idx]:
				dot.node(f"{int(n[0])}_{n[1]}", f"{int(n[0])}_{n[1]}\n{v_label[n[0]]}", shape="doublecircle")
			elif n[1] in self.output_states and not self.alive[idx]:
				dot.node(f"{int(n[0])}_{n[1]}", f"{int(n[0])}_{n[1]}\n{v_label[n[0]]}", shape="doublecircle", fillcolor="red")
			else:
				dot.node(f"{int(n[0])}_{n[1]}", f"{int(n[0])}_{n[1]}\n{v_label[n[0]]}", shape="circle")

		return dot

	def pltfromendnodes(self, file_name: str, file_format = "png", node_attr= None, edge_attr= None):		
		dot = Digraph(
					name=file_name,
					format=file_format,
					node_attr=node_attr,
					edge_attr=edge_attr,
				)
		dot.graph_attr["rankdir"] = "LR"
		dot.node("", "", shape="plaintext")
		v_label = self.h_node_label
		end_nodes = [n for n in self.p_nodes if n[0] == self.sink_node]
		queue = deque([])
		for end_node in end_nodes:
			in_edges = [e for e in self.p_edges if e[1] == end_node]
			if len(in_edges) == 0:
				continue
			
			if end_node[1] in self.output_states:
				dot.node(f"{end_node[0]}_{end_node[1]}", f"{end_node[0]}_{end_node[1]}\n{v_label[end_node[0]]}", shape="doublecircle")
			else:
				dot.node(f"{end_node[0]}_{end_node[1]}", f"{end_node[0]}_{end_node[1]}\n{v_label[end_node[0]]}", shape="circle")
			
			for edge in in_edges:
				if edge[0][1] in self.output_states:
					dot.node(f"{edge[0][0]}_{edge[0][1]}", f"{edge[0][0]}_{edge[0][1]}\n{v_label[edge[0][0]]}", shape="doublecircle")
				else:
					dot.node(f"{edge[0][0]}_{edge[0][1]}", f"{edge[0][0]}_{edge[0][1]}\n{v_label[edge[0][0]]}", shape="circle")
				dot.edge(f"{edge[0][0]}_{edge[0][1]}", f"{edge[1][0]}_{edge[1][1]}", label="")
				queue.append(edge[0])
			
		while queue:
			node = queue.popleft()
			in_edges = [e for e in self.p_edges if e[1] == node]
			if len(in_edges) == 0:
				continue
			for edge in in_edges:
				if edge[0][1] in self.output_states:
					dot.node(f"{edge[0][0]}_{edge[0][1]}", f"{edge[0][0]}_{edge[0][1]}\n{v_label[edge[0][0]]}", shape="doublecircle")
				else:
					dot.node(f"{edge[0][0]}_{edge[0][1]}", f"{edge[0][0]}_{edge[0][1]}\n{v_label[edge[0][0]]}", shape="circle")
				dot.edge(f"{edge[0][0]}_{edge[0][1]}", f"{edge[1][0]}_{edge[1][1]}", label="")
				queue.append(edge[0])

		dot.render()
  
	def pltfromstartnode(self, file_name: str, file_format = "png", node_attr= None, edge_attr= None):		
		dot = Digraph(
					name=file_name,
					format=file_format,
					node_attr=node_attr,
					edge_attr=edge_attr,
				)
		dot.graph_attr["rankdir"] = "LR"
		v_label = self.h_node_label
		start_node = (0,0)
		queue = deque([])
		out_edges = [e for e in self.p_edges if e[0] == start_node]
		if len(out_edges) == 0:
			return None

		dot.node(f"{start_node[0]}_{start_node[1]}", f"{start_node[0]}_{start_node[1]}\n{v_label[start_node[0]]}", shape="circle")
		
		for edge in out_edges:
			if edge[1][1] in self.output_states:
				dot.node(f"{edge[1][0]}_{edge[1][1]}", f"{edge[1][0]}_{edge[1][1]}\n{v_label[edge[1][0]]}", shape="doublecircle")
			else:
				dot.node(f"{edge[1][0]}_{edge[1][1]}", f"{edge[1][0]}_{edge[1][1]}\n{v_label[edge[1][0]]}", shape="circle")
			dot.edge(f"{edge[0][0]}_{edge[0][1]}", f"{edge[1][0]}_{edge[1][1]}", label="")
			queue.append(edge[1])
		
		while queue:
			node = queue.popleft()
			out_edges = [e for e in self.p_edges if e[0] == node]
			if len(out_edges) == 0:
				continue
			for edge in out_edges:
				if edge[1][1] in self.output_states:
					dot.node(f"{edge[1][0]}_{edge[1][1]}", f"{edge[1][0]}_{edge[1][1]}\n{v_label[edge[1][0]]}", shape="doublecircle")
				else:
					dot.node(f"{edge[1][0]}_{edge[1][1]}", f"{edge[1][0]}_{edge[1][1]}\n{v_label[edge[1][0]]}", shape="circle")
				dot.edge(f"{edge[0][0]}_{edge[0][1]}", f"{edge[1][0]}_{edge[1][1]}", label="")
				queue.append(edge[1])

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

	g, top_order = load_gfa("data/TOY/dummy_graph.gfa", sort=True)
	v_label = g.vp['v_label']
	gt.all.graph_draw(g, vertex_text=v_label, output="dummy_graph.svg", ink_scale=0.7)
	gt.all.graph_draw(g, vertex_text=g.vertex_index, output="dummy_graph_id.svg", ink_scale=0.7)
 
 
	logging.info(top_order)
	adjacency_csr = gt.spectral.adjacency(g).T.tocsr()

 
	patterns = ["CCT", "TGC"]
	# text = "ACGATCGATTTCGTAGC"
	# view_graph(g,-1,-1, top_o=top_order, file_name="H_origin", file_format="png",edge_attr={"color": "black", "len": ".4"}, node_attr={"color": "lightblue", "style": "filled"})
	# gt.all.graph_draw(g, vertex_text=v_label, output="dummy_fool_graph.svg", ink_scale=0.7)

	# paths = list(gt.topology.all_paths(g, 1,9))
	# paths_str = ["".join([v_label[n] for n in p]) for p in paths]

	ac = Levenshtein_DFA_slim("GCC", 1)
	

	prod_graph = ProductGraph(g,ac, top_order[-1])
	# prod_graph.view(True, file_name="dummy_automaton_fail")
	prod_graph.view(file_name="dummy_product_graph")
	st_time= time.perf_counter_ns()
	path_cover_complete = prod_graph.check_path_cover((0, 0))
	logging.info(time.perf_counter_ns()-st_time)
	if not path_cover_complete:
		logging.info(f"Path still existing.")
	st_time= time.perf_counter_ns()
	path_cover_complete = prod_graph.check_path_cover_faster((0, 0))
	logging.info(time.perf_counter_ns()-st_time)
	if not path_cover_complete:
		logging.info(f"2nd impl: Path still existing.")
	else:
		logging.info("huhu")
	logging.info("Script Done.")
 
 
	# check 3
	
