import inspect
import logging
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import graph_tool.all as gt
import numpy as np

TS_LEN=20
font = "sans"
PRNT_DNT = "\n\t\t\t\t\t\t\t\t\t"
logger = logging.getLogger(__name__)

### IO ####
def load_gfa(gfa_file, sort=False):
	with open(gfa_file, "r") as infile:
		nodes = []
		edges = defaultdict(list)
		for line in infile:
			fields = line.strip().split('\t')
			if fields[0] == "S":
				nodes.append((fields[1], fields[2]))
			elif fields[0] == "L":
				edges[fields[1]].append(fields[3])
	logger.info(f"Loaded GFA file: {gfa_file}")
	logger.info(f"Create Graph: {gfa_file}")
	g = gt.Graph(edges)
	v_label = g.new_vp("string")
	v_gfaID = g.new_vp("int")
	v_seq_len = g.new_vp("int")

	for n in nodes:
		v_label[n[0]] = n[1]
		v_seq_len[n[0]] = len(n[1])
		v_gfaID[n[0]] = int(n[0])


	g.vp["v_label"] = v_label
	g.vp["v_seq_len"] = v_seq_len
	g.vp["v_gfaID"] = v_gfaID

	logger.info("Created Graph")
	isolated = [v for v in g.vertices() if v.out_degree() + v.in_degree() == 0]
	logger.info(f"Number of isolated vertices: {len(isolated)}")
	logger.info(f"Isolated vertices: {isolated}")
	logger.info(f"Seemingly missing node: {isolated}")

	# Remove in reverse order to avoid index shifting
	g.remove_vertex(isolated)

	if sort:
		topology_sort = gt.topological_sort(g)
		logger.info("Topological sort done")
		return g, topology_sort
	else:
		return g

def get_bubbles(file):
	with open(file, "r") as file:
		lines = file.readlines()
		# assuming all are in the same direction

		bubbles = {int(x.strip().split(" ")[0][:-1]): int(x.strip().split(" ")[1][:-1])for x in lines[1:]}
	logger.info(f"Number of Bubbles: {len(bubbles)}")
	return bubbles



def get_node_ids(bubble_id, gr, v_prop):
	g_id = [v for v in  gr.vertices() if v_prop[v] == bubble_id]
	if len(g_id) != 1:
		logger.error(f" {inspect.stack()[1][3]} - No or too many vertices found for bubble start GFA ID {bubble_id}. Check GFA ID property: {g_id}")
		sys.exit(1)
	
	return g_id[0]

def get_targetsites(g, s_node, e_node, o_bbls, ts_len):
	v_GFA_ID = g.vp['v_gfaID']
	v_label = g.vp['v_label']

	origin_bubbles = {k:v for k,v in o_bbls.items() if s_node<= k < e_node}
	# logging.debug(f"Origin bubbles for {s_node} to {e_node}: {len(origin_bubbles)}")
	kmers = set()
	suffixes = []
	num_paths = 1
	b_s = min(origin_bubbles.keys())
	b_e = origin_bubbles.get(b_s,None)
	g_s = get_node_ids(b_s, g, v_GFA_ID)
	suffixes = [v_label[g_s]]
	# logging.info(f" Start seq: {suffixes}")
	while True:
		g_s = get_node_ids(b_s, g, v_GFA_ID)
		g_e = get_node_ids(b_e, g, v_GFA_ID)
		paths = []
		for p in gt.topology.all_paths( g,g_s, g_e):
			paths.append(p)
			if len(paths) % 20000 == 0:
				logger.info(f"Node: {s_node} Still collecting... {len(paths)} paths so far")
			if len(paths) > 300000:
				logger.info("Path count exploded, stopping")
				break
		
		# logging.debug(f"Number of paths for bubble {b_s} to {b_e}: {len(paths)}")
		# logging.debug(f"paths: {paths}")
		# logging.debug(f"suffixes: {len(suffixes)}")

		temp_suff = set()
		for _, p in enumerate(paths):
			seq = "".join([v_label[n] for n in p[1:]])            
			for suf in suffixes:
				temp_seq = suf + seq
				if len(temp_seq) >= ts_len:
					kmers.update([temp_seq[i:i+ts_len] for i in range(len(temp_seq)-ts_len+1)])
					temp_suff.add(temp_seq[-(ts_len-1):])
				else:
					temp_suff.add(temp_seq)

		suffixes = list(temp_suff)   
		num_paths *= len(paths)
		b_s = b_e
		b_e = origin_bubbles.get(b_s,None)
		if not b_e:
			break
	return list(kmers), num_paths

def uncompress_nodes(sub, s_node, e_node, origin_s_node, draw=False, r_code=None):
	"""Converts a sequence-labeled graph into a character-labeled graph.
	"""
	label = sub.vp['v_label']
	v_GFA_ID = sub.vp['v_gfaID']
	sink_node = get_node_ids(e_node, sub,v_GFA_ID)
	origin_source_node = get_node_ids(origin_s_node, sub,v_GFA_ID)
	gr_suffix = label[sink_node]
	
	for v in sub.vertices():
		if v == sink_node:
			v_sub_end = sub.vertex(sink_node) # The node to split
			
			new_v_suffix = sub.add_vertex(len(gr_suffix)-1)
			for idx, new_n in enumerate(new_v_suffix):
				label[new_n] = gr_suffix[idx+1]
				sub.add_edge(v_sub_end, new_n)
				v_sub_end = new_n
			
			label[sink_node] = gr_suffix[0]
		else:
			if len(label[v]) > 2:
				# logging.info(f"Node: {v_GFA_ID[v]}")
				edges_to_remove = list(v.out_edges()) # The node to split
				v_out_neighbours = list(v.out_neighbors()) # Save neighbors
				temp = v

				# logging.info(f"Start expanding prefix or intermediate node")
				new_vs = sub.add_vertex(len(label[v])-1)

				for idx, new_n in enumerate(new_vs):
					label[new_n] = label[v][idx+1]
					sub.add_edge(temp, new_n)
					temp = new_n
				if v == origin_source_node:
					origin_source_node = temp
				for neigh in v_out_neighbours:
					sub.add_edge(temp, neigh)
				label[v] = label[v][0]
				for e in edges_to_remove:
					sub.remove_edge(e)
			elif len(label[v]) > 1:
				# logging.info(f"Node: {v_GFA_ID[v]}")
				edges_to_remove = list(v.out_edges()) # The node to split
				v_out_neighbours = list(v.out_neighbors()) 
				new_vs = sub.add_vertex(len(label[v])-1)

				label[new_vs] = label[v][1]
				label[v] = label[v][0]
				sub.add_edge(v, new_vs)
				if v == origin_source_node:
					origin_source_node = new_vs
				for neigh in v_out_neighbours:
					sub.add_edge(new_vs, neigh)
				for e in edges_to_remove:
					sub.remove_edge(e)
	sub = gt.Graph(sub, prune=True)

	if draw:
		logger.info(f"Draw subgraph, {Path.cwd()}")
		label = sub.vp['v_label']
		if r_code:
			folder = f"bubble_plots/{r_code}"
			Path(folder).mkdir(parents=True, exist_ok=True)
			gt.draw.graph_draw(sub, vertex_text=label, output=f"{folder}/bubble_nt_{origin_s_node}_{e_node}.svg", ink_scale=0.7)
			gt.draw.graph_draw(sub, vertex_text=sub.vertex_index, output=f"{folder}/bubble_idx_{origin_s_node}_{e_node}.svg", ink_scale=0.7)
		else:
			logger.info("Skipping. No run code supplied")
	return sub, int(origin_source_node) #, sub_order

def nodes_between(g, s, e):
	"""
	Return all vertices of g that lie on at least one directed path
	from `source` to `sink` (source and sink themselves included).
	"""
	# vertices reachable FROM source
	out_comp = gt.label_out_component(g, s)

	# vertices that can reach sink == out-component of sink in the reversed graph
	g_rev = gt.GraphView(g, reversed=True, directed=True)
	in_comp = gt.label_out_component(g_rev, e)

	# a vertex is "between" source and sink iff it's in both sets
	on_path = g.new_vertex_property("bool")
	on_path.a = out_comp.a & in_comp.a

	return [v for v in g.vertices() if on_path[v]]

def extract_subgraph(graph, s_node, e_node, ts_len):
	vfilt = graph.new_vertex_property("bool")
	v_GFA_ID = graph.vp['v_gfaID']
	source_node = get_node_ids(s_node, graph,v_GFA_ID)
	sink_node = get_node_ids(e_node, graph,v_GFA_ID)
	# logging.info(f"Source: {s_node}-{int(source_node)}\t{e_node}{int(sink_node)}")
	nodes_subgraph = nodes_between(graph, source_node, sink_node)
	for n in list(nodes_subgraph):
		vfilt[n] = True

	sub = gt.GraphView(graph, vfilt=vfilt)
	sub = gt.Graph(sub, prune=True) 
	v_GFA_ID = sub.vp['v_gfaID']
	source_node = get_node_ids(s_node, sub,v_GFA_ID)
	sink_node = get_node_ids(e_node, sub,v_GFA_ID)
	v_label = sub.vp['v_label']
	v_label[source_node] = v_label[source_node][-(ts_len -1):]
	v_label[sink_node] = v_label[sink_node][:(ts_len -1)]
	# logging.info(f"Number of nodes in subgraph: {sub.num_vertices()}")
	return sub

def merge_bubbles_new(bbls, graph, min_length):
	g_GFA_id = graph.vp["v_gfaID"]
	g_vseq_length = graph.vp["v_seq_len"]

	nodes = set()
	for k,v in bbls.items():
		nodes.add(k)
		nodes.add(v)
	nodes = list(nodes)

	new_bubbles = {}
	new_bubbles_bw = {}
	bubbles_bw = {}
	bubbles_bw_len = {}
	b_s = 1
	temp_len = 0
	temp_b_s = 1
	short_s_nodes = set()
	while b_s:
		source_n = get_node_ids(b_s, graph, g_GFA_id)
		if not bbls.get(b_s, None):
			break
		sink_n = get_node_ids(bbls[b_s], graph, g_GFA_id)
		bubbles_bw[bbls[b_s]] = b_s
		min_path = 9999999
		p_num = 0
		for path in gt.all_paths(graph,source_n, sink_n ):
			path_len= sum([g_vseq_length[p] for p in path])
			if g_vseq_length[path[0]] < min_length:
				short_s_nodes.add(int(path[0]))
			min_path = min(min_path, path_len)
			p_num += 1 
			if p_num >= 30000:
				break
		bubbles_bw_len[bbls[b_s]] = min_path
		min_path -= g_vseq_length[source_n]
		temp_len += min_path
		if temp_len >= min_length and g_vseq_length[sink_n] >=3:
			new_bubbles[temp_b_s] = bbls[b_s]
			new_bubbles_bw[bbls[b_s]] = temp_b_s
			temp_b_s = bbls[b_s]
			temp_len = 0
	
		
		b_s = bbls.get(b_s, None)
	logger.info(f"Number of Bubbles after forward merge: {len(new_bubbles)}")
	#forwaord merge done
	start_nodes_for_sg = {} 
	chain_bubbles_start =[]
	for s_node in sorted(new_bubbles.keys()):
		if s_node == (new_bubbles[s_node] -1):
			chain_bubbles_start.append(s_node)
			logger.info(f"Chain bubbel: {s_node} - {new_bubbles[s_node]}")
		source_n = get_node_ids(s_node, graph, g_GFA_id)
		node, _ = find_startnode_forPrefix(graph, source=source_n, vertex_prop=g_vseq_length, elegible_snodes= nodes,min_length=19)

		gfa_node = g_GFA_id[node]
		start_nodes_for_sg[s_node] = gfa_node
		if gfa_node != s_node and gfa_node in new_bubbles:
			new_bubbles[gfa_node] = new_bubbles[s_node]
			new_bubbles.pop(s_node)
			start_nodes_for_sg.pop(s_node)
	logger.debug(f"startnodes per bubble start: {start_nodes_for_sg}")
	for item in chain_bubbles_start:
		del new_bubbles[item]
	return new_bubbles, start_nodes_for_sg

def find_startnode_forPrefix(g, source, vertex_prop, elegible_snodes, min_length):
	"""
	BFS from source, accumulating vertex property values along the path.
	Returns the first node where cumulative sum >= min_length.
	"""
	g_GFA_id = g.vp["v_gfaID"]
	# Store cumulative cost to reach each node
	if vertex_prop[source] >= min_length:
		return source,  vertex_prop[source]
	
	cum_cost = {source: vertex_prop[source]}

	queue = deque([source])

	best_node = None
	best_cost = float("inf")

	while queue:
		current = queue.popleft()
		current_cost = cum_cost[current]
		# Check threshold
		if current_cost >= min_length and current_cost < best_cost and g_GFA_id[current] in elegible_snodes :
			best_cost = current_cost
			best_node = current
			# Don't break — a neighbour at same BFS level might be cheaper
			continue

		# If we already exceeded best found, no need to expand further
		# if current_cost >= best_cost and vertex_prop[current] > minseq_length_e_node  and g_GFA_id[current] in elegible_snodes :
		if current_cost >= best_cost and g_GFA_id[current] in elegible_snodes :
			continue

		for neighbor in g.vertex(current).in_neighbors():
			n = int(neighbor)
			new_cost = current_cost + vertex_prop[neighbor]

			# Only visit if we found a cheaper route to this node
			if n not in cum_cost or new_cost < cum_cost[n]:
				cum_cost[n] = new_cost
				queue.append(n)

	return best_node, best_cost 

def get_allPaths_lengths(g, s_node, e_node, o_bbls, path_cap):
	v_GFA_ID = g.vp['v_gfaID']
	v_label = g.vp['v_label']

	origin_bubbles = {k:v for k,v in o_bbls.items() if s_node<= k < e_node}
	logger.info(f"Origin bubbles for {s_node} to {e_node}: {len(origin_bubbles)}")
	num_paths = 1
	b_s = min(origin_bubbles.keys())
	b_e = origin_bubbles.get(b_s,None)
	lastnodes_length = 0
	path_lengths_all = Counter({0:1})
	while True:
		g_s = get_node_ids(b_s, g, v_GFA_ID)
		g_e = get_node_ids(b_e, g, v_GFA_ID)
		logger.info(f"{g_s}, {g_e}")
		paths_lengths = []
		paths = 0
		for p in gt.all_paths( g,g_s, g_e):
			paths_lengths.append(sum([len(v_label[n]) for n in p])- lastnodes_length)
			paths += 1
			if paths % 20000 == 0:
				logger.info(f"Node: {s_node} Still collecting... {paths} paths so far")
			if paths > path_cap:
				logger.info("Path count exploded, stopping")
				break
		lastnodes_length = len(v_label[g_e])
		logger.info(f"paths: {paths}")

		temp = np.add.outer(list(path_lengths_all.keys()),paths_lengths)
		multsd = list(path_lengths_all.values())
		new_c = Counter()
		for i in range(len(temp)):
			for e in temp[i]:
				new_c[e] += multsd[i]
		path_lengths_all = new_c
		num_paths *= paths
		logger.info(f"Number of paths for bubble {b_s} to {b_e}: {num_paths} ({len(path_lengths_all)})")
		b_s = b_e
		b_e = origin_bubbles.get(b_s,None)
		if not b_e:
			break

	sum_all_paths = sum(int(k) * v for k, v in path_lengths_all.items())

	return sum_all_paths, num_paths #sum([k*v for k,v in path_lengths_all.items()])

def subgraph_stats(g, start_gfa_id: int, end_gfa_id: int, origin_bubbles, path_cap: int = 3_000_000) -> dict:
	sg = extract_subgraph(g, start_gfa_id, end_gfa_id)
	v_label  = sg.vp["v_label"]
	logger.debug(f"stats - source: {start_gfa_id}, sink: {end_gfa_id} ")
	all_path_lengths, num_paths = get_allPaths_lengths(sg, start_gfa_id,end_gfa_id, origin_bubbles, path_cap)
	
	num_nodes = sg.num_vertices()
	num_edges = sg.num_edges()
	
	logger.info(f"Number of nodes: {num_nodes}, Number of edges: {num_edges}")
	edge_to_node_ratio = num_edges / num_nodes if num_nodes else float("nan")

	# graph length = sum of label lengths over all nodes
	graph_length = sum(len(v_label[v]) for v in sg.vertices())


	return {
		"start_gfa_id":        start_gfa_id,
		"end_gfa_id":          end_gfa_id,
		"num_nodes":           num_nodes,
		"num_edges":           num_edges,
		"edge_to_node_ratio":  edge_to_node_ratio,
		"graph_length":        graph_length,
		"num_paths":           num_paths,
		"total_path_length":   all_path_lengths,
		"graph_compaction":		all_path_lengths/graph_length,
		"is_acyclic": 	gt.is_DAG(sg)
	}