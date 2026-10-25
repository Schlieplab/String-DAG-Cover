import os, sys, argparse, logging
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prototypes_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {project_root}")
print(f"Prototypes dir: {prototypes_dir}")
sys.path.insert(0, project_root)   # for src.*
sys.path.insert(0, prototypes_dir) # for load_GFA
from utils.logging import LevelFormatter
from utils.utils import (
    get_bubbles,
    load_gfa,
    merge_bubbles_new,
    subgraph_stats,
)

os.environ.setdefault('DISPLAY', '')
font = "sans"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Merge bubbles based on sequence length"
    )

    parser.add_argument('--graph', "-g",
                        type=str,
                        help="Graph file in GFA format")
    parser.add_argument('--bubbles', "-b",
                        type=str,
                        help="Bubble from bubble finder")
    parser.add_argument('--seq_len', '-k',
                        type=int,
                        help="Sequence length threshold for merging bubbles")
    parser.add_argument('--output', "-o",
                        type=str,
                        help="Output file with new bubbles and start node ids for DP")
    args = parser.parse_args()

    graph_file = os.path.expanduser(args.graph)
    bubble_file = os.path.expanduser(args.bubbles)
    ts_len = args.seq_len
    output_file = args.output
    handler = logging.StreamHandler()
    handler.setFormatter(LevelFormatter())
    debug = False


    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(f"{sys.argv[0]} starting up")

    logger.info("Load graph and bubbles") 
    graph = load_gfa(graph_file, sort=False)
    bubbles = get_bubbles(bubble_file)
    g_GFA_id = graph.vp["v_gfaID"]

    logger.info("Merge bubbles based on sequence length")  
    new_bubbl, start_ids_bubb = merge_bubbles_new(bubbles, graph, ts_len-1)

    logger.info(f"Number of bubbles after merging: {len(new_bubbl)}")
    with open(output_file, "w") as out:
        out.write("bubble_start,bubble_end,prefix_node,nodes,edges,edges_node_ratio,paths,all_paths_length,graph_length,graph_compaction,is_acyclic\n")
        for b_s, b_e in new_bubbl.items():
            dic = subgraph_stats(graph, start_ids_bubb.get(b_s, b_s), b_e, bubbles)
            out.write(f"{b_s},{b_e},{start_ids_bubb.get(b_s, b_s)},{dic['num_nodes']},{dic['num_edges']},{dic['edge_to_node_ratio']},{dic['num_paths']},{dic['total_path_length']},{dic['graph_length']},{dic['graph_compaction']},{dic['is_acyclic']}\n")
            logger.info(f"{b_s},{b_e},{start_ids_bubb.get(b_s, b_s)},{dic['num_nodes']},{dic['num_edges']},{dic['edge_to_node_ratio']},{dic['num_paths']},{dic['total_path_length']},{dic['graph_length']},{dic['graph_compaction']},{dic['is_acyclic']}\n")

    logger.info("Script Done.")