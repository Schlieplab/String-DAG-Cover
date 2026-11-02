# Algorithms for Designing Therapeutic Oligonucleotides on Pangenome Graphs

This repository has the complementary code to the article "Algorithms for Designing Therapeutic Oligonucleotides on Pangenome Graphs" submitted to RECOMB 2027

## Installation

The environment currently is only configured for Linux (`linux-64`) and is managed with [pixi](https://pixi.sh).

1. Install pixi (skip if already installed):
   ```
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone this repository:
   ```
   git clone https://github.com/Schlieplab/String-DAG-Cover.git
   cd String-DAG-Cover
   ```

3. Install the environment:
   ```
   pixi install
   ```
   This also installs [QuickMers](https://github.com/Schlieplab/quickmers) from PyPI, which is used to enumerate
   edit balls.

Any script is run inside the environment with:
```
pixi run python <script>
```

## Data

`data/` contains pangenome graphs for three genes (`KRAS`, `SOD1`, `TP53`). Each folder has:

| File | Content |
|---|---|
| `graph_reindexed.gfa` | the pangenome graph |
| `directed_superbubbles.txt` | directed superbubbles found with BubbleFinder |
| `subgraphs.csv` | bubble subgraphs to evaluate, one per row, with their size (`nodes`, `edges`, `paths`, ...) |

A subgraph is selected by its `bubble_start` id from `subgraphs.csv`. `step2_extract_subgraph_ids.sh` shows how
`directed_superbubbles.txt` and `subgraphs.csv` were created.

## Quick run

The examples below use the SOD1 bubble with id `91`: 28 nodes, 640 paths and 834 target site candidates of
length 20. It is big enough to be non-trivial and small enough to run on a laptop.

To shorten the commands, set:
```
D=data/SOD1
```

All scripts write a CSV to `results/<gene>/<method>/`. The run times are rough wall-clock times on a single core.

### Product graph

Check for each target site candidate whether every path of the subgraph contains a string within `max_edit` edits
of it. The subgraph is combined with an automaton of the target site, and the target site covers the subgraph if no
path in the product graph reaches the sink. The second positional argument is `max_edit`.

```
pixi run python string_dag_cover/run_Ukk_PG.py       $D/graph_reindexed.gfa 1 $D/subgraphs.csv $D/directed_superbubbles.txt -s 91
pixi run python string_dag_cover/run_Ukk_PG_fixed.py $D/graph_reindexed.gfa 1 $D/subgraphs.csv $D/directed_superbubbles.txt -s 91
pixi run python string_dag_cover/run_AC_PG.py        $D/graph_reindexed.gfa 1 $D/subgraphs.csv $D/directed_superbubbles.txt -s 91
pixi run python string_dag_cover/run_AC_PG_fixed.py  $D/graph_reindexed.gfa 1 $D/subgraphs.csv $D/directed_superbubbles.txt -s 91
```

| Script | Automaton | Output | Time |
|---|---|---|---|
| `run_Ukk_PG.py` | Ukkonen Levenshtein automaton | `results/SOD1/Ukk_PG/result_91_1.csv` | ~20 s |
| `run_Ukk_PG_fixed.py` | fixed-length Ukkonen Levenshtein automaton | `results/SOD1/Ukk_PG_fixed/result_91_1.csv` | ~25 s |
| `run_AC_PG.py` | Aho-Corasick automaton of the edit ball | `results/SOD1/AC_PG/result_91_1.csv` | ~3.5 min |
| `run_AC_PG_fixed.py` | Aho-Corasick automaton of the fixed-length edit ball | `results/SOD1/AC_PG_fixed/result_91_1.csv` | ~1.5 min |

The column header is `max_edit`. A covering target site gets the value `max_edit`, all others stay empty:
```
string,1
AATACAGCAGGTGGGTGTTG,1.0
ACAGCAGGTGGGTGTTGTGC,1.0
AAATTTGGAGACAATACAGC,
```

Run time grows quickly with `max_edit`; with `max_edit = 2`, `run_Ukk_PG.py` takes about 1.5 min on this bubble.

### Lossless filter DP

Compute for each target site candidate its bottleneck value, the smallest number of edits with which the target
site covers every path of the subgraph.

```
pixi run python string_dag_cover/run_LLF_DP.py       $D/graph_reindexed.gfa $D/subgraphs.csv $D/directed_superbubbles.txt -s 91
pixi run python string_dag_cover/run_LLF_DP_fixed.py $D/graph_reindexed.gfa $D/subgraphs.csv $D/directed_superbubbles.txt -s 91
```

| Script | Edit model | Output | Time |
|---|---|---|---|
| `run_LLF_DP.py` | Levenshtein distance | `results/SOD1/LLF_DP/result_91.csv` | ~5 s |
| `run_LLF_DP_fixed.py` | fixed-length Levenshtein distance | `results/SOD1/LLF_DP_fixed/result_91.csv` | ~6 s |

Sample output:
```
ts_candidates,bn_value
AGACAATATAGCAGGTGGGT,3
TTGGAGATAATACAGCAGGT,2
ATTTTCATAAGTTTGGAGAC,7
```

### Consistency check

The methods answer the same question and give the same result on this example. With `max_edit = 1`, 8 of the 834
candidates cover all paths. These are the candidates with `bn_value` 1 in `LLF_DP`. In the fixed-length variant,
7 candidates cover all paths and have `bn_value` 1 in `LLF_DP_fixed`.

### Options

All run scripts accept:

| Option | Default | Meaning |
|---|---|---|
| `-s`, `--b_start` | | `bubble_start` id of the subgraph from `subgraphs.csv` |
| `-k`, `--kmer_length` | 20 | target site length |

### Authors

Nathalie Gocht, Nicola Cotumaccio, Alexandru Tomescu, Alexander Schliep and Veli Mäkinen