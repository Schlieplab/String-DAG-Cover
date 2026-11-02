#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G          
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --partition=smp 
#SBATCH --account=root
#SBATCH --propagate=NONE
#SBATCH --job-name=Step2
#SBATCH --output='slurm_out/%x/slurm-%j.out'



gene_tag=$1

output_dir=data/${gene_tag}
bubbles_out=${output_dir}/directed_superbubbles.txt
k=20
subgraphs_out=${output_dir}/subgraphs.csv

/home/ngocht/software/BubbleFinder/BubbleFinder directed-superbubbles -g "${output_dir}/graph_reindexed.gfa" -o "${bubbles_out}" -j "${SLURM_CPUS_PER_TASK}" -T

pixi run python string_dag_cover/merge_bubbles.py -g "${output_dir}/graph_reindexed.gfa" -b ${bubbles_out} -k ${k} -o ${subgraphs_out}