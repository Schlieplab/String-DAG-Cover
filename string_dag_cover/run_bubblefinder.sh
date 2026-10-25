#! /bin/bash
#SBATCH --cpus-per-task=104
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --partition=smp
#SBATCH --account=root
#SBATCH --job-name="bubblefinder"
#SBATCH --output="slurm_out/%x/slurm-%j.out"

conda activate py312_pangenomic
MAX_NT=32
# GRAPHS_FOLDER="../graphs/"
GRAPHS_FOLDER="../src/pggb_sod1_95_1000/"
BBLS_FOLDER="../bubbles/"

echo $MAX_NT

/home/ngocht/software/BubbleFinder/BubbleFinder  directed-superbubbles -g "${GRAPHS_FOLDER}"SOD1_haplos.fasta.eb16b4e.7bdde5a.0fd8284.smooth.fix.gfa -o "${BBLS_FOLDER}"directed_superbubbles_pggb.txt -j 104 -T
 