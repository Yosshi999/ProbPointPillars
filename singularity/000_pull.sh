#!/bin/bash
#SBATCH -p v
#SBATCH -c 4

export PATH=/home/app/singularity/bin:$PATH
singularity pull -n second-pytorch.simg docker://yosshi999/second.pytorch:cudaarch7.0
