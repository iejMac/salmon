#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=compute,h100bldg40
#SBATCH --exclusive
#SBATCH --output=ddp_training_%j.out

# Set environment variables for NCCL
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=2
# export LOGLEVEL=INFO

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

# Run the job using the container environment
srun --container-image=$CONTAINER_IMAGE \
     --container-mounts=$CONTAINER_MOUNTS \
     bash -c "
     cd /home/maciej/code/junk/salmon && pip install -e . && \
     torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 $REPO_DIR/workloads/dist_mlp/dist_mlp.py
     "