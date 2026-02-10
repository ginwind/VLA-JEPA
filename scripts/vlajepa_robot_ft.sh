export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=disabled

accelerate launch \
  --config_file /home/dataset-local/ginwind/VLA-JEPA/starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  /home/dataset-local/ginwind/VLA-JEPA/starVLA/training/train_starvla.py \
  --config_yaml /home/dataset-local/ginwind/VLA-JEPA/scripts/config/vlajepa_robot_ft.yaml
