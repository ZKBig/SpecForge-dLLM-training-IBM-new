# SHIFT-CONVENTION variant: identical to hybrid-14b-b7warm-sgu-consis-k3.sh except --block-convention shift
# (+ output/wandb names). deepseek-ai/dflash_qwen3_14b_block7 is shift-native, so this arm
# warm-starts WITHOUT the fillin-retrofit hole (8B gate A: shift drafter_accept 4.04 at step
# 1000 vs the fillin arm's 210k-step ceiling 3.84). All 7 slots draft.
#
# Qwen3-14B + b7 DFlash drafter, SGU head, K=3 Jacobi consistency, drafter CO-TRAINED.
# The 14B counterpart of hybrid-s1-b7warm-sgu-linear-mixer-consis-k3-test1.sh: same head
# architecture and objective, retargeted at 14B with our freshly generated 14B data.
#
# 14B-specific values (all verified, not guessed):
#   target      Qwen3-14B: 40 layers, hidden 5120, vocab 151936  (HF cache, pulled during datagen)
#   drafter     deepseek-ai/dflash_qwen3_14b_block7 -> block_size 7, 5 draft layers,
#               hidden 5120, target_layer_ids [1,10,19,28,37], num_target_layers 40,
#               markov_rank 0 (no head in the checkpoint -- we train ours from scratch)
#   data        our 14B regeneration, 1,348,810 train / 1,000 eval holdout (disjoint)
#
# num-anchors 400 = same as the 8B runs, so 14B-vs-8B curves stay comparable (DeepSpec uses
# 512, but with local_batch_size 1 + grad accumulation to 512 sequences -- a different recipe).
#
#   bash submit.sh 2 xpress hybrid-14b-b7warm-sgu-consis-k3

export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1

export QUACK_COMPILE_WORKERS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${2:-1800}
export TORCH_FR_BUFFER_SIZE=2097152

export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
: "${WANDB_API_KEY:?export WANDB_API_KEY in your environment before running}"

MASTER_ADDRESS=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=$(( 20000 + (${LSB_JOBID:-0} % 40000) ))
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $(echo $HOSTNAME | cut -d'.' -f1) | cut -d':' -f1)-1))

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m scripts.train_hybrid_refiner_IBM \
    --block-convention shift \
    --l1-alpha 0.9 \
    --ce-alpha 0.1 \
    --loss-decay-gamma 4.0 \
    --markov-rank 256 \
    --learning-rate 0.0006 \
    --num-anchors 400 \
    --max-length 4096 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --num-epochs 10 \
    --eval-interval 1000 \
    --target-model-path Qwen/Qwen3-14B \
    --train-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/refiner_train_nothink.jsonl \
    --eval-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/refiner_eval_nothink.jsonl \
    --cache-dir /proj/checkpoints/zwang619/ \
    --chat-template qwen \
    --attention-backend sdpa \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --drafter-lr-scale 0.1 \
    --log-interval 50 \
    --save-interval 10000 \
    --dflash-model-path deepseek-ai/dflash_qwen3_14b_block7 \
    --lambda-base-start 0.6 \
    --lambda-base-floor 0.6 \
    --lambda-base-decay-ratio 1.0 \
    --no-sublayer-norm \
    --no-mix-out \
    --no-residual \
    --consistency-weight 0.3 \
    --consistency-passes 3 \
    --mixer-init zeros \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-14b-sgu-b7shift-k3 \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-14b-sgu-b7shift-k3 \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
