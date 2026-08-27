# The 14B FROM-SCRATCH XPRESS (SGU K3) arm, official-frame batch: the head-to-head
# comparison partner of hybrid-14b-b7shift-markov-official-repro.sh (job 812474).
#
# Both arms: random-init 5-layer drafter (arch from dflash_qwen3_14b_block7), shift b7,
# anchors 512, seq 4096, lr 6e-4, warmup 0.04, wd 0, clip 1.0, bf16, GLOBAL BATCH 512
# (bs1 x 32 GPUs x accum 16), 10 epochs ~= 26k optim steps, same 14B dataset.
# The deltas -- OUR method vs the official recipe:
#   ~ head: XPress SGU rank 256 + K=3 Jacobi consistency 0.3, zeros mixer init
#     (vs vanilla markov r256)
#   ~ lambda_base 0.6 constant drafter anchor (vs none) -- the b16 arm proved this
#     compatible with a RISING scratch drafter (6.50->8.83); the markov repro arm is
#     live-demonstrating what lambda=0 does (drafter stalls at ~2.05 internal while the
#     head climbs). Bare-drafter usability is part of our method's claim, so it stays.
#
# Verdict protocol at 26k (gsm8k x128, temp0): package (xpress path K=6) vs the markov
# repro package vs official dspark_14b direct; drafter-only export -- ours should be
# USABLE (lambda-protected), the markov repro's collapsed. Compare drafter curves in
# wandb too: this arm's drafter_accept should keep rising where the repro's flatlined.
#
#   bash submit.sh 4 xpress hybrid-14b-b7shift-sgu-scratch-gb512

# pin the training env: bare `torchrun` resolves via the SUBMISSION shell PATH.
export PATH=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_train/bin:$PATH

export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1

export QUACK_COMPILE_WORKERS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${2:-1800}
export TORCH_FR_BUFFER_SIZE=2097152

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
    --random-init-drafter \
    --l1-alpha 0.9 \
    --ce-alpha 0.1 \
    --loss-decay-gamma 4.0 \
    --markov-rank 256 \
    --no-sublayer-norm \
    --no-mix-out \
    --no-residual \
    --consistency-weight 0.3 \
    --consistency-passes 3 \
    --mixer-init zeros \
    --learning-rate 0.0006 \
    --num-anchors 512 \
    --max-length 4096 \
    --batch-size 1 \
    --accumulation-steps 16 \
    --num-epochs 10 \
    --eval-interval 500 \
    --target-model-path Qwen/Qwen3-14B \
    --train-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/refiner_train_nothink.jsonl \
    --eval-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/refiner_eval_nothink.jsonl \
    --cache-dir /proj/checkpoints/zwang619/ \
    --chat-template qwen \
    --attention-backend sdpa \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --drafter-lr-scale 1.0 \
    --log-interval 10 \
    --save-interval 3000 \
    --dflash-model-path deepseek-ai/dflash_qwen3_14b_block7 \
    --lambda-base-start 0.6 \
    --lambda-base-floor 0.6 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-sgu-14b-b7shift-scratch-gb512 \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-sgu-14b-b7shift-scratch-gb512 \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
