# The FROZEN-DRAFTER ("floor") shift b7 SGU arm: official deepseek dFlash stays untouched,
# ONLY the XPress head trains. Motivation (measured, both scales): co-training erodes the
# standalone drafter early (8B 5.43->4.60 by 10k, 14B 5.41->4.74) and NOTHING recovers it --
# not 40k extra steps (14B flat 4.74@40k..80k), not full-weight anchoring (fix A, identical
# @6 floor). This arm removes the erosion channel by construction: the package can only be
#     official drafter (guaranteed baseline)  +  whatever the head adds on top.
# 14B says the head's gross gain is ~+1.1 on an eroded drafter; even a fraction of that on
# the intact drafter puts the package ABOVE the no-refiner baseline -- which no co-train arm
# has achieved yet.
#
# vs hybrid-s1-b7shift-sgu-consis-k3.sh (the co-train sibling), the deltas:
#   + --no-cotrain-drafter          (drafter requires_grad=False; trainer verifies 0 trainable)
#   ~ --lambda-base-start/floor 0   (anchor loss on a frozen drafter has no gradient; the
#                                    trainer REFUSES lambda>0 with --no-cotrain-drafter)
#   ~ --drafter-lr-scale 1.0        (no drafter params in the optimizer; keep single LR group)
#   ~ output-dir / wandb-name
# Everything else (head flags, K=3 consistency, loss alphas, gamma, rank 256, lr, gb32,
# data, shift convention) is IDENTICAL to the co-train arm -> the pair is a clean A/B on
# exactly one question: does freezing the drafter beat co-training once erosion is priced in?
#
# Expected wandb signature: eval/drafter_accept_len pinned at its step-0 value forever (it
# IS the guardrail -- any drift means the freeze failed); eval/refiner_accept_len is the
# only moving curve.
#
#   bash submit.sh 2 xpress hybrid-s1-b7shift-sgu-consis-k3-frozen

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
    --no-cotrain-drafter \
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
    --num-anchors 400 \
    --max-length 4096 \
    --batch-size 2 \
    --accumulation-steps 1 \
    --num-epochs 10 \
    --eval-interval 1000 \
    --target-model-path /proj/checkpoints/daviswer/results/specu/checkpoints/Qwen3-8B \
    --train-data-path /proj/checkpoints/daviswer/results/specu/refiner_train_nothink.jsonl \
    --eval-data-path /proj/checkpoints/daviswer/results/specu/refiner_eval_nothink.jsonl \
    --cache-dir /proj/checkpoints/zwang619/ \
    --chat-template qwen \
    --attention-backend sdpa \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --drafter-lr-scale 1.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --dflash-model-path deepseek-ai/dflash_qwen3_8b_block7 \
    --lambda-base-start 0 \
    --lambda-base-floor 0 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-sgu-b7shift-consis-k3-frozen \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-sgu-b7shift-consis-k3-frozen \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
