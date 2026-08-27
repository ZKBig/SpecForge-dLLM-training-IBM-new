# The FROM-SCRATCH b7 shift SGU arm: drafter RANDOM-INIT (arch-only from the official b7
# config), co-trained jointly with the XPress head from step 0.
#
# Why (the unifying law, settled by the finished b16 arm): co-train's effect on the drafter
# depends on how far it sits below its teacher-forced optimum. Saturated warm start
# (deepseek b7 5.43) -> gradients can only re-divide labor -> erosion (all our b7 arms, and
# official dspark's own 2.75 bare drafter). Under-saturated start (z-lab b16: bare 6.50) ->
# co-train IMPROVED the drafter 6.50->8.83 while the head added +1.13 (package 9.96, temp0).
# Scratch is the extreme under-saturated case: every gradient is genuine improvement, the
# drafter can only go UP, and the head co-adapts from day one -- the co-adaptation ceiling
# the frozen arm gives up, without the erosion the warm-started co-train arms suffer.
# Official dspark itself is a joint run (its drafter is ~uncorrelated with released dflash);
# they just trained it UNPROTECTED. We keep lambda_base 0.6, which the b16 arm proved
# compatible with a rising drafter.
#
# vs hybrid-s1-b7shift-sgu-consis-k3.sh (warm-started co-train sibling), the deltas:
#   + --random-init-drafter        (weights fresh; arch/target_layers/mask from the b7 config)
#   ~ --drafter-lr-scale 1.0       (a scratch drafter needs the full LR; 0.1 was erosion
#                                   protection for the saturated warm start -- pointless here.
#                                   The successful b16 arm also ran scale 1.0.)
#   ~ output-dir / wandb-name
# Everything else (head flags, K=3 consistency, alphas, gamma, rank 256, lr, gb32, data,
# shift convention, lambda_base 0.6) is IDENTICAL.
#
# Success criteria (external, DeepSpec temp0/temp1 vs deepseek b7 baselines 5.98/5.43):
#   - drafter-only must RISE monotonically across checkpoints (the b16 signature);
#   - package must eventually beat the frozen arm's curve to justify co-adaptation.
# Judge at 50k/100k; a scratch drafter needs time before comparisons are meaningful.
#
#   bash submit.sh 2 xpress hybrid-s1-b7shift-sgu-consis-k3-scratch

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
    --lambda-base-start 0.6 \
    --lambda-base-floor 0.6 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-sgu-b7shift-consis-k3-scratch \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-sgu-b7shift-consis-k3-scratch \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
