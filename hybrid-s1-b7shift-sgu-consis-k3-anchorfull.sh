# The SHIFT-CONVENTION b7 SGU (XPress) co-train arm: sgu_linear_mixer head + K=3 Jacobi
# consistency, on the natively warm-started shift drafter. Gate A already passed on the
# sibling markov arm (drafter_accept 4.04 at step 1000 vs the fillin ceiling 3.84).
#
# Design: warm-start from deepseek-ai/dflash_qwen3_8b_block7 in its NATIVE label alignment
# (--block-convention shift), so the drafter starts AT official quality instead of in the
# fillin-retrofit hole (which permanently capped the standalone drafter at ~3.9 vs native
# 5.43). All 7 slots draft (no 6-slot tax). Protections: lambda_base 0.6 drafter anchor +
# drafter-lr-scale 0.1; guardrail = eval/drafter_accept_len vs its OWN FIRST value.
#
# GATE A (index-mapping correctness, read it off the FIRST eval at step 1000):
#     eval/drafter_accept_len ~ 5    -> every index in the shift port is right; let it run
#     eval/drafter_accept_len ~ 2    -> convention misalignment somewhere; kill and debug
# The signal is unmistakable because step-1000 weights are still ~= the official checkpoint
# (warmup + lr-scale 0.1), and a fillin/shift mix-up collapses deep positions to noise.
#
# vs hybrid-s1-b7shift-markov-cotrain.sh (the shift markov arm), the deltas:
#   ~ head flags: --no-sublayer-norm --no-mix-out --no-residual --mixer-init zeros
#     (sgu_linear_mixer, no --markov-only) + --consistency-weight 0.3 --consistency-passes 3
#     == exactly how the fillin SGU arm (-test1) differs from the fillin markov arm
#   ~ output-dir / wandb-name
# Everything else (loss alphas, gamma, rank 256, lambda_base, lr, gb32, data) is kept.
#
#   bash submit.sh 2 xpress hybrid-s1-b7shift-sgu-consis-k3

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
    --anchor-full-weight \
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
    --drafter-lr-scale 0.1 \
    --log-interval 50 \
    --save-interval 10000 \
    --dflash-model-path deepseek-ai/dflash_qwen3_8b_block7 \
    --lambda-base-start 0.6 \
    --lambda-base-floor 0.6 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-sgu-b7shift-consis-k3-anchorfull \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-sgu-b7shift-consis-k3-anchorfull \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
