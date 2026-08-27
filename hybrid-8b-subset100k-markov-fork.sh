# OFFICIAL-RECIPE REPRODUCTION ARM: the released dspark_qwen3_8b_block7 training config
# (DeepSpec config/dspark/dspark_qwen3_8b.py), run through our fork's online pipeline.
# Goal: reproduce the official checkpoint's numbers on the SAME data (user-verified: the
# released dflash/dspark were trained on this very dataset) -- package ~= official dspark,
# bare drafter ~= 2.75. Success validates our data+code+recipe end to end; every other
# arm's differences then attribute to our DELIBERATE changes, not hidden bugs.
#
# Official config, mapped 1:1 (config/dspark/dspark_qwen3_8b.py):
#   scratch 5-layer drafter, target_layers [1,9,17,25,33], mask 151669  -> --random-init-drafter
#   vanilla markov r256, NO consistency, NO drafter anchor              -> --markov-only, lambda 0
#   loss on refined only: gamma 4.0, ce 0.1, l1 0.9                     -> same flags
#   num_anchors 512, seq 4096, chat qwen                                -> same
#   lr 6e-4, warmup 0.04, wd 0, clip 1.0, bf16, 10 epochs               -> same
#   GLOBAL BATCH 512 (local 1 x 8 GPUs x accum 64)                      -> bs2 x 32 GPUs x accum 8
#     (=> ~2,636 optim steps/epoch, ~26k total -- the official schedule is SHORT)
#
# KNOWN FIDELITY GAPS (accepted, in writing):
#   - confidence head: official trains one at alpha=1.0 and its gradients DO enter the
#     drafter backbone. The fork has no confidence head. If this run misses the official
#     numbers, the confidence head is the prime suspect and gets implemented next.
#   - no torch.compile; FSDP instead of no_shard DDP; dataloader order/seed differ.
#
# Eval protocol for the verdict (temp0+temp1, gsm8k x128):
#   package (markov graft, native decode) vs deepseek-ai/dspark_qwen3_8b_block7 direct;
#   drafter-only export vs 2.75 (official dspark bare) -- NOT vs 5.43 (that's dflash).
#
#   bash submit.sh 4 xpress hybrid-s1-b7shift-markov-official-repro

# pin the training env: bare `torchrun` resolves via the SUBMISSION shell PATH (blaunch
# inherits it), so submitting from e.g. deepspec_eval crashes with missing `accelerate`.
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
    --markov-only \
    --l1-alpha 0.9 \
    --ce-alpha 0.1 \
    --loss-decay-gamma 4.0 \
    --markov-rank 256 \
    --learning-rate 0.0006 \
    --num-anchors 512 \
    --max-length 4096 \
    --batch-size 1 \
    --accumulation-steps 64 \
    --num-epochs 10 \
    --eval-interval 500 \
    --target-model-path /proj/checkpoints/daviswer/results/specu/checkpoints/Qwen3-8B \
    --train-data-path /proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_8b_subset100k/refiner_train_100k.jsonl \
    --eval-data-path /proj/checkpoints/daviswer/results/specu/refiner_eval_nothink.jsonl \
    --cache-dir /proj/checkpoints/zwang619/ \
    --chat-template qwen \
    --attention-backend sdpa \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --tp-size 1 \
    --drafter-lr-scale 1.0 \
    --log-interval 10 \
    --save-interval 1600 \
    --dflash-model-path deepseek-ai/dflash_qwen3_8b_block7 \
    --lambda-base-start 0 \
    --lambda-base-floor 0 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-markov-8b-subset100k-fork \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-markov-8b-subset100k-fork \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
