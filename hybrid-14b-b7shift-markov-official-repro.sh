# OFFICIAL-RECIPE REPRODUCTION ARM, 14B: the released dspark_qwen3_14b_block7 training
# config (DeepSpec config/dspark/dspark_qwen3_14b.py) through our fork's online pipeline.
# Sibling of hybrid-s1-b7shift-markov-official-repro.sh (8B) -- same rationale, same
# fidelity gaps (no confidence head [official alpha=1.0, gradients reach the drafter];
# no torch.compile; FSDP vs no_shard; dataloader order).
#
# Official 14B config == official 8B config except target_layer_ids [1,10,19,28,37]
# (40-layer target). Our --random-init-drafter takes the arch from the released
# dflash_qwen3_14b_block7 config, which carries exactly those layer ids (verified).
#
# Official mapped 1:1: scratch 5-layer drafter, vanilla markov r256 (--markov-only),
# NO drafter anchor (lambda 0), loss on refined only (gamma 4, ce 0.1, l1 0.9),
# num_anchors 512, seq 4096, lr 6e-4, warmup 0.04, wd 0, clip 1.0, bf16, 10 epochs,
# GLOBAL BATCH 512 = local 1 x 32 GPUs x accum 16 (14B target forward needs bs1).
# ~2,636 optim steps/epoch, ~26k total. save-interval 3000 = official checkpointing.
#
# Verdict protocol (gsm8k x128): markov-graft package vs deepseek-ai/dspark_qwen3_14b_block7
# direct; drafter-only export vs the official 14B dspark BARE drafter (strip heads first --
# reproduce the collapse, not dflash's 5.41/5.96).
#
#   bash submit.sh 4 xpress hybrid-14b-b7shift-markov-official-repro

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
    --lambda-base-start 0 \
    --lambda-base-floor 0 \
    --lambda-base-decay-ratio 1.0 \
    --output-dir /proj/checkpoints/zwang619/hybrid_out/xpress-markov-14b-b7shift-official-repro \
    --report-to wandb \
    --wandb-project ripple-dspark \
    --wandb-name xpress-markov-14b-b7shift-official-repro \
    --resume \
    > >(stdbuf -oL sed "s/^/[$(hostname)] /") \
    2> >(stdbuf -oL sed "s/^/[$(hostname)] /" >&2)
