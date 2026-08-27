# SpecForge-dLLM-training-IBM(我们的主力 fork)使用手册 — IBM LSF 集群

> 这是**我们做研究的主力框架**:DFlash 草稿模型 + XPress/Markov refiner 头的联合训练。
> 另外两个仓库的分工:`DeepSpec` = 官方 DSpark 复现与对照;`speculators` = 面向 vLLM 上游生态的实现;`Domino` = 墙钟性能基准。
> 本仓库同时也是**所有集群任务的提交入口**(`submit.sh` 只认自己目录下的脚本,DeepSpec 的运行脚本以软链挂在这里)。

---

## 0. 一分钟上手

```bash
cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM

# 提交一个训练臂:submit.sh <节点数> <结果目录组名> <脚本名不含.sh>
bash submit.sh 1 xpress hybrid-s1-b16warm-sgu-linear-mixer-consis-k3   # 8B b16 黄金臂
bash submit.sh 2 xpress hybrid-14b-b7shift-sgu-consis-k3               # 14B XPress 臂

# 看状态 / 日志 / 杀任务
bjobs -w
tail -f /proj/checkpoints/zwang619/results/xpress/<脚本名>/output.log
bkill <jobid>
```

---

## 1. 环境

不需要 `conda activate`,脚本里统一用 PATH 前置(在 LSF job 里最可靠):

```bash
export PATH=/proj/checkpoints/zwang619/miniconda3/envs/dLLM_train/bin:$PATH
export HF_HOME=/proj/checkpoints/zwang619/.cache/huggingface
```

| env | 用途 |
|---|---|
| `dLLM_train` | **本仓库训练**(torch 2.9.1+cu128) |
| `dLLM_35b` | 35B MoE 臂(torch 2.11) |
| `speculators_train` | speculators 训练 / Domino 基准 |
| `deepspec_eval` | DeepSpec 训练与评测 |

重建:`conda create -n dLLM_train python=3.12` → 装 torch 2.9.1+cu128 → `pip install -e .`(本仓库)。

---

## 2. 集群约定(踩过的坑,务必读)

```bash
bsub -q normal -G grp_ai_compiler_design -M 2000G -hl -n 2 -J xpress/my-run \
  -gpu "num=8/task:mode=exclusive_process" \
  -R "select[hname != 'p1-r04-n2' && hname != 'p1-r10-n4']" \
  -oo /proj/checkpoints/zwang619/results/xpress/my-run/output.log \
  -eo /proj/checkpoints/zwang619/results/xpress/my-run/err.log \
  blaunch bash my-run.sh
```

- **`-n N` 是 task ≈ 节点数**(配 `num=8/task`),不是 CPU 核数。纯 CPU 活(建索引、转换)一律 `-n 1`。
- **`blaunch`** 让脚本在每个节点各跑一遍(多节点训练必需)。
- **`-oo/-eo` 必须绝对路径**(相对路径 job 会秒死)。
- **坏节点排除**:`p1-r04-n2`、`p1-r10-n4`(故障 GPU,会 CUDA illegal access)。`submit.sh` 里没带,重要任务建议手写 bsub 带上。
- **登录节点的后台进程会被清理**,任何长任务都要走 LSF。
- job 无 traceback 就 `exit 255` / `lsb_launch(): Failed while waiting for tasks` → 节点级故障,原样重投。

日志目录统一:`/proj/checkpoints/zwang619/results/<组名>/<脚本名>/{output,err}.log`。

---

## 3. 训练入口与臂的组织方式

所有训练臂都是 `scripts/train_hybrid_refiner_IBM.py` 的不同参数组合(22 个脚本共用这一个入口)。**每个 `.sh` 的文件头都有详细的设计说明**(为什么这么设、判据是什么、与兄弟臂的差异),改臂之前先读那段注释。

### 命名规则

```
hybrid - <目标模型> - <暖启/约定> - <头类型> - <变体>
         s1 = 8B      b7shift      markov     anchorfull / frozen / scratch
         14b          b7warm       sgu(=XPress)
                      b16warm
```

### 主要臂速查

| 脚本 | 说明 |
|---|---|
| `hybrid-s1-b16warm-sgu-linear-mixer-consis-k3.sh` | **8B 黄金臂**:z-lab b16 暖启 + XPress + K3。404k 步收官:裸 drafter 8.83 / 整包 9.96(T0) |
| `hybrid-s1-b7shift-sgu-consis-k3.sh` | 8B deepseek b7 暖启(shift 约定)+ XPress + K3 |
| `hybrid-s1-b7shift-sgu-consis-k3-frozen.sh` | **饱和暖启的主推配方**:冻结 drafter(`--no-cotrain-drafter`,λ=0),保证整包 ≥ 官方基线 |
| `hybrid-s1-b7shift-sgu-consis-k3-scratch.sh` | 从零训 drafter(`--random-init-drafter`) |
| `hybrid-s1-b7shift-sgu-consis-k3-anchorfull.sh` | anchor 项不做位置衰减的变体 |
| `hybrid-s1-b7shift-markov-cotrain.sh` | 同上但用 Markov 头(XPress 的对照组) |
| `hybrid-14b-b7shift-sgu-consis-k3.sh` / `-markov-official-repro.sh` | 14B 的 XPress / Markov 官方复现臂 |
| `hybrid-8b-subset100k-markov-fork-4node.sh` | 10 万行子集(与 DeepSpec 官方代码做同数据 A/B 用) |
| `speculators-b16-2node.sh` | **speculators 框架**的训练(见 speculators/README_CLUSTER.md) |
| `train-35b-sgu-dspark-*.sh` | Qwen3.5-35B MoE 臂 |
| `deepspec-online-*.sh`、`xpress-online-*.sh` | **软链**到 DeepSpec 仓库的脚本(见 DeepSpec/README_CLUSTER.md) |

### 关键参数(以黄金臂为例)

```bash
--target-model-path .../Qwen3-8B          # 教师
--dflash-model-path z-lab/Qwen3-8B-DFlash-b16   # drafter 暖启来源
--train-data-path .../refiner_train_nothink.jsonl    # 1.35M 行,三个框架共用同一份
--eval-data-path  .../refiner_eval_nothink.jsonl
--num-anchors 400 --max-length 4096 --batch-size 1 --accumulation-steps 1
--learning-rate 6e-4 --warmup-ratio 0.04 --num-epochs 10 --max-grad-norm 1.0
--l1-alpha 0.9 --ce-alpha 0.1 --loss-decay-gamma 4.0     # DSpark 目标函数
--markov-rank 256                                        # 头的低秩瓶颈
--lambda-base-start 0.6 --lambda-base-floor 0.2 --lambda-base-decay-ratio 1.0   # 裸 drafter anchor 调度
--consistency-weight 0.3 --consistency-passes 3          # K=3 Jacobi 一致性(XPress)
--no-sublayer-norm --no-mix-out --no-residual --mixer-init zeros   # sgu_linear_mixer 头结构
--report-to wandb --wandb-project ripple-dspark --wandb-name <臂名>
--resume                                                 # 自动续训
```

**方法学要点(不要随便改)**:

| 参数 | 为什么 |
|---|---|
| `--block-convention shift` vs 默认 fillin | **必须匹配 drafter 的原生约定**:deepseek b7 是 shift(7 个 slot 全预测),z-lab b16 是 fillin(slot 0 放锚点)。用错会让每个位置错位一格,接受长度塌到 ~2 |
| `--lambda-base-*` | 裸 drafter 的 anchor 保护。**饱和暖启(deepseek 5.43)必须保护或冻结**,否则联合训练会侵蚀它(已在官方代码里复现,是目标函数的内在性质);欠饱和暖启(z-lab b16)反而越训越好 |
| `--drafter-lr-scale` | drafter 相对 lr。黄金臂用 1.0(欠饱和);饱和暖启臂用 0.1 |
| `--consistency-passes` | K 越大质量越好但每步越贵 |

### 临时改参数

直接复制一个最接近的 `.sh` 改几行,**记得同时改 `--output-dir` 和 `--wandb-name`**,否则会写进别的臂的目录、还可能被 `--resume` 误接。

---

## 4. checkpoint 与续训

- 位置:`/proj/checkpoints/zwang619/hybrid_out/<臂名>/epoch_E_step_N/`,内含 `refiner_cotrain.pt`(联训权重:`draft_state_dict` + 头 + args)和 32 个 `optim_rank*.pt`。
- `--resume` 会自动找最新 checkpoint 续训 → **被杀的 job 原样重投即可继续**。
- `--save-interval` 控制存档频率;有 `auto_archiver.sh`(作为 LSF job 跑)做旧 checkpoint 轮转,避免撑爆配额。

---

## 5. 观测

wandb 项目 `ripple-dspark`。关键曲线:

| 指标 | 含义 |
|---|---|
| `eval/refiner_accept_len` | 整包接受长度 |
| `eval/drafter_accept_len` | **裸 drafter**——侵蚀/冻结臂的护栏,必须盯 |
| `train/{loss,accuracy,ce,l1,tf,cons}` | 目标函数分项(`l1` = `sum|p−q|` = 2×TV) |
| `train/cons_pass{j}`、`train/cons_spread` | 各 Jacobi 轮的一致性损失与跨轮差。**spread ≈ 0 说明 rollout 已到不动点,K>1 白花算力** |
| `train/lambda_base` | 当前 λ |
| `progress/optim_step`、`progress/samples` | **跨三个框架通用横坐标**(DeepSpec / speculators 发同名指标,可直接叠图) |

命令行:

```bash
grep "Train - Step" /proj/checkpoints/zwang619/results/xpress/<臂名>/output.log | tail -3
grep "Eval - Step"  /proj/checkpoints/zwang619/results/xpress/<臂名>/output.log | tail -3
# ===== Eval - Step 211000: refiner_accept=7.530 drafter_accept=6.842 =====
```

---

## 6. 评测本仓库训出来的 checkpoint

`refiner_cotrain.pt` 是 fork 格式,两条现成链路:

```bash
# A. 质量(接受长度):DeepSpec 的 eval.py 外挂头
export PATH=/proj/checkpoints/zwang619/miniconda3/envs/deepspec_eval/bin:$PATH
cd /proj/checkpoints/zwang619/DeepSpec
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
    --target_name_or_path Qwen/Qwen3-8B \
    --draft_name_or_path /proj/checkpoints/zwang619/eval_ckpts/zlab_b16_dspark_shell \
    --xpress-refiner-path /proj/checkpoints/zwang619/hybrid_out/xpress-consis-k3/epoch_10_step_404010/refiner_cotrain.pt \
    --xpress-refine-passes 6 --temperature 0.0

# B. 速度(tok/s):Domino 基准
bash /proj/checkpoints/zwang619/Domino/code/bench_404k_local.sh gsm8k
```

---

## 7. 数据

三个框架共用同一份数据(这点很重要,保证跨框架比较干净):

| 用途 | 路径 |
|---|---|
| 8B 训练 | `/proj/checkpoints/daviswer/results/specu/refiner_train_nothink.jsonl`(1.35M 行,5.4GB) |
| 8B 评测 | `/proj/checkpoints/daviswer/results/specu/refiner_eval_nothink.jsonl` |
| 14B | `/proj/checkpoints/zwang619/DeepSpec/train_datasets/qwen3_14b/`(也已上传 HF 私有 repo `VictorZheng/qwen3-14b-refiner-nothink`) |

数据生成脚本:`datagen-qwen3-14b.sh`、`datagen-qwen35-35b.sh`、`ibm_dflash_datagen_sharded.sh`。

---

## 8. 故障排查

| 症状 | 处理 |
|---|---|
| 接受长度只有 ~2 | `--block-convention` 与 drafter 原生约定不符(shift/fillin 搞反) |
| 裸 drafter 越训越差 | 侵蚀。饱和暖启就该用 frozen 臂或调高 `--lambda-base-*` |
| `cons_spread` ≈ 0 | Jacobi 已到不动点,K>1 无增益,可降 K 省算力 |
| CUDA OOM | 降 `--num-anchors`,或确认 `--grad-checkpoint-loss` 类开关已开 |
| job 秒死无日志 | `-oo/-eo` 写了相对路径 |
| 反复在同一台节点崩(illegal memory access) | 坏 GPU → 把 hostname 加进 `-R select[hname != ...]` |
| 训练"从很大的 step 开始" | `--resume` 接上了旧 checkpoint → 换 `--output-dir` 或挪走旧目录 |

---

## 9. 仓库里的临时垃圾

跑崩的任务会在仓库根目录留下 `pymp-*`(128 个)、`tmp*wandb-*`、`torchelastic_*`、`__pycache__` 等空壳目录,以及 1.9G 的 `wandb/` 本地缓存。它们**不影响运行**,清理:

```bash
cd /proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM
rm -rf pymp-* tmp*wandb-* torchelastic_* __pycache__ torchinductor_zhengw619
# wandb/ 是本地运行缓存(1.9G),云端已有备份,确认后可删:
# rm -rf wandb
```

打包分发时这些已被排除(见 `/proj/checkpoints/zwang619/SpecForge-dLLM-training-IBM.tar.gz`)。
