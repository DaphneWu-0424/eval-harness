# Agent 测评与 GPU 队列

## 容器边界

| 角色 | 默认镜像 | GPU | 职责 |
|---|---|---|---|
| 推理 | `lmsysorg/sglang:0.5.19-qwen38-runtime` | 每任务 1 张 | 加载 checkpoint，提供 HTTP 接口 |
| ALFWorld | `eval-harness/alfworld-env:20261116` | 无 | 环境、轨迹、判分 |
| WebShop | `eval-harness/webshop-env:20261116` | 无 | 环境、Lucene 检索、轨迹、判分 |
| ScienceWorld | `eval-harness/scienceworld-env:20261116` | 无 | SDK、Java 模拟器、轨迹、判分 |
| AppWorld | `ghcr.io/stonybrooknlp/appworld:latest` | 无 | SDK、Python REPL、应用数据库、官方测试判分 |

每个队列任务有独立推理容器、环境容器和 Docker 网络。环境通过 HTTP 请求推理服务，不挂载模型权重。ScienceWorld 只挂载项目和结果目录，不依赖原 EDGE。运行后保存日志、容器检查记录和解析后的配置，再清理属于该任务的容器与网络。

ScienceWorld 环境镜像复用同事的 `seq-grpo-3b:20261111`，保留 ScienceWorld 1.2.3、Py4J、Java 11 和打包的 JAR。镜像含有训练依赖，但测评进程只启动 SDK/JVM，不导入 Trainer、Ray 或 Torch。推理由独立 SGLang 镜像承担。

## 镜像与数据准备

以下命令在集群宿主机执行。n30285 已准备四个环境镜像；同步代码不会自动同步镜像或外部数据。

```bash
cd /ssd/eval-harness
# 从已有空闲旧环境容器创建 ALFWorld / WebShop 环境镜像
python3 scripts/snapshot_agent_envs.py
# 检查 SDK/Java/JAR，再为同事镜像建立 ScienceWorld 环境标签
python3 scripts/prepare_scienceworld_image.py
```

准备脚本不会启动训练。ScienceWorld 脚本使用无网络、无 GPU 的临时容器检查依赖，拒绝覆盖指向其他镜像 ID 的既有标签，并保存来源记录。

ALFWorld / WebShop 默认外部资源：

```text
/ssd/slime-ckpt/EDGE/agent_system/environments/env_package
/ssd/slime-ckpt/agent-cpu-smoke/alfworld_games_128.jsonl
/ssd/slime-ckpt/agent-cpu-smoke/webshop_sessions_500.jsonl
/ssd/slime-ckpt/agent-cpu-smoke/webshop/search_engine/indexes_1k
/ssd/slime-ckpt/agent-cpu-smoke/webshop/data -> /webshop_data/data
/ssd/datasets/alfworld
/ssd/datasets/webshop
```

ScienceWorld 固定清单已提交在 `configs/manifests/`。来源路径、版本和哈希保存在 `scienceworld-provenance.json`，无需读取训练目录。

## 一个 checkpoint 的测评

```bash
python3 scripts/run_agent_suite.py \
  --model-dir /path/to/hf-checkpoint --gpus 0 --benchmark scienceworld

# 使用已有服务
python3 scripts/run_agent_suite.py \
  --endpoint http://HOST:PORT/v1 --model-name current-checkpoint \
  --benchmark scienceworld
```

同一次 suite 选择多个数据集会共享模型服务并依次评测。需要多个数据集分别占卡时，使用队列入口。

## checkpoint × 数据集的动态队列

```bash
# 一个 checkpoint 在三个数据集上各形成一个 GPU 任务
CHECKPOINT=/path/to/ckpt bash scripts/run_checkpoint_sweep_parallel.sh \
  --benchmark alfworld --benchmark webshop --benchmark scienceworld

# 扫描 checkpoint 目录，限制 GPU 池
GPU_IDS=0,1,2,3,4,5,6,7 CHECKPOINT_DIR=/path/to/checkpoints \
  bash scripts/run_checkpoint_sweep_parallel.sh --benchmark scienceworld

# 显式指定多个 checkpoint
python3 scripts/run_checkpoint_sweep.py --parallel \
  --checkpoint /path/to/global_step_20 --checkpoint /path/to/global_step_40 \
  --benchmark scienceworld --gpus auto --max-parallel 4
```

每个任务使用 TP1、一张 GPU。队列检查显存和 GPU 计算进程，再通过宿主机文件锁租用空闲卡；哪张卡先空闲就领取下一项任务。多个本项目队列须共享默认锁目录 `/tmp/eval-harness-gpu-leases`。外部程序不参与该锁协议，因此不应同时向相同 GPU 提交其他工作。`--queue-timeout 0` 默认持续等待，正数可设置空闲等待上限。

HF checkpoint 直接使用；FSDP checkpoint 通过已有 `edge-dev` 容器串行导出，每个 checkpoint 只导出一次。可用 `--base-model`、`--merge-container`、`--merge-root` 调整。串行入口为 `scripts/run_checkpoint_sweep.sh`。

未指定数据集时默认选择四个环境。`--dataset NAME=CONFIG.yaml` 可为自定义配置指定唯一名称；配置路径相对于项目根目录（也支持绝对路径），清单必须位于声明的容器挂载内。`--steps` 可筛选 checkpoint step。

## ScienceWorld 协议与分数

默认配置只有一个：`configs/agent/scienceworld.yaml`。

| 参数 | 值 |
|---|---|
| 默认清单 | CLAIM，271 episodes / 30 task types |
| 可选清单 | in-run，149 episodes / 30 task types |
| 动作上限 | 50 次 agent 动作 |
| 生成 | greedy，temperature 0，top_p 1，最多 512 tokens |
| repetition penalty | 1.0 |
| 历史 | 最近 4 次 observation/action |
| prompt 预算 | 4096 tokens |
| 环境并发 | 每任务 4 个 episode，共用该任务的推理服务 |

采用同事的单环境 worker、固定 variation/seed、提示词和动作解析规则。每个 episode 创建独立 JVM，控制 Scala 随机种子和无序对象展示顺序。清单中的 `env_seed` 必须等于 `variation_id`。

prompt 为单条 user 消息，要求 `<think>...</think><action>...</action>`。动作小写化，检查格式，不额外用候选动作列表过滤。请求与回复写入轨迹。返回的 `usage.prompt_tokens` 缺失或超过 4096 时，该 episode 标记失败且不执行本次动作；不静默截断提示词。

分别记录以下指标：

- `scienceworld_score`：最终原始得分裁剪下限为 0，先在各任务类型内部平均，再对类型宏平均，单位 0–100。
- `scienceworld_success_rate`：最终原始得分等于 100 的比例，同样按类型宏平均；`accuracy` 为此值的兼容字段。
- `raw_score`、`retained_score`、`episode_return`：轨迹诊断字段。原生失败可能返回负分；之前保留的正分和累计奖励不充当最终任务得分。

报告包含每类分数、实际和预期的 episode/type 数量及 `full_panel`。基础设施或生成失败不会计成普通零分；存在未解决 episode 时总体指标留空并使任务失败。`--limit` 或 `--max-steps` 仅供冒烟，不代表完整清单结果。

若使用 in-run 清单，在项目内复制默认配置，只修改 `dataset.path` 为 `/eval_harness/configs/manifests/scienceworld-in-run.json`，使用 `--dataset sw-inrun=新配置路径`。不同 checkpoint step 共用配置，不为每个 step 复制 YAML。

原同事推理使用 vLLM，本项目使用 SGLang；环境与提示词对齐不代表不同后端生成结果逐 token 相同。

## AppWorld 协议

默认配置为 `configs/agent/appworld.yaml`，使用官方镜像，无需启动已有训练容器或端口池。SDK 位于镜像的 `/opt/venv/bin/python`，runner 显式使用这个解释器。

```bash
CHECKPOINT=/path/to/ckpt bash scripts/run_checkpoint_sweep_parallel.sh --benchmark appworld
```

默认数据集为 `test_normal`（168 tasks / 56 scenarios），可选 `test_challenge`（417 tasks / 139 scenarios）。固定 ID 清单和镜像/提示词哈希位于 `configs/manifests/appworld-*.json`。可复制默认配置并替换清单路径，通过 `--dataset aw-challenge=配置路径` 选择 challenge。

环境数据从宿主机 `/ssd/datasets/appworld/data` 只读挂载到 `/run/data`；SDK 输出数据库、交互日志和测试结果写入任务结果目录的 `<数据集名>-native/`。环境容器只挂载项目、数据和本任务结果，不挂载 GPU、权重或 Docker socket。运行时将镜像标签解析为实际 image ID，并保存到 `suite.json`。

当前验证版本为 AppWorld 0.1.3，image ID `sha256:f27cd05dfdef2afd696d05591ea4837ad6c8320b343874bfbe7b24eea5aedbf1`。`latest` 是可变标签；在另一台机器准备镜像后，先执行下方 SDK 检查，再运行正式测评。

```bash
docker run --rm --init --network none \
  -v /ssd/eval-harness:/eval_harness:ro \
  -v /ssd/datasets/appworld/data:/run/data:ro \
  -v /ssd/eval-results:/validation \
  --entrypoint /opt/venv/bin/python ghcr.io/stonybrooknlp/appworld:latest \
  /eval_harness/scripts/check_appworld_environment.py \
  --gold-train --output /validation/appworld-sdk-check.json
```

每任务默认 50 次代码执行、每次执行超时 60 秒，greedy 生成最多 2048 tokens，repetition penalty 1.0，episode 并发为 1。每个 episode 在独立子进程的主线程运行 SDK，隔离 AppWorld 的全局数据库、时间冻结、执行计时器和安全补丁；模型请求在父进程中运行。

提示词按现有 EDGE AppWorld 训练模板冻结，使用最近 4 次 code/result，历史文本按原实现截取最后 10000 字符。模型输出 `<think>...</think><code>...</code>`；提取 code 时保留 Python 大小写，格式错误按原 projection 使用末尾 100 字符并记录 invalid。此协议是现有训练协议，不声称复现官方 leaderboard agent 的完整配置。

`apis.supervisor.complete_task()` 只表示模型声明结束，不直接算成功。在声明结束或达到步数上限时执行官方 `world.evaluate()`；只有全部状态测试通过才成功。SDK/模型异常、测试缺失不会当作普通零分。对用户可见的观察不包含判分测试或参考答案。

- `appworld_tgc` / `accuracy`：Task Goal Completion，按 task 平均的成功率。
- `appworld_sgc`：Scenario Goal Completion，只有同一 scenario 的全部变体都成功才记 1；缺少变体或有未解决 episode 时留空。
- `appworld_full_panel`：是否覆盖完整任务清单。部分冒烟结果不代表完整测试集。

官方 API 说明见 [AppWorld README](https://github.com/StonyBrookNLP/appworld)，镜像见 [官方 GHCR 页面](https://github.com/StonyBrookNLP/appworld/pkgs/container/appworld)。

## 结果与验证

默认结果目录为 `/ssd/eval-results/separated/`。每次提交使用唯一 run-id，不覆盖已有结果。队列 `<run-id>-sweep/` 包含 `sweep.json`、`matrix.csv` 和每项任务日志；子任务目录包含 `suite.json`、配置、轨迹、判分、指标和报告。矩阵将成功率、ScienceWorld 的 0–100 得分及 AppWorld 的场景成功率分列保存。

2026-11-16 在 n30285 已验证：

- 从 30 类各取一个 episode，对比同事 worker 的 reset 和 `look / inventory / look` 三步，全部一致。
- 两个原生 gold 路径均达到 100 分；此 SDK 检查临时使用 1000 步上限，正式测评仍为 50 步。
- ScienceWorld CPU fake 流程及真实 SGLang 单 GPU 队列流程均完成，各 2 个 episode、最多 3 步，容器与网络正常清理。
- 三种环境的 CPU fake 回归均完成，每种各 2 个 episode、最多 3 步，全部正常判分和清理。
- ScienceWorld 镜像内完整 Linux 单元测试 77 项通过。测试启动时清空镜像继承的 `BASH_ENV` / `ENV`，避免镜像的 shell 初始化递归调用 mock `nvidia-smi`；正式环境入口直接运行 Python。

原生验证工具为 `scripts/check_scienceworld_environment.py`。尚未运行完整 271 个 episode 的模型测评。

AppWorld 同日验证：官方镜像内 Linux 单元测试 84 项通过；CPU fake 测试 2 个任务各 2 步、真实 SGLang 单 GPU 队列测试 1 个任务 2 步均完成。训练集一个原生 gold 路径通过全部 2 项测试，未执行时判定失败。尚未运行完整 168 或 417 个任务的模型测评。
