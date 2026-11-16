# Trajectory Evaluation Harness

通用、组件化、以轨迹为核心的模型与 Agent 测评框架。目前支持 Reasoning / Math，以及 ALFWorld、WebShop、ScienceWorld、AppWorld。

## Agent 测评

推理和环境在不同容器中运行。推理容器使用 SGLang 并占用 GPU；环境容器只使用 CPU，通过 HTTP 请求推理服务。环境镜像、协议、数据准备和完整用法见 [AGENT_EVAL.md](AGENT_EVAL.md)。

并行队列将一个 checkpoint × 一个数据集作为一个任务，每个任务占一张 GPU。空闲 GPU 自动领取下一个任务；可以指定一个 checkpoint，也可以扫描目录中的 checkpoint：

```bash
cd /ssd/eval-harness
CHECKPOINT=/path/to/ckpt bash scripts/run_checkpoint_sweep_parallel.sh \
  --benchmark scienceworld

CHECKPOINT_DIR=/path/to/checkpoints bash scripts/run_checkpoint_sweep_parallel.sh \
  --benchmark alfworld --benchmark webshop --benchmark scienceworld --benchmark appworld
```

`--checkpoint` 和 `--benchmark` 均可重复指定，`--dataset NAME=CONFIG.yaml` 支持同一环境的不同数据集。默认使用全部可见 GPU；`GPU_IDS=0,1,2,3` 可限制池子。未指定数据集时运行以上四个环境，因此首次使用建议显式选择。

ScienceWorld 默认采用同事固定的 CLAIM 测试清单：30 类、271 个 episode、每个最多 50 次动作。分别报告按任务类型宏平均的任务得分（0–100）和成功率（原始得分达到 100）。已验证原生环境一致性与真实推理冒烟；尚未运行完整 271 个 episode 的模型测评。

AppWorld 使用 `ghcr.io/stonybrooknlp/appworld:latest`，默认 `test_normal` 的 168 个任务、每任务 50 步。通过官方 SDK 执行 Python/API 调用并用官方状态测试判分，分别报告 TGC 和 SGC；环境数据挂载自 `/ssd/datasets/appworld/data`。

```bash
CHECKPOINT=/path/to/ckpt bash scripts/run_checkpoint_sweep_parallel.sh --benchmark appworld
```

## Reasoning / Math

```bash
python -m eval_harness run --config configs/reasoning/pcl_qwen35_9b_base.yaml
```

远端可通过 `scripts/run_ephemeral_sglang.sh` 创建一次性 SGLang TP8 容器，任务结束后删除容器，结果保留在 `/ssd/eval-results`。已验证的 smoke 配置使用 Qwen3.5-9B、TP8、server context 131,072 和最大输出 81,920 tokens；上下文必须容纳输入及输出预算。

正式配置对 Math rule 的 `UNKNOWN` 和 `INCORRECT` 使用 `deepseek-chat` 二次判分。凭据只从环境变量 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 读取；可放入远端 `/ssd/eval-harness/.secrets/deepseek.env`，权限设为 `600`。该目录已被 Git 忽略，runner 会自动加载。

架构说明见 [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) 和 [TODO.md](TODO.md)。本地分析用 HTML、数据目录和临时文件不提交。
