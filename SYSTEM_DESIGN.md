# 通用 Model / Agent Evaluation Harness 系统设计

## 1. 设计目标

本系统是一个通用、组件化、trajectory-centric 的模型与 Agent 评测框架。第一阶段完整支持 Reasoning / Math，但从 core schema、执行抽象和 pipeline 边界开始，就把 Agent 作为一等任务类型保留。

用户默认只执行一条命令：

```bash
eval-harness run --config configs/reasoning/pcl_qwen35_9b_base.yaml
```

系统完成：

```text
配置解析
  -> 数据集加载
  -> 模型 / Agent 执行
  -> Raw Trajectory 持久化
  -> Evaluator / Judge
  -> Metrics 聚合
  -> Text + JSON Report
```

内部阶段保持解耦，使推理、Judge、Metrics 和报告可独立恢复或重跑。

## 2. 关键设计原则

### 2.1 Trajectory 是核心事实记录

核心不是 `question -> answer`，而是：

```text
TaskSample -> Trajectory -> EvaluationResult
```

Reasoning trajectory 可以只有 prompt、reasoning 和 final answer；Agent trajectory 可以包含多轮消息、tool call/result、action 与 environment observation。两者使用同一个顶层 pipeline。

### 2.2 Core 不感知具体任务假设

以下假设只能存在于 reasoning-specific 组件，不得进入 core：

- reference 一定是字符串；
- 一定有 final answer；
- 一定是单轮纯文本；
- 一定包含 `\\boxed{}`；
- evaluation 一定是二元值；
- trajectory 一定只有一次模型调用。

### 2.3 Raw 数据不可变，派生结果可重建

Raw trajectory 是执行阶段的事实记录，采用 append-only 存储。Evaluator 不得覆盖 raw；evaluated、metrics 和 report 都应能从 raw 与 resolved config 重建。

### 2.4 一条龙 UX，阶段化内部实现

`run` 是正式入口。`execute`、`judge`、`report` 是高级入口，用于断点恢复、替换 Judge、重算 Metrics 和调试，不要求普通用户手工串联。

### 2.5 宿主机控制面与一次性容器执行面分离

任务 runner 在远端宿主机上控制生命周期；本次任务的 SGLang server 与 harness 在同一个一次性容器内运行。结果目录通过 bind mount 写回 `/ssd/`。容器是可丢弃的，实验数据不是。

## 3. 系统上下文与部署边界

```text
本地开发机 D:\eval-harness
  |
  | scp（Review 后才执行）
  v
n30283:/ssd/eval-harness                 宿主机控制面
  |
  | task runner: preflight/create/wait/exec/cleanup
  v
+----------------------------------------------------------------+
| 一次性 Docker Container: eval-harness-<run_id>                 |
|                                                                |
|  SGLang 0.5.19 TP8 server <-> OpenAI-compatible Backend        |
|                                      |                         |
|                              Evaluation Pipeline               |
|                                      |                         |
|             Dataset -> Executor -> Trajectory -> Evaluator     |
|                                              -> Metrics/Report  |
+----------------------------------------------------------------+
       | read-only mount                         | writable mount
       v                                         v
  /ssd/models, /ssd/datasets              /ssd/eval-results/<run>
```

部署约束：

- 开发和无 GPU 测试先在本地完成。
- GPU、Docker、真实 SGLang 测试只在 `n30283` 集群 `/ssd/` 范围进行。
- 模型固定为 `Qwen3.5-9B`，宿主机权重路径为 `/ssd/models/Qwen3.5-9B`。
- 镜像使用 `lmsysorg/sglang:0.5.19-qwen38-runtime`。2026-09-10 在远端查询到 image ID `sha256:aaf1e4cbe15938736b50697b2891f866da4ce919dea12e9458648fe4402e419f`，该本地镜像当前没有 registry digest，因此运行时记录实际 image ID。
- 第一阶段使用单节点全部 8 卡启动一个 TP8 SGLang server，不做多节点或多 endpoint。
- 模型和数据只读挂载；实验结果、日志和状态可写挂载。
- 容器删除后所有实验 artifacts 必须仍保留在宿主机。

## 4. 分层架构

```text
CLI / Config
    |
EvaluationPipeline ------------------------------------+
    |                                                  |
Dataset -> TaskSample                                  |
    |                                                  |
Executor                                               |
    +-> ModelExecutor -> PromptBuilder -> ModelBackend |
    +-> AgentExecutor -> AgentHarness + Environment    | future
    |                                                  |
Trajectory + TrajectoryEvent                           |
    |                                                  |
Evaluator -> EvaluationResult                          |
    |                                                  |
Metrics -> MetricResult                                |
    |                                                  |
Reporters -> summary.txt / summary.json                |
    |                                                  |
Persistence / Resume / Metadata -----------------------+

宿主机 Runtime Supervisor
    -> Docker lifecycle
    -> SGLang readiness
    -> container exec
    -> guaranteed cleanup
```

Runtime Supervisor 是 pipeline 外围的部署层。Pipeline 只消费可用的 backend endpoint，不负责创建或删除 Docker 容器。这样本地 FakeBackend、外部 API、长期服务和一次性 SGLang 都可以复用同一个 pipeline。

## 5. 核心领域模型

### 5.1 TaskSample

统一承载任务输入：

```python
TaskSample(
    sample_id: str,
    input: Any,
    reference: Any | None,
    source: str | None,
    metadata: dict,
)
```

Dataset 只负责 raw record 到 `TaskSample`，不负责执行或判分。

### 5.2 TrajectoryEvent

事件是可扩展的最小轨迹单元：

```python
TrajectoryEvent(
    type: str,
    role: str | None,
    content: Any,
    timestamp: float | None,
    metadata: dict,
)
```

第一阶段可使用较简单的 reasoning events，但 schema 必须支持 tool call/result、action 和 environment observation。

### 5.3 Trajectory

```python
Trajectory(
    sample_id: str,
    rollout_id: int,
    events: list[TrajectoryEvent],
    final_output: Any | None,
    status: str,
    usage: dict,
    timing: dict,
    metadata: dict,
    error: str | None,
)
```

`(sample_id, rollout_id)` 是稳定逻辑主键。`reasoning`、`answer` 等便利属性可以存在，但底层必须可映射到 events/final_output。

### 5.4 ModelResponse

后端响应必须分别保存 reasoning、content 和 tool calls，不能提前拼成一个字符串：

```python
ModelResponse(
    reasoning: str | None,
    content: str | None,
    tool_calls: list | None,
    finish_reason: str | None,
    usage: dict,
    raw_response: Any,
    metadata: dict,
)
```

### 5.5 EvaluationResult

```python
EvaluationResult(
    sample_id: str,
    rollout_id: int,
    status: str,
    reward: float | None,
    scores: dict[str, float],
    labels: dict[str, Any],
    details: dict,
    evaluator: str,
)
```

这一结构既能表示 math correctness，也能表示未来 Agent 的 task success、tool validity、efficiency 等连续或多维指标。

## 6. 组件职责与扩展点

### Dataset

- 输入：原始文件或数据源。
- 输出：`TaskSample`。
- 不调用模型，不构造 trajectory，不 Judge。

### Executor

统一接口为异步 `run(sample, rollout_id) -> Trajectory`。

- `ModelExecutor`：PromptBuilder -> ModelBackend -> Trajectory。
- `AgentExecutor`：未来组合 ModelBackend、Environment、AgentHarness 和 max steps。

Pipeline 不区分两者内部实现。

### ModelBackend

统一异步生成接口。第一阶段实现 OpenAI-compatible backend；SGLang 与 vLLM 尽量共享该实现，只通过 capability adapter 处理差异。

### Evaluator

接收 `TaskSample + Trajectory`，返回 `EvaluationResult`。Math Evaluator 内部组合 AnswerExtractor 和 rule judge。判定为三态：

- `CORRECT`：可可靠证明答案等价；
- `INCORRECT`：已成功提取且可可靠证明不等价；
- `UNKNOWN`：提取失败、解析失败或规则无法可靠决定。

正式配置在 rule verdict 为 `UNKNOWN` 或 `INCORRECT` 时调用异源 DeepSeek LLM Judge。Judge 只接收 problem、reference、提取后的 candidate answer 和 rule verdict，不接收完整长 reasoning。最终 `EvaluationResult` 同时保留 rule verdict、DeepSeek verdict、reason、model、usage、request metadata 与错误；DeepSeek 返回 UNKNOWN 或调用失败时不抹掉已有的可靠 rule verdict。

DeepSeek 使用独立 OpenAI-compatible backend，默认模型为 `deepseek-chat`，与被评测的 Qwen3.5/SGLang backend 完全分离。API key 和 base URL 仅从 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 环境变量读取，禁止落入 resolved config 和 artifacts。

### Metric

对 samples、trajectories 和 evaluations 聚合，独立于 Evaluator。第一阶段包括 accuracy、avg@n、pass@k、长度、截断、generation error、judge coverage 和 per-source。

### Registry

Dataset、Executor、Backend、PromptBuilder、Evaluator、Metric 和 Reporter 均通过 registry 按配置构建，避免在 pipeline 中散布类型分支。

## 7. 一次性容器生命周期设计

### 7.1 生命周期状态机

```text
INIT
  -> PREFLIGHT_OK
  -> CONTAINER_CREATED
  -> SGLANG_STARTING
  -> READY
  -> PIPELINE_RUNNING
  -> PIPELINE_FINISHED | FAILED | INTERRUPTED
  -> CLEANING
  -> CONTAINER_REMOVED
  -> EXIT
```

只要进入 `CONTAINER_CREATED`，之后所有终止路径都必须经过 `CLEANING`。

### 7.2 任务启动

宿主机 task runner 负责：

1. 解析最小 runtime 配置并生成不可碰撞的 `run_id`。
2. 校验镜像、模型、数据、输出目录、GPU 和端口。
3. 注册 EXIT/SIGINT/SIGTERM cleanup handler。
4. 以精确名称和 labels 创建容器。
5. 在容器内启动 SGLang server，并把日志写到宿主机挂载目录。
6. 通过 `docker exec` 在容器内轮询 localhost `/health`，直到 TP8 server ready 或启动超时。
7. 在同一容器内执行 `eval-harness run --config ...`。

建议容器身份：

```text
name:  eval-harness-<run_id>
label: eval-harness.managed=true
label: eval-harness.run_id=<run_id>
label: eval-harness.owner=<configured-owner>
```

### 7.3 任务结束与强制清理

cleanup 必须满足：

- 使用 runner 保存的精确 container ID/name；
- 先验证 label/run ID，避免误删其他人的容器；
- `docker rm -f` 幂等执行；
- 正常结束、模型启动失败、health timeout、pipeline exception、Ctrl+C、SIGTERM 都会触发；
- 清理不覆盖 pipeline 原始退出码；
- 清理失败时输出容器 ID、原因和精确人工清理命令，并以非零状态结束；
- 不清理模型、数据或实验输出 bind mount。

容器内部不应作为唯一 cleanup 责任方，因为容器内进程无法可靠删除自身。清理由宿主机 supervisor 保证。

### 7.4 为什么 SGLang 与 harness 放在同一容器

这符合“脚本启动时创建 container，在 container 里执行指令，结束后直接 rm”的要求，并带来：

- 依赖和版本封闭，运行可复现；
- pipeline 可直接访问 localhost endpoints；
- 容器生命周期与评测任务一一对应；
- 不遗留常驻 SGLang 服务占用 GPU。

控制脚本仍在宿主机执行，因为它必须在容器退出或异常时负责删除容器。

## 8. 单节点 TP8 Endpoint 与并发设计

第一阶段明确采用单节点 8 卡、一个 SGLang server、`tensor-parallel-size=8` 和一个容器内 endpoint：

```text
startup: 检查唯一 endpoint 的 /health
dispatch: global asyncio semaphore 控制并发
failure: 按错误类型执行有限 retry
metadata: 记录 endpoint、attempt、latency 和错误类别
```

当前不实现多节点、8 个 TP1 server、round-robin 或 endpoint 故障隔离。Backend 配置仍使用 endpoint 列表形式，保留未来扩展能力；后续增加 multi-endpoint scheduler 不改变 Executor、Trajectory 或 Pipeline 接口。

## 9. Pipeline 执行流程

```text
1. Resolve and validate config
2. Compute config/dataset/execution fingerprints
3. Create or validate experiment directory
4. Load TaskSamples
5. Build Executor
6. Discover completed rollouts from raw.jsonl
7. Execute only missing/failed rollouts
8. Append and flush every completed Trajectory
9. Evaluate trajectories not yet evaluated under current evaluator fingerprint
10. Persist EvaluationResults
11. Compute metrics
12. Atomically write reports and final state
```

`n` 表示每个 sample 的 rollout 数。调度单位是 `(sample_id, rollout_id)`，因此能够精确 resume。

## 10. 持久化与恢复

建议实验目录：

```text
/ssd/eval-results/<experiment>/<run_id>/
|-- raw.jsonl
|-- evaluated.jsonl
|-- summary.json
|-- summary.txt
|-- config.yaml
|-- metadata.json
|-- state.json
`-- logs/
    |-- runner.log
    |-- sglang.log
    `-- harness.log
```

存储规则：

- `raw.jsonl` 只追加，不由 Judge 修改。
- 并发 worker 通过单 writer queue 或等价串行写入点落盘。
- 每条完整 JSONL 记录写入后 flush；需要时配置 fsync 策略。
- `state.json`、metadata 和 summary 使用临时文件 + 原子 replace。
- resume 扫描稳定主键和 status；只有成功 trajectory 视为完成。
- 文件尾部半条记录可被识别为 crash residue；中间损坏必须 fail-fast。
- config/dataset/execution fingerprint 不兼容时禁止误续跑。
- evaluator 或 metrics fingerprint 改变时复用 raw，仅重跑相应下游阶段。

Stage state 示例：

```json
{
  "execution": "completed",
  "evaluation": "running",
  "metrics": "pending",
  "report": "pending"
}
```

## 11. 配置设计

配置完整描述一次实验，但 core pipeline 配置与部署 runtime 配置保持清晰边界：

```yaml
experiment:
  name: pcl_qwen35_9b_base
  output_dir: /workspace/results/pcl_qwen35_9b_base
  seed: 20260910

task:
  type: reasoning

dataset:
  type: jsonl
  path: /workspace/datasets/pcl_test.jsonl
  input_key: problem
  reference_key: ground_truth
  source_key: source

executor:
  type: model

prompt:
  type: math_boxed

backend:
  type: openai_compatible
  model: Qwen3.5-9B
  endpoints:
    - http://127.0.0.1:31000/v1

generation:
  n: 1
  temperature: 1.0
  top_p: 0.95
  top_k: 20
  presence_penalty: 1.5
  max_tokens: 81920
  concurrency: 32
  timeout_sec: 7200
  retries: 5

evaluators:
  - type: math
    extractor:
      type: composite_math
    llm_fallback:
      enabled: false

metrics:
  - type: accuracy
  - type: pass_at_k
    k: [1]
  - type: avg_at_k
  - type: length
  - type: truncation
  - type: per_source
```

宿主机 runtime 可放在同一 resolved config 的独立顶层段，或由 wrapper config 引用：

```yaml
runtime:
  type: ephemeral_docker
  image: lmsysorg/sglang:0.5.19-qwen38-runtime
  container_name_prefix: eval-harness
  network: bridge
  remove_on_exit: true
  startup_timeout_sec: 1800
  gpus: all
  ipc: host
  sglang:
    tensor_parallel_size: 8
    model_path: /workspace/models/Qwen3.5-9B
    endpoint: http://127.0.0.1:31000/v1
    context_length: 131072       # 为 max_tokens=81920 加上输入空间
  mounts:
    project: /ssd/eval-harness:/workspace/eval-harness
    models: /ssd/models/Qwen3.5-9B:/workspace/models/Qwen3.5-9B:ro
    datasets: /ssd/datasets:/workspace/datasets:ro
    results: /ssd/eval-results:/workspace/results
```

实际字段在实现前根据集群约束 Review；这里表达的是职责和边界，不是最终 Docker 命令。

## 12. Math 评测语义

AnswerExtractor 顺序建议为：

1. 最后一个合法 `\\boxed{...}`；
2. `Final Answer:`；
3. `Answer:`；
4. Markdown 强调形式；
5. 末尾短数学表达式；
6. 明确记录 fallback/失败。

Rule judge 组合：

- normalized exact match；
- 数值与分数等价；
- SymPy 表达式等价；
- tuple/set/interval 等结构等价；
- 无法可靠解析或判断时返回 UNKNOWN。

必须测试 `455/4 == 113.75` 与 `sqrt(6)/2 == sqrt(3/2)`。第一阶段 LLM fallback 接口存在但默认关闭。

## 13. Metrics 语义

每题有 `n` 个 rollout、其中 `c` 个正确时：

```text
avg@n = c / n
pass@k = 1 - C(n-c, k) / C(n, k)
```

dataset level 对样本指标取平均。实现时必须显式规定：

- generation failure 是否占计划 rollout denominator；
- UNKNOWN 是否按不正确计入能力指标，同时单独报告 coverage；
- rollout 少于 k 时的行为；
- accuracy 与 pass@1 的定义是否完全对齐。

推荐报告计划数、成功生成数、有效判定数与 UNKNOWN 数，使 denominator 可审计。

## 14. 失败模型与可观测性

### 执行错误分类

- connection/read error；
- timeout；
- HTTP 5xx；
- invalid response；
- empty response；
- retry exhausted；
- cancelled/interrupted。

错误 trajectory 仍需落盘，包含 attempts、endpoint、timing 和错误摘要；但 resume 不将其视为成功完成。

### Runtime 错误分类

- preflight failure；
- image/container create failure；
- SGLang startup/health timeout；
- harness non-zero exit；
- signal interruption；
- container cleanup failure。

`runner.log` 记录状态转换、精确容器身份和 cleanup 结果，不记录不必要的 secrets。

## 15. 建议目录结构

```text
eval-harness/
|-- pyproject.toml
|-- README.md
|-- TODO.md
|-- SYSTEM_DESIGN.md
|-- configs/
|   |-- reasoning/
|   |-- coding/
|   `-- agents/
|-- scripts/
|   `-- runtime/                 # 宿主机一次性容器 runner
|-- eval_harness/
|   |-- cli.py
|   |-- config.py
|   |-- core/
|   |-- datasets/
|   |-- executors/
|   |-- backends/
|   |-- environments/
|   |-- harnesses/
|   |-- prompts/
|   |-- extractors/
|   |-- evaluators/
|   |-- metrics/
|   |-- pipeline/
|   |-- persistence/
|   |-- reports/
|   `-- utils/
`-- tests/
    |-- unit/
    |-- integration/
    `-- e2e/
```

`scripts/runtime` 属于 deployment adapter，不属于 `eval_harness/core`。

## 16. 测试策略

测试由低成本到真实环境逐层推进：

```text
Unit tests
  -> FakeBackend E2E（3 samples，无 GPU）
  -> Remote non-GPU tests in container
  -> Container lifecycle fault injection
  -> Real SGLang smoke（2–5 samples）
  -> Resume smoke
  -> PCL 500 formal run
```

容器 lifecycle 必须覆盖正常退出、启动失败、health timeout、pipeline 失败、SIGINT、SIGTERM 和 cleanup 重入。每个 case 都验证：

- 本次容器已删除；
- 原始退出状态可追踪；
- `/ssd/` artifacts 保留；
- 不影响其他容器。

## 17. 第一阶段交付范围

必须完整实现：

- one-command pipeline；
- Dataset/TaskSample/Trajectory 等 core schema；
- ModelExecutor 与 OpenAI-compatible backend；
- 单节点 8 卡 TP8 SGLang endpoint，并为未来 multi-endpoint 保留配置接口；
- task-scoped ephemeral container supervisor；
- n rollouts、append-only raw 与 resume；
- Math extractor 和三态 evaluator；
- accuracy、avg@n、pass@k、长度、截断、per-source 等 metrics；
- text/JSON reports 与 basic inspect；
- unit、Fake E2E 和真实 SGLang smoke。

只预留接口：

- AgentExecutor；
- Environment 与 NullEnvironment；
- AgentHarness；
- Agent/tool-use events 与 metrics；
- LLM judge fallback。

明确不引入 Ray、数据库、Celery、Kafka、Web UI、微服务或复杂分布式 controller。第一阶段使用 Python asyncio + JSONL 即可。

## 18. Review 待确认项

以下信息不会改变核心架构，但会决定 runtime 配置和首次 smoke test：

1. 是否接受当前容器默认值：bridge network、不发布宿主机端口、容器内 health check、`--ipc=host`。
2. 是否还有除项目、模型、数据集和结果目录之外的额外挂载。

已确认模型、模型路径、镜像、单机 8 卡 TP8、单 endpoint、数据集和远端项目目录。
