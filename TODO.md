# Evaluation Harness 实施 TODO

## 实施状态（2026-09-10）

- [x] 本地实现已同步至 `n30283:/ssd/eval-harness`。
- [x] 远端 SGLang 镜像内 compile、unit、Fake E2E 和 mock-Docker lifecycle 共 31 项测试全部通过。
- [x] 真实 PCL test 已验证为 500 条，SHA256 为 `d8322c1485de9337effbbf05163487cd2d30b46a42c49ff292c8e7031e88b773`。
- [x] runner 的成功、pipeline 非零退出、SIGINT=130、GPU 占用 preflight 和精确容器清理路径已验证。
- [x] 经授权仅停止、未删除 `qwen38-teacher-a`、`qwen38-teacher-b`、`slime16`，释放单机 8 卡用于真实 smoke；三者当前仍为 stopped，未擅自重启。
- [x] 真实 Qwen3.5-9B TP8 3-sample smoke 成功：3/3 generation 完成，generation error 0%，truncation 0%，平均 completion 43,316.67 tokens。
- [x] 真实 resume smoke 成功：raw 保持 6 行，没有重复生成；Evaluator implementation 升级只追加重判记录。
- [x] 两次真实任务结束后，一次性 harness 容器均被删除，`eval-harness.managed=true` 容器列表为空。
- [ ] 是否启动 PCL 500 正式评测由用户确认；该任务可能消耗较长 GPU 时间。
- [x] 已实现异源 DeepSeek LLM Judge，对 rule `UNKNOWN` 和 `INCORRECT` 兜底；远端 35 项测试通过。
- [ ] DeepSeek 真实 smoke 与 PCL 500 正式评测被凭据阻塞：本地和 `n30283` 均未发现可导出的 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`，需用户提供安全加载位置。

## 0. 当前共识与实施边界

- [x] 项目定位为通用、组件化、trajectory-centric 的 Model / Agent Evaluation Harness。
- [x] 顶层统一采用 `Task/Dataset -> Execution -> Trajectory -> Evaluator -> Metrics -> Report`。
- [x] 第一阶段完整实现 Reasoning / Math；Agent、Environment、Harness 只提供稳定接口与 schema 骨架。
- [x] 默认用户体验是一条命令跑完整 pipeline，同时保留 `execute`、`judge`、`report` 等阶段命令用于恢复和调试。
- [x] 本地目录仅用于设计与开发；真实 SGLang/GPU 测试在 `n30283` 集群的 `/ssd/` 下进行。
- [x] SGLang 采用任务级一次性容器：任务脚本启动时创建容器，任务结束或异常退出时删除容器。
- [x] 第一阶段模型固定为 `Qwen3.5-9B`，远端权重路径为 `/ssd/models/Qwen3.5-9B`。
- [x] 第一阶段使用单节点 8 卡、单 SGLang 实例、tensor parallel 8 和单 endpoint；暂不做多节点/多 endpoint。
- [x] 远端项目目录采用 `/ssd/eval-harness`。
- [x] 远端镜像确认为 `lmsysorg/sglang:0.5.19-qwen38-runtime`；2026-09-10 查询到的本地 image ID 为 `sha256:aaf1e4cbe15938736b50697b2891f866da4ce919dea12e9458648fe4402e419f`，无 registry digest。
- [x] 本轮只交付设计文档，不编写实现代码、不传输文件、不启动远端容器。

## 1. Review 后首先确认的运行参数

- [x] 远端项目目录：`/ssd/eval-harness`。
- [x] 模型：`Qwen3.5-9B`。
- [x] 模型权重宿主机路径：`/ssd/models/Qwen3.5-9B`；建议容器内只读挂载为 `/workspace/models/Qwen3.5-9B`。
- [x] 镜像：`lmsysorg/sglang:0.5.19-qwen38-runtime`；远端已确认存在。
- [x] GPU 拓扑：单节点 8 卡，第一阶段启动一个 `tensor-parallel-size=8` 的 SGLang server，暴露一个容器内 endpoint。
- [x] 数据集根目录：`/ssd/datasets/PCL-Reasoner-V1.5-RL-Math`；正式测试文件已确认为 `split/test.jsonl`。
- [ ] 确认容器运行参数。当前建议：默认 bridge network、不发布宿主机端口，runner 通过 `docker exec` 在容器内检查 localhost `/health`；单机 8 卡使用 `--ipc=host`；额外挂载仅在数据集/输出路径确认后补充。

## 2. 项目骨架与工程约束

- [x] 初始化 Python 项目、依赖管理、CLI entry point 和测试配置。
- [x] 创建 `configs/`、`eval_harness/`、`tests/`、`scripts/` 的基础目录。
- [x] 建立 core 与 reasoning-specific 模块边界，禁止将 `ground_truth string`、`final answer`、单轮文本、`\\boxed{}`、二元正确率等假设写入 core。
- [x] 规定 JSON/JSONL schema version，所有持久化对象包含 schema version。
- [x] 实现统一 registry，供 Dataset、Executor、Backend、Evaluator、Metric 等组件按配置构建。
- [ ] 定义稳定错误分类与状态枚举，避免业务逻辑依赖自由文本错误。

## 3. 核心数据模型

- [x] 实现 `TaskSample`，允许任意输入、可选 reference、source 与 metadata。
- [x] 实现 `TrajectoryEvent`，支持消息、reasoning、tool call/result、action、observation 等扩展事件。
- [x] 实现 `Trajectory`，包含 events、final output、status、usage、timing、metadata 和 error。
- [x] 实现 `ModelResponse`，分别保留 reasoning、content、tool calls、finish reason、usage 与 raw response。
- [x] 实现 `EvaluationResult`，支持 reward、多个连续 score、labels、details 与 evaluator provenance。
- [x] 为所有核心对象实现无损序列化、反序列化与历史 raw 兼容入口。
- [x] 保证 `(sample_id, rollout_id)` 是 rollout 的稳定逻辑主键。

## 4. 配置与实验身份

- [x] 实现 YAML 配置加载、默认值与严格基础校验；复杂 CLI override 尚未加入。
- [x] 保存 resolved config，计算 config hash。
- [x] 区分 execution fingerprint 与 evaluator fingerprint，Metrics 变化不会导致 raw 失效。
- [ ] 配置应完整描述 dataset、executor、prompt、backend、generation、evaluators、metrics、reports 和 runtime。
- [ ] 配置中加入一次性容器 runtime 段，但不把 Docker 参数写入 core model backend。
- [ ] 自动记录 dataset 路径、SHA256、样本数、模型、镜像 digest、生成参数、框架版本、Git commit、开始/结束时间。
- [ ] 防止同名实验在 fingerprint 不兼容时误续跑；必须报错或创建新 run。

## 5. Dataset 与 Prompt

- [ ] 定义 Dataset 抽象：只负责 raw record 到 `TaskSample` 的转换。
- [x] 实现通用 JSONL dataset adapter。
- [x] 实现 PCL reasoning 配置并验证 500 条真实 test 数据。
- [ ] 校验 sample ID 唯一、字段存在、测试集样本数与 SHA256。
- [ ] 定义 PromptBuilder 抽象。
- [ ] 实现 math boxed prompt，提示逐步推理并将最终答案放入 `\\boxed{}`。

## 6. Executor 与模型 Backend

- [x] 定义异步 `Executor.run(sample, rollout_id) -> Trajectory`。
- [x] 实现 `ModelExecutor`：PromptBuilder -> ModelBackend -> Trajectory。
- [x] 定义异步 `ModelBackend.generate(messages, generation_config) -> ModelResponse`。
- [x] 实现 OpenAI-compatible backend，分别解析 reasoning/content/tool calls/usage/finish reason。
- [ ] 第一阶段实现单 endpoint client、启动前 health check 和运行期错误处理；multi-endpoint scheduler 延后。
- [ ] 支持 global concurrency；为后续 multi-endpoint/per-endpoint concurrency 保留配置和接口。
- [ ] 支持 timeout、指数退避/抖动、最大重试次数与可重试错误分类。
- [ ] 将 ReadError、timeout、connection reset、HTTP 5xx、invalid/empty response 写入失败 trajectory。
- [ ] 确保失败 rollout 在 resume 时仍是待完成项，而不是被误判为成功完成。

## 7. 远端一次性 SGLang 容器生命周期

- [x] 提供宿主机侧 task runner，负责容器生命周期和 pipeline 编排。
- [ ] 每次任务生成唯一 `run_id` 与容器名，并增加 project/run/owner labels。
- [ ] 在容器创建前检查 image、模型路径、数据路径、输出路径、GPU 和端口是否可用。
- [ ] 使用 `lmsysorg/sglang:0.5.19-qwen38-runtime`，每次运行记录实际 image ID；如后续获得 registry digest，再固定并记录 digest。
- [ ] 启动容器时明确 GPU、network、IPC、shared memory、ulimit、只读模型/数据挂载和可写结果挂载。
- [ ] 容器成为本次任务唯一执行面：SGLang server 与 harness 命令都在该容器内执行。
- [ ] 等待 `/health` ready 后才开始 dataset execution，设置启动超时并保存启动日志。
- [x] 正常结束、pipeline 失败、启动失败、SIGINT、SIGTERM 和 shell 退出时均执行幂等 `docker rm -f <exact-container-name>`；真实 SIGTERM fault injection 待 GPU smoke 阶段补充。
- [ ] 清理逻辑不得依赖模糊 name filter；只清理由本次 runner 创建且 label/run ID 匹配的精确容器。
- [ ] cleanup 保留任务原始退出码；容器删除失败要显式报警并给出精确人工清理命令。
- [ ] 原始 trajectory、state、report 和日志写到宿主机 `/ssd/` 挂载目录，容器删除后仍永久保留。
- [ ] 在任务启动时检测同一 `run_id` 的遗留容器，并按明确策略 fail-fast 或只清理可验证属于该 run 的容器。
- [ ] 验证任务结束后 `docker ps -a` 中无本次 run 的容器残留。

## 8. Pipeline、持久化与断点恢复

- [x] 实现统一 `EvaluationPipeline.run()`：resolve -> load -> execute -> evaluate -> aggregate -> report。
- [ ] 保留 `execute`、`judge`、`report` 子命令，但 `run` 是默认入口。
- [ ] 设计实验目录：`raw.jsonl`、`evaluated.jsonl`、`summary.json`、`summary.txt`、`config.yaml`、`metadata.json`、`state.json`、`logs/`。
- [x] raw trajectory 采用 append-only 语义，Judge 永不覆盖 raw。
- [ ] 使用单 writer 或等价机制避免高并发 JSONL 行交叉写入。
- [ ] 每条记录 flush，并采用临时文件 + 原子 replace 更新小型 state/summary 文件。
- [x] resume 扫描并校验现有 raw，修复最后一条 crash residue，不静默忽略中间损坏。
- [ ] 基于稳定主键跳过已成功 rollout，只补跑缺失或失败 rollout。
- [ ] generation 完成而 judge 中断时，不重新调用模型，只继续 evaluate/metrics/report。
- [ ] stage state 至少支持 pending/running/completed/failed，并记录计数与错误摘要。
- [ ] 支持把历史 raw trajectory 导入统一 schema 后直接重新 judge。

## 9. Math Extractor 与 Evaluator

- [x] 定义 AnswerExtractor 语义与结构化 ExtractResult。
- [x] 实现组合数学答案提取：最后一个 `\\boxed{}`、Final Answer、Answer、Markdown Answer、末尾短表达式、fallback。
- [ ] 记录 extracted answer、method、confidence 和失败原因。
- [x] 实现三态 Math Evaluator：`CORRECT` / `INCORRECT` / `UNKNOWN`。
- [ ] 明确区分“无法提取/无法可靠判断”与“已判断不等价”。
- [ ] 复用或适配底层 math/sympy grader，并补充 normalized exact、numeric、fraction、tuple/set/interval 等价判断。
- [ ] LLM judge fallback 仅预留接口，第一阶段默认关闭。

## 10. Metrics 与报告

- [x] Metrics Registry 与 Evaluator 分离。
- [x] 实现 accuracy、avg@n、标准无偏 pass@k。
- [ ] 实现 response/reasoning length 的 mean、median、p90、p95。
- [ ] 实现 truncation rate、generation error rate、judge coverage 和 per-source metrics。
- [ ] 明确缺失 rollout、UNKNOWN、失败 trajectory 在各指标中的 denominator 规则。
- [x] 生成面向阅读的 `summary.txt` 与结构化 `summary.json`。
- [ ] 报告包含样本数、计划/完成 rollout、错误、性能、长度、截断、Judge coverage、per-source 与 provenance。
- [ ] 提供 basic inspect，能按 sample/rollout 查看 raw trajectory、判分详情与错误。

## 11. Agent 扩展骨架（第一阶段只预留）

- [x] 提供 `AgentExecutor` 接口骨架，不实现复杂 Agent loop。
- [x] 提供 `Environment.reset/step/close` 与 `StepResult` schema。
- [x] 提供 `NullEnvironment`。
- [x] 提供 `AgentHarness.initialize/next_action/finalize` 接口骨架。
- [ ] 预留 tool-use/action/observation event 类型和 Agent metrics registry。
- [ ] 用接口测试证明未来新增 Agent 任务无需改写 `EvaluationPipeline` 主流程。

## 12. 本地自动化测试

- [ ] 单测 Answer extraction 的各提取策略及失败分支。
- [ ] 单测 `455/4 == 113.75`。
- [ ] 单测 `sqrt(6)/2 == sqrt(3/2)`。
- [ ] 单测 pass@k、avg@k、UNKNOWN/失败 denominator 规则。
- [ ] 单测 raw/evaluation serialization round-trip 与 schema version。
- [ ] 单测 resume、重复记录、末行损坏和 incompatible fingerprint。
- [ ] 单测 failed rollout retry 与 retry exhaustion。
- [ ] 单测单 endpoint health check、请求失败和 retry；为后续 multi-endpoint scheduler 保留接口测试。
- [ ] 单测一次性容器 runner 的命令构造、signal cleanup、退出码保持与幂等清理（使用 mock，不依赖 Docker/GPU）。
- [x] 使用 FakeBackend + 3 条 fake dataset 完整验证 Dataset -> Trajectory -> Evaluator -> Metrics -> Report 与 resume。

## 13. 远端交付与验证顺序

- [ ] 在本地完成阶段代码和无 GPU 测试。
- [ ] 将项目以 `scp` 同步到 `n30283:/ssd/eval-harness`；不在远端直接进行不可追踪的手工改码。
- [ ] 在远端先运行 lint/unit/Fake E2E，不启动 GPU 服务。
- [ ] 运行 container lifecycle smoke：启动、ready、正常退出并确认容器删除。
- [ ] 注入启动失败，确认容器仍被删除且退出码正确。
- [ ] 注入 pipeline 失败和 SIGINT/SIGTERM，确认容器仍被删除、结果仍在宿主机。
- [x] 使用真实 SGLang 跑 3 个样本，核对 reasoning/content/usage/finish reason/judge/report。
- [x] 对相同 smoke run 执行 resume，确认已成功 rollout 不重复生成。
- [ ] Review smoke artifacts 后才运行 PCL 500 × n 的正式 baseline。
- [ ] 正式运行后保存 resolved config、dataset hash、image digest、完整日志、raw/evaluated 数据与报告。

## 14. 第一阶段验收标准

- [ ] 单条 `eval-harness run --config ...` 可从零完成 PCL load、推理、raw、math judge、metrics 与报告。
- [ ] crash 后重跑相同命令能安全续跑，不重复生成已成功 rollout。
- [ ] 更换 evaluator/metrics 可复用 raw，不重新调用模型。
- [ ] 单个 TP8 endpoint 在正式请求前完成 health check；第一阶段不验收多 endpoint。
- [ ] Fake E2E、要求的单测和真实 2–5 样本 smoke 全部通过。
- [ ] 任意成功或失败路径结束后，本次任务的一次性容器均不存在。
- [ ] 删除容器不删除 `/ssd/` 下实验结果；raw 始终 append-only。
- [ ] 后续新增 Agent task 仅需新增/组合 Executor、Environment、Harness、Evaluator 与 Metrics，不改写 pipeline 核心。

## 15. 明确暂不实施

- [ ] 不引入 Ray、数据库、Celery、Kafka、微服务、Web UI 或复杂分布式 controller。
- [ ] 不提前实现完整 Shell/Browser/Search/Coding Agent。
- [ ] 不把 Docker/SGLang 生命周期耦合进 trajectory、evaluator 或 metrics core schema。
- [ ] 不在第一阶段启用 LLM judge fallback，除非 Review 后明确纳入。
