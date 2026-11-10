# Agent 评测运行指南（ALFWorld / WebShop）

本文记录 eval-harness 的两套交互式 Agent 评测（ALFWorld、WebShop）如何在服务器 `n30285` 上运行，包括**在哪个容器**、**用哪个配置**、**跑什么命令**。

> 所有评测都在 `n30285` 上；代码已同步到 `n30283:/ssd/eval-harness` 和 `n30285:/ssd/eval-harness`（两端 git tree 一致）。

---

## 0. 公共底座：确认镜像

两套评测都基于同一份已验证的镜像：

| 镜像 | 内容 |
|---|---|
| `alfworld-eval-confirmed:20261109` | `lmsysorg/sglang:0.5.19-qwen38-runtime` + textworld 1.7.0 + fast-downward-textworld 20.6.4 + nltk + networkx + termcolor |

公共挂载（两个容器相同）：

| 宿主机路径 | 容器路径 | 用途 |
|---|---|---|
| `/ssd/eval-harness` | `/eval_harness`（ro） | eval-harness 代码 |
| `/ssd/slime-ckpt/EDGE/agent_system/environments/env_package` | `/env_package`（ro） | vendored alfworld/webshop 环境包 |
| `/ssd/models/<model>` | `/workspace/model`（ro） | 模型权重 |
| `/ssd/slime-ckpt/agent-cpu-smoke` | `/work`（rw） | 数据集 jsonl + 结果 + 脚本 |

模型：

| 模型 | 路径 | 用途 |
|---|---|---|
| base | `/ssd/models/Qwen2.5-1.5B-Instruct` | 基座模型 |
| step100 | `/ssd/models/Qwen2.5-1.5B-Instruct-step100` | ALFWorld 训练 100 步后（由 EDGE checkpoint 合并） |

---

## 1. ALFWorld 评测

### 1.1 容器

- 名称：`alfworld-eval`
- 镜像：`alfworld-eval-confirmed:20261109`
- GPU：`--gpus all`，SGLang 用 **GPU 0**（`--tp-size 1`）
- 额外挂载：`/ssd/datasets/alfworld` → `/alfworld_data`（ro）

创建命令（参考）：

```bash
docker run -d --name alfworld-eval --gpus all \
  -v /ssd/eval-harness:/eval_harness:ro \
  -v /ssd/slime-ckpt/EDGE/agent_system/environments/env_package:/env_package:ro \
  -v /ssd/datasets/alfworld:/alfworld_data:ro \
  -v /ssd/models/Qwen2.5-1.5B-Instruct:/workspace/model:ro \
  -v /ssd/slime-ckpt/agent-cpu-smoke:/work \
  --entrypoint sleep alfworld-eval-confirmed:20261109 infinity
```

### 1.2 数据集 jsonl

在宿主机 `/ssd/slime-ckpt/agent-cpu-smoke/`（容器内 `/work/`）：

| 文件 | 内容 |
|---|---|
| `alfworld_games.jsonl` | 2 个 valid_seen 任务（smoke） |
| `alfworld_games_128.jsonl` | 128 个 valid_seen 任务（由 `alfworld_fixed_128.json` manifest 生成） |

每条格式：`{"game_file": "/alfworld_data/json_2.1.1/valid_seen/.../game.tw-pddl"}`

### 1.3 配置

| 配置 | 说明 |
|---|---|
| `configs/agent/alfworld.yaml` | ?? 128 ? valid_seen ???base / checkpoint ???????? |

关键项：`environment.type=alfworld`、`harness.type=alfworld`、`executor.type=agent`、`evaluators=[agent_success]`、`concurrency=16`、`max_steps=50`。

### 1.4 运行步骤

1. 起 SGLang（容器内 `/work/start_sglang.sh`，`--served-model-name` 对应配置里的 `backend.model`）：

```bash
docker exec -d alfworld-eval bash /work/start_sglang.sh
```

2. 等就绪后跑评测：

```bash
docker exec -e ALFWORLD_DATA=/alfworld_data \
  -e PYTHONPATH=/eval_harness -w /eval_harness \
  alfworld-eval python -m eval_harness run \
  --config configs/agent/alfworld.yaml
```

### 1.5 结果（对齐 EDGE）

| 模型 | eval-harness（valid_seen） | EDGE（train） |
|---|---|---|
| base | 2.34%（3/128） | 7.03%（9/128） |
| step100 | **74.2%（95/128）** | 75.8% |

> 差距主要来自任务集（valid_seen vs train）；prompt 已对齐 EDGE（system prompt + 完整模板 + `<think>/<action>`）。

---

## 2. WebShop 评测

### 2.1 容器

- 名称：`webshop-eval`
- 镜像：`alfworld-eval-confirmed:20261109` + **额外依赖**（见 2.2）
- GPU：`--gpus all`，SGLang 用 **GPU 1**（`--base-gpu-id 1`，避免和 alfworld 的 GPU 0 冲突）
- 额外挂载：`/ssd/datasets/webshop` → `/webshop_data`（ro）

### 2.2 额外依赖（已装进容器）

- Java 21（`default-jdk-headless`，pyserini 需要 javac）
- pyserini 2.4.0
- spacy 模型 `en_core_web_sm`（`pip install <wheel>`，reward 词性标注用）
- WebShop Python 依赖：flask / gym / pandas / rank_bm25 / thefuzz / beautifulsoup4 / selenium / cleantext / scikit-learn

### 2.3 容器内 WebShop 环境（一次性搭建，已做）

因为 vendored webshop 是只读挂载，需复制到可写目录 + 软链数据 + 建 Lucene 索引：

```bash
docker exec webshop-eval cp -r /env_package/webshop/webshop /work/webshop
docker exec webshop-eval ln -sfn /webshop_data/data /work/webshop/data
# 建索引（convert 需先 mkdir resources_*）
docker exec -w /work/webshop/search_engine webshop-eval mkdir -p resources_100 resources resources_1k resources_100k
docker exec -w /work/webshop/search_engine webshop-eval python convert_product_file_format.py
docker exec -w /work/webshop/search_engine webshop-eval python -m pyserini.index.lucene \
  --collection JsonCollection --input resources_1k --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

> 注意：pyserini 导入时会尝试下载 GitHub 预建索引信息（国内超时），已 patch `/opt/sglang/lib/python3.12/site-packages/pyserini/prebuilt_index_info.py` 跳过网络下载（`records = []` + `.get(key, {})`）。

### 2.4 数据集 jsonl

宿主机 `/ssd/slime-ckpt/agent-cpu-smoke/`（容器内 `/work/`）：

| 文件 | 内容 |
|---|---|
| `webshop_sessions.jsonl` | 5 个 session（商品索引 0-4，smoke） |

每条格式：`{"session": 0}`（session = 商品索引，500 测试商品为 0..499）

### 2.5 配置

| 配置 | 说明 |
|---|---|
| `configs/agent/webshop.yaml` | 5 任务 smoke，base 模型 |

关键项：`environment.type=webshop`（`webshop_package_path=/work/webshop`、`num_products=1000`）、`harness.type=webshop`、`executor.type=agent`、`evaluators=[agent_success]`、`concurrency=1`、`max_steps=15`。

### 2.6 运行步骤

1. 起 SGLang（GPU 1）：

```bash
docker exec -d webshop-eval bash /work/start_sglang.sh   # 内部 --base-gpu-id 1
```

2. 等就绪后跑评测：

```bash
docker exec -e PYTHONPATH=/eval_harness -w /eval_harness \
  webshop-eval python -m eval_harness run \
  --config configs/agent/webshop.yaml
```

### 2.7 结果

- base 模型：0%（0/5）—— 1.5B 模型第 2 步起切换中文（`继续搜索[...]`），动作非法。
- 框架正确：第一条 `search[...]` 合法并正确返回 50 个商品；解析/判非法/reward/持久化正常。0% 是模型对齐问题，不是框架 bug。

---

## 3. 动作 / 评分语义

| 维度 | ALFWorld | WebShop |
|---|---|---|
| 动作格式 | `<think>...</think><action>go to X</action>` | `<think>...</think><action>search[kw] / click[value]</action>` |
| 动作解析 | 提取 `<action>` + lowercase + `<think>`/CJK 校验 | 同左 + `search[...]`/`click[...]` 正则 |
| success | env `won`（`reward==1.0`） | env `reward==1.0`（买到正确商品） |
| max_steps | 50 | 15 |
| 并发 | 16（contextvars 每样本独立 env） | 1（单 env，session 切换） |

---

## 4. 结果目录

评测结果写到宿主机 `/ssd/slime-ckpt/agent-cpu-smoke/results/<output_dir>/`：

```
raw.jsonl        完整轨迹（每轮 observation/action/reasoning）
evaluated.jsonl  评分（success/episode_length/invalid_action_count）
summary.json     汇总指标（accuracy/counts/judge_coverage）
summary.txt      文本报告
config.yaml      resolved 配置
metadata.json    模型/指纹/数据集 hash
state.json       阶段状态
```

日志：SGLang → `/work/sglang.log`；评测 stdout → `/work/alfworld_128_*.log` / `/work/webshop_*.log`。
