# Trajectory Evaluation Harness

通用、组件化、trajectory-centric 的 Model / Agent Evaluation Harness。第一阶段支持 Reasoning / Math，并预留 Agent 扩展接口。

```bash
python -m eval_harness run --config configs/reasoning/pcl_qwen35_9b_base.yaml
```

远端正式运行通过 `scripts/run_ephemeral_sglang.sh` 创建一次性 SGLang TP8 容器；任务结束或异常退出后容器会被强制删除，结果保留在 `/ssd/eval-results`。

已验证的远端 smoke 配置使用 Qwen3.5-9B、TP8、server context 131,072 和最大输出 81,920 tokens。server context 必须大于输出预算，以容纳输入 prompt。

正式配置对 Math rule 的 `UNKNOWN` 和 `INCORRECT` 使用异源 `deepseek-chat` 二次判分。凭据只从 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL` 环境变量读取，不写入配置、日志或轨迹。

远端运行前编辑 `/ssd/eval-harness/.secrets/deepseek.env`：

```dotenv
DEEPSEEK_API_KEY=填入密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

该文件权限应为 `600`，已被 `.gitignore` 排除；runner 会自动加载。

详见 [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) 和 [TODO.md](TODO.md)。
