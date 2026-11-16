# PetitCoding

本地 Qwen3.8-27B 推理服务和 coding agent 客户端配置。默认使用 SGLang 的 Anthropic 兼容端点连接 Claude Code，也提供 OpenCode 配置和 vLLM 启动配置。模型权重与 KV cache 使用 BF16，最多使用 4 张 GPU。

## 环境

- Python 3.11+，4 张可用 H100（可在配置中改为 1 或 2 张），本地模型目录。
- 安装与 Qwen3.8 兼容的 SGLang 或 vLLM，以及 Transformers >= 5.8。依赖与 CUDA、驱动版本相关，请参考 [SGLang 模型指南](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) 和 [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)。下面使用独立环境，不影响系统环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'sglang==0.5.23' 'transformers==5.19.0' 'torch==2.14.1'
# 或另建环境安装 vllm==0.33.0（见 vLLM 官方安装指南）
```

## 启动

```bash
python -m petitcoding show --config configs/sglang.toml
python -m petitcoding serve --config configs/sglang.toml
```

`serve` 前台运行，Ctrl-C 停止；选 vLLM 使用 `configs/vllm.toml`。编辑 TOML 中的模型目录、GPU、上下文和端口。`show` 只打印启动命令；扩展上下文所需环境变量在 `serve` 内自动设置。后端默认仅监听 localhost，启动完成后可访问 `http://127.0.0.1:30000/v1/models`。若端口或模型目录不匹配，请先调整配置。使用过程中勿将服务暴露到公网；默认没有认证。

上下文默认 262144（输入与输出合计）；可以将 `context_length` 改为 524288 或 1000000，自动注入模型说明书提供的静态 YaRN 配置。扩展上下文的实际容量和质量依赖硬件与引擎版本，静态 YaRN 也可能影响短文本质量。显存不足时请降低并发或上下文，工具不会自动使用低精度。

## VS Code Claude Code

1. 安装官方 Claude Code 扩展。运行 `python -m petitcoding client claude --config configs/sglang.toml`。
2. 将输出的 JSON 属性合并到 VS Code 的 `settings.json` 中；保留已有设置，不要覆盖整个文件。设置 `"claudeCode.disableLoginPrompt": true`，重新加载窗口并开启新会话。
3. VS Code 与服务不在同一台机器时，通过 SSH 隧道把服务端口转发到运行扩展的机器；Remote SSH 模式需确认扩展进程实际运行的位置。确保生成配置中的地址对扩展进程可达。

这里使用所选后端的原生 `/v1/messages`，无需自行实现协议转换；vLLM 接入仍需按所用版本验证。Anthropic 不支持经网关使用非 Claude 模型，此集成的完整工作流仍需在实际扩展中验证；插件更新可能影响兼容性。环境变量详细说明参见 [SGLang 文档](https://docs.sglang.io/docs/basic_usage/anthropic_api) 和 [VS Code 扩展文档](https://code.claude.com/docs/en/vs-code)。示例 token 是本机无鉴权服务的占位值，不是密码。

运行 `python -m petitcoding client opencode` 可生成 OpenCode JSON 配置；`client codex` 输出实验性的 Responses 配置，需自行确认所用后端和客户端版本的 Responses 兼容性。Claude 和 Codex 集成不因模型 API 可访问而自动视为工作流已验证。

## 并发 benchmark

```bash
python -m petitcoding.benchmark --concurrency 1 2 4 8 --requests 16 --output-tokens 256 --output benchmark-results
# 自定义同一输入文本（不同请求可共享前缀缓存）：
python -m petitcoding.benchmark --prompt-file prompt.txt --requests 16 --concurrency 1 2 4
```

输出逐请求 `requests.jsonl` 和 `summary.json`，包含流式首 token 时间（首个非空输出事件）、完整延迟、请求吞吐及服务端返回 usage 时的输出 token 吞吐。各档固定请求数，闭环发压；延迟从线程开始发出请求计时，不包括客户端线程池内等待，失败也记录且返回非零状态；`--warmup 0` 禁用预热。数据是同一 prompt 的热缓存场景；如需冷缓存、准确 token 长度、GPU 指标、TPOT、开环和真实 coding 工作流评估，需额外工具，不能直接从当前结果推断。为防止覆盖，以不同 `--output` 目录保存每次运行。长上下文请求可能非常耗时，输入需预留输出空间。

## 验证

```bash
python -m unittest discover -s tests -v
```

项目计划与验收范围见 [计划文档](docs/plans/v0.0.1.md)。

本机测试记录与当前未验证项见 [2026-11-16 冒烟报告](docs/reports/2026-11-16-smoke.md)。
