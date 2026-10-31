# 服务器部署：AML 商业产品榜 · 文本记忆

当前部署保留 Token Plan 编译、qwen3.7-text-embedding-flash 向量和可选决策旁路。
按 [AML 官方 API Guide](https://agentmemoryleaderboard.ai/api-guide) 核对于 2026-10-31：
商业产品榜没有开源方法榜的指定模型约束。本项目提供文本 Add/Search；没有宣称支持多模态请求。

## 从远端拉取并启动

服务器需要 Docker Engine、Docker Compose v2、Git，以及访问 PyPI、GitHub 和模型服务的网络。
以下命令不主动发送评测或模型请求；启动后的平台 Add/Search 会使用模型额度。

```bash
git clone --branch feat/aml-20261013 --single-branch https://github.com/Jasonatafricanow/MR-Mem.git
cd MR-Mem
cp .env.example .env
chmod 600 .env
# 编辑 .env，填写 QWEN_PLAN_API_KEY、DASHSCOPE_API_KEY、MRMEM_AML_TOKEN
# 可用 python3 -c 'import secrets; print(secrets.token_urlsafe(32))' 生成服务访问令牌
docker compose build
docker compose run --rm --no-deps --entrypoint python mrmem scripts/smoke_offline.py
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:8765/health
```

`.env` 由 Compose 注入，不进镜像、不进 Git。不要把 AML 发给你的 Eval Key 当成
`MRMEM_AML_TOKEN`；后者是你生成并提供给 AML 的 Memory System Key。

镜像使用 Python 3.12、固定 LCE 提交和 `deploy/constraints.txt` 中的运行依赖版本。
单进程、单容器拥有自己的 SQLite/Qdrant 数据目录；不要让多个副本同时写同一数据卷。
服务使用非 root 用户，根文件系统只读，数据库和向量缓存都位于持久卷 `/var/lib/mrmem`。
默认不需要本地 GPU。内存和磁盘容量取决于正式题库规模，本项目未完成 Full 容量认证。

## 公网接入

Compose 默认只发布到服务器回环地址。建议在宿主使用 Nginx/TLS 反向代理：
参考 `deploy/nginx.conf.example`，替换域名并安装证书，再开放防火墙相应端口。
示例将代理读写超时设置为 1900 秒，以容纳官方最多 30 分钟的单次 Add/Search。
若自行采用其他入口，避免其默认短超时中断长请求。

需要直接暴露端口时，可在 `.env` 设置 `MRMEM_BIND_ADDRESS=0.0.0.0`，但 TLS 和防火墙仍由服务器负责。
不要向 AML 提交 `localhost`、私有地址或带密码的 URL。

提交给平台的接口与参数：

| 项 | 值 |
|---|---|
| 榜单 / 类型 | 商业产品榜 / 文本记忆 |
| Add URL | `https://你的域名/add` |
| Search URL | `https://你的域名/search` |
| Health URL | `https://你的域名/health`，无需鉴权 |
| 鉴权 | `Bearer`；Memory System Key 填你的 `MRMEM_AML_TOKEN` |
| 同步模式 | Add 完成持久化和可检索后返回 200；不使用 202 / Add-status |
| Search 上限 | 返回不超过请求 `top_k`；正式平台为 100 |

Token 和 X-Api-Key 也受支持，但提交时选择一种并保持一致。Add 成功原样回显
request_id、user_id、session_id；Search 只按 user_id 隔离，返回带 id/content 的 data 数组。
重复 Add 使用同一 request_id 和相同请求体；不要重建 ID 来重试。
平台会有界重试暂时失败，重复 ID 配不同内容会返回 422。

## 配置与运维

- 服务配置是 `deploy/runtime.server.json`。模型和提示词沿用已确定的版本。
- `MRMEM_COMPILE_CONCURRENCY` 默认 8。它是内部模型调用并发，不是 Full 吞吐保证。
  平台 Add/Search 并发需按服务器及供应商额度设定，先通过官方兼容 smoke 再启动 Full。
- 决策旁路默认总等待上限 15 秒，超时返回基础检索；`/health.decision` 可看到降级计数。
- 健康检查不调用模型。`mr-mem-aml check` 会调用编译与向量服务，
  `--check-decision` 还会探测决策服务，均可能计费，不会在启动或 CI 自动执行。
- 看日志：`docker compose logs --tail=100 mrmem`。日志可能包含请求错误，只保存在受控服务器。
- 更新：在没有评测任务时 `git pull --ff-only`，重新 build/up。正式 Full 期间固定提交，
  不自动拉取、不切模型、不调整提示词。用 `git rev-parse HEAD` 记录参评版本。
- 停机前先停止平台任务并等待请求结束。重启保留命名卷；**不要运行 `docker compose down -v`**。
- 备份需要先停服务，再备份整个数据卷；不要只复制正在写入的单个 SQLite 文件。

## 不用 Docker

Python 3.12 环境中可执行 `pip install -r requirements-server.txt`，然后复制
`deploy/runtime.server.json` 为被忽略的 `runtime.json`，将 data_dir/cache_path 改为服务器
可写的持久目录，设置同样的环境变量，再运行 `mr-mem-aml serve --config runtime.json`。
由你的进程管理器负责重启、日志和开机启动。

离线 smoke 验证的是合成输入的接口、重试、用户隔离和重启读取，不验证供应商额度或正式榜单成绩。
实际公网和账号接通后，仍需在平台运行官方兼容 smoke；仓库不会自动提交付费评测。
