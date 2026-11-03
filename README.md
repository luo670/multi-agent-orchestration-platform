# Multi-Agent Orchestration Platform

> 企业级多 Agent 协作自动化平台 —— 面向技术媒体编辑部的 AI 内容生产流水线

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1c3c3c.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

用户输入一个技术主题，系统自动调度 **Researcher → Writer → Reviewer** 三个 Agent 协同工作，最终输出一篇高质量技术文章。系统具备高并发、异步处理、状态持久化和链路评估能力。

---

## ✨ 核心特性

- **多 Agent 图编排** — 基于 LangGraph 的有状态条件循环（审核未通过自动重写）
- **异步任务调度** — FastAPI 提交 + Celery Worker 执行，接口毫秒级响应
- **实时 SSE 推送** — 基于 Redis Pub/Sub 桥接，前端实时收到节点流转状态
- **链路可观测** — 每次 LLM 调用自动落库（耗时 / Token / 审核结果）
- **状态可恢复** — LangGraph Checkpointer + Celery acks_late + Tenacity 重试
- **零硬编码密钥** — 全链路环境变量加载，`.gitignore` 双重保险
- **一键容器化** — Docker Compose 拉起 5 个服务

---

## 🏗 系统架构

```
┌──────────────┐   POST /tasks      ┌──────────────┐
│   Client     │ ─────────────────> │   FastAPI    │
│  (Browser)   │ <── {task_id} ──── │   (API)      │
│              │ GET /tasks/{id}/stream    │ delay()
│              │ ─────────────────> ┌──────▼───────┐
│              │ <── SSE stream ─── │  Redis PubSub│
└──────────────┘                    └──────┬───────┘
                           ┌───────────────┴───────────────┐
                           ▼                               ▼
                   ┌──────────────┐                ┌──────────────┐
                   │    Redis     │ <── publish ── │   Celery     │
                   │  (Pub/Sub)   │                │   Worker     │
                   └──────────────┘                └──────┬───────┘
                                                          │
                                                  ┌───────▼────────┐
                                                  │   LangGraph    │
                                                  │ Researcher →   │
                                                  │  Writer →      │
                                                  │  Reviewer ─┐   │
                                                  │     ↑      │   │
                                                  │  Writer ←──┘   │
                                                  └───────┬────────┘
                                                          ▼
                                                  ┌────────────────┐
                                                  │     MySQL      │
                                                  │  tasks 表      │
                                                  │  agent_metrics  │
                                                  └────────────────┘
```

### Agent 协作流程

```
START → Researcher → Writer → Reviewer ─┬─ passed=True ─────────────→ END
                                         ├─ revision_count >= 3 ────→ END（强制收尾）
                                         └─ rejected（有重写额度）──→ Writer（循环重写）
```

| Agent | 职责 | 温度 |
|-------|------|------|
| **Researcher** | 6 段式调研提纲（背景/现状/玩家/原理/影响/参考） | 0.3 |
| **Writer** | 基于调研笔记生成/改写 1200-2000 字文章 | 0.7 |
| **Reviewer** | 5 项标准审核，输出结构化 JSON `{passed, feedback}` | 0.1 |

---

## 🛠 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| Web 框架 | FastAPI | 全异步 HTTP 网关 + SSE |
| Agent 编排 | LangGraph | 有状态多 Agent 图编排 |
| 大模型 | DashScope (Qwen) | OpenAI 兼容接口 |
| 数据库 | MySQL / SQLite | SQLAlchemy ORM |
| 任务队列 | Celery + Redis | 异步执行 pipeline |
| 实时通信 | SSE + Redis Pub/Sub | 节点流转状态推送 |
| 韧性 | Tenacity | 指数退避重试 |
| 容器化 | Docker Compose | 一键部署 |

---

## 📦 项目结构

```
.
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置中心（pydantic-settings，零硬编码）
│   ├── database.py        # 异步引擎 + 会话工厂
│   ├── models.py        # ORM：tasks + agent_metrics
│   ├── events.py        # Redis Pub/Sub 发布/订阅
│   ├── graph.py          # LangGraph 核心（State + 3 Agent + 路由）
│   ├── celery_app.py     # Celery 实例配置
│   ├── tasks.py          # Celery 任务体
│   ├── main.py           # FastAPI 应用 + SSE 端点
│   └── utils.py          # 评估拦截器 + 重试 + 限流锁
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
├── .dockerignore
├── 技术文档.docx
└── README.md
```

---

## 🚀 快速开始

### 前置条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (依赖管理)
- Redis (本地或 Docker)
- (可选) MySQL 8.0，本地开发可用 SQLite

### 1. 克隆并安装

```bash
git clone https://github.com/your-username/multi-agent-orchestration-platform.git
cd multi-agent-orchestration-platform

# 安装依赖
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 启动服务

**方式 A：Docker Compose（推荐，一键拉起全部）**

```bash
docker compose up --build
```

**方式 B：本地开发（分进程启动）**

```bash
# 启动 Redis
docker run -d -p 6379:6379 redis:7-alpine

# 终端 1：FastAPI
uv run uvicorn app.main:app --reload

# 终端 2：Celery Worker
uv run celery -A app.celery_app worker --loglevel=info
```

### 4. 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 提交一个主题
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"topic": "LangChain 的 Agent 架构演进"}'

# 用浏览器或 curl 监听 SSE
curl -N http://localhost:8000/api/v1/tasks/{task_id}/stream
```

- Swagger UI: http://localhost:8000/docs
- Flower 监控: http://localhost:5555

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tasks` | 提交主题，返回 task_id |
| GET | `/api/v1/tasks/{task_id}/stream` | SSE 实时推送执行状态 |
| GET | `/api/v1/tasks/{task_id}` | 查询最终结果 |
| GET | `/health` | 健康检查 |

### SSE 事件类型

| 事件 | 触发时机 | data 示例 |
|------|---------|----------|
| `node_start` | 节点开始执行 | `{"node": "researcher", "topic": "..."}` |
| `node_end` | 节点执行完成 | `{"node": "researcher", "chars": 3200}` |
| `review_result` | Reviewer 完成审核 | `{"passed": false, "feedback": "..."}` |
| `task_done` | 任务成功完成 | `{"chars": 2100, "revisions": 2}` |
| `task_failed` | 任务失败 | `{"error": "..."}` |

---

## 🔒 安全说明

- **API Key 零硬编码**：所有凭证通过 `.env` + 环境变量加载，`.env` 已被 `.gitignore` 排除
- 仓库只提供 `.env.example` 模板（占位符，无真实值）
- Docker 镜像不 bake 任何凭证，通过 `env_file` 注入

---

## 📊 可观测性

每次 LLM 调用自动写入 `agent_metrics` 表：

| 字段 | 说明 |
|------|------|
| node_name | researcher / writer / reviewer |
| duration_ms | 节点执行耗时 |
| prompt_tokens | 输入 Token |
| completion_tokens | 输出 Token |
| passed | 仅 Reviewer：True=通过 |
| feedback | Reviewer 反馈 |

---

## 🐳 Docker Compose 服务

| 服务 | 端口 | 作用 |
|------|------|------|
| redis | 6379 | Broker + Pub/Sub + 限流锁 |
| mysql | 3306 | 生产数据库 |
| api | 8000 | FastAPI 网关 |
| celery | - | Worker 执行 pipeline |
| flower | 5555 | Celery 监控（可选） |

水平扩展 Worker：`docker compose up --scale celery=4`

---

## 📋 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | ✅ | 通义千问 API Key |
| `DASHSCOPE_BASE_URL` | - | OpenAI 兼容端点 |
| `QWEN_MODEL` | - | 模型名，默认 qwen-plus |
| `DATABASE_URL` | - | 数据库连接，默认 SQLite |
| `REDIS_URL` | - | Redis 连接 |
| `CELERY_BROKER_URL` | - | Celery broker |
| `MAX_REVISIONS` | - | 最大重写次数，默认 3 |

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交变更：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

---

## 📄 License

[MIT](LICENSE)