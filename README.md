# 多智能体 ReAct 协作推理（硕士论文实验）

本项目聚焦 LLM 多智能体协作推理系统工程，不做底层大模型预训练，核心贡献定位在编排、通信、验证与调试四层。系统以动态通信拓扑实现回合级信息路由，以 vote-first 共识结合 verifier/inspector 提升可靠性，并通过可重放轨迹日志支撑失败归因与干预复现实验，在 MARBLE 主线与 WebArena Verified 外推任务上验证方法有效性与成本可控性。

## 题目
融合共识机制与动态角色分配的多智能体 ReAct 协作推理系统研究

## 方法主线
- 动态拓扑路由（Dynamic Topology Routing）
- vote-first 共识 + verifier/inspector
- trace log + replay + failure analysis

## Benchmark 方案
- 主线：MARBLE / MultiAgentBench（Research + Coding）
- 外推：WebArena Verified（Hard subset，服务器执行）

## 当前进度
- [x] 阶段0：题目边界冻结（`project_scope.md`）
- [x] 阶段1：独立环境与最小闭环（LangGraph + LinkAPI + MARBLE smoke）
- [x] 阶段2：仓库初始化与目录骨架
- [x] 阶段3：统一状态/黑板/轨迹 schema（`core/state.py`、`core/blackboard.py`、`core/trace.py`）
- [x] 阶段4：single-agent baseline（`graphs/baseline_single.py` + 20样本运行日志）

## 关键路径
- 主环境：`conda env: mas-react (Python 3.10)`
- MARBLE环境：`conda env: mas-react-marble (Python 3.10 + poetry)`
- 模型接口：`https://api.linkapi.ai/v1`
- 模型名称：`qwen3.5-35b-a3b`

## WebArena 执行策略
- 本地不安装 Docker。
- WebArena Verified 在服务器侧执行并回传日志到 `runs/`。
- 服务器命令模板见 `configs/webarena_server_commands.md`。
