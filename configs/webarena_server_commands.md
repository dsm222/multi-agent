# WebArena Verified 服务器执行命令（模板）

## 1) 新建环境
conda create -n webarena-verified python=3.11 -y
conda activate webarena-verified

## 2) 安装
pip install -U pip
pip install git+https://github.com/ServiceNow/webarena-verified.git

## 3) 配置 API
export OPENAI_API_KEY="<your_key>"
export OPENAI_BASE_URL="https://api.linkapi.ai/v1"

## 4) 官方 Demo（示例 task 108）
webarena-verified eval-tasks --task-ids 108 --output-dir /path/to/agent_logs/demo

## 5) 结果回传
将输出目录同步回本地项目 `runs/`。
