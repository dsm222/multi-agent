from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

import yaml
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.blackboard import BlackboardStore
from core.state import AgentState
from core.trace import TraceEvent, TraceRecorder

DEFAULT_RUNTIME_CONFIG_PATH = ROOT / "configs" / "model_runtime.yaml"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimeConfig:
    base_url: str
    model: str
    api_key_env: str


class SingleAgentGraphState(TypedDict, total=False):
    task_id: str
    task_text: str
    benchmark_name: str
    thought: str
    need_tool: bool
    tool_name: str
    tool_input: str
    tool_output: str
    observation: str
    final_answer: str
    token_usage: Dict[str, int]
    tool_calls: List[Dict[str, Any]]
    trace_events: List[Dict[str, Any]]
    error: str


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RuntimeConfig(
        base_url=data["base_url"],
        model=data["model"],
        api_key_env=data["api_key_env"],
    )


def merge_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    result = dict(a)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        result[k] = int(result.get(k, 0)) + int(b.get(k, 0))
    return result


def call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    retries: int = 5,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            content = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }
            return {"content": content, "usage": usage, "latency_ms": latency_ms}
        except Exception as err:  # pragma: no cover
            last_err = err
            if attempt >= retries:
                break
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"model call failed after {retries} retries: {last_err}")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        maybe = json.loads(text)
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        maybe = json.loads(m.group(0))
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        return None
    return None


def extract_math_expression(task_text: str) -> Optional[str]:
    m = re.search(r"([0-9\.\s\+\-\*\/\(\)%]{3,})", task_text)
    if not m:
        return None
    expr = m.group(1).strip()
    if any(op in expr for op in "+-*/%"):
        return expr
    return None


ALLOWED_BIN_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
)
ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)


def eval_math_expr(expr: str) -> float:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ALLOWED_BIN_OPS):
            l = _eval(n.left)
            r = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r
            if isinstance(n.op, ast.Mod):
                return l % r
            if isinstance(n.op, ast.Pow):
                return l**r
            if isinstance(n.op, ast.FloorDiv):
                return l // r
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ALLOWED_UNARY_OPS):
            v = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +v
            if isinstance(n.op, ast.USub):
                return -v
        raise ValueError("unsupported expression")

    return _eval(node)


def append_event(
    state: SingleAgentGraphState,
    *,
    node_name: str,
    input_summary: str,
    output_summary: str,
    latency_ms: int,
    tool_name: Optional[str] = None,
    cost: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    prev = list(state.get("trace_events", []))
    prev.append(
        {
            "event_id": str(uuid4()),
            "timestamp": utc_now_iso(),
            "round_id": 1,
            "node_name": node_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "tool_name": tool_name,
            "cost": cost,
            "latency_ms": latency_ms,
            "metadata": metadata or {},
        }
    )
    return prev


def build_graph(client: OpenAI, model: str):
    def think_node(state: SingleAgentGraphState) -> Dict[str, Any]:
        prompt = (
            "你是单智能体ReAct基线的think步骤。\n"
            "请只输出JSON对象，包含字段：\n"
            '{"thought": "...", "need_tool": true/false, "tool_name": "calculator|none", "tool_input": "..."}\n'
            "规则：如果任务包含明确算式，need_tool必须为true且tool_name=calculator。\n"
            f"任务：{state['task_text']}"
        )
        rsp = call_model(
            client,
            model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
            temperature=0.0,
        )
        parsed = extract_json_object(rsp["content"]) or {}
        thought = str(parsed.get("thought", rsp["content"][:400]))
        need_tool = bool(parsed.get("need_tool", False))
        tool_name = str(parsed.get("tool_name", "none"))
        tool_input = str(parsed.get("tool_input", "")).strip()

        expr = extract_math_expression(state["task_text"])
        if expr and (tool_name in ("none", "", "null") or not need_tool):
            need_tool = True
            tool_name = "calculator"
            tool_input = expr

        return {
            "thought": thought,
            "need_tool": need_tool,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "token_usage": merge_usage(state.get("token_usage", {}), rsp["usage"]),
            "trace_events": append_event(
                state,
                node_name="think",
                input_summary=state["task_text"][:160],
                output_summary=f"need_tool={need_tool}, tool_name={tool_name}",
                latency_ms=rsp["latency_ms"],
            ),
        }

    def act_tool_node(state: SingleAgentGraphState) -> Dict[str, Any]:
        start = time.perf_counter()
        tool_name = state.get("tool_name", "none")
        tool_input = state.get("tool_input", "")
        output = "NO_TOOL"
        err = ""
        if state.get("need_tool", False):
            if tool_name == "calculator":
                try:
                    output = str(eval_math_expr(tool_input))
                except Exception as e:
                    err = f"calculator_error: {e}"
                    output = err
            else:
                output = f"unsupported_tool: {tool_name}"
        latency_ms = int((time.perf_counter() - start) * 1000)
        call_item = {
            "timestamp": utc_now_iso(),
            "round_id": 1,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": output,
            "success": err == "",
        }
        return {
            "tool_output": output,
            "error": err,
            "tool_calls": list(state.get("tool_calls", [])) + [call_item],
            "trace_events": append_event(
                state,
                node_name="tool",
                input_summary=f"{tool_name}({tool_input})",
                output_summary=output[:160],
                latency_ms=latency_ms,
                tool_name=tool_name,
            ),
        }

    def observe_node(state: SingleAgentGraphState) -> Dict[str, Any]:
        start = time.perf_counter()
        if state.get("need_tool", False):
            observation = f"tool observation: {state.get('tool_output', '')}"
        else:
            observation = "no external tool was used"
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "observation": observation,
            "trace_events": append_event(
                state,
                node_name="observe",
                input_summary=state.get("thought", "")[:160],
                output_summary=observation[:160],
                latency_ms=latency_ms,
            ),
        }

    def answer_node(state: SingleAgentGraphState) -> Dict[str, Any]:
        prompt = (
            "你是单智能体ReAct基线的answer步骤。\n"
            "基于任务、thought和observation，给出最终答案。\n"
            "输出要求：简洁、直接，不要输出过程JSON。\n\n"
            f"任务: {state['task_text']}\n"
            f"thought: {state.get('thought', '')}\n"
            f"observation: {state.get('observation', '')}\n"
        )
        rsp = call_model(
            client,
            model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=420,
            temperature=0.0,
        )
        answer = rsp["content"].strip()
        return {
            "final_answer": answer,
            "token_usage": merge_usage(state.get("token_usage", {}), rsp["usage"]),
            "trace_events": append_event(
                state,
                node_name="answer",
                input_summary=state.get("observation", "")[:160],
                output_summary=answer[:160],
                latency_ms=rsp["latency_ms"],
            ),
        }

    def route_after_think(state: SingleAgentGraphState) -> str:
        return "tool" if state.get("need_tool", False) else "observe"

    graph = StateGraph(SingleAgentGraphState)
    graph.add_node("think", think_node)
    graph.add_node("tool", act_tool_node)
    graph.add_node("observe", observe_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", route_after_think, {"tool": "tool", "observe": "observe"})
    graph.add_edge("tool", "observe")
    graph.add_edge("observe", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def default_tasks() -> List[str]:
    return [
        "计算 27*43 + 19 的结果。",
        "计算 (128-37) / 7 的结果。",
        "用一句话解释什么是过拟合。",
        "把“多智能体协作推理”翻译成英文。",
        "计算 12.5*8 - 6.4 的结果。",
        "简要说明监督学习和无监督学习的区别。",
        "给出一个提高论文实验可复现性的建议。",
        "计算 (99+101)*(37-12) 的结果。",
        "一句话解释为什么需要验证器(verifier)。",
        "计算 2**8 + 3**4 的结果。",
        "给出一个适合记录实验配置的字段列表（简短）。",
        "计算 (3.6+7.4)*5 的结果。",
        "解释什么是ReAct框架（不超过两句话）。",
        "计算 1000/25 + 17 的结果。",
        "简述为什么需要动态通信拓扑。",
        "计算 (14%5) + (23%7) 的结果。",
        "请给出一个单智能体baseline的定义。",
        "计算 ((18-3)*4)/6 的结果。",
        "给出一个失败分析标签的示例。",
        "计算 (45+55+65)/5 的结果。",
    ]


def run_single_task(
    app: Any,
    task_id: str,
    task_text: str,
    benchmark_name: str,
    trace_output_path: Path,
) -> Dict[str, Any]:
    start = time.perf_counter()
    result: SingleAgentGraphState = app.invoke(
        {
            "task_id": task_id,
            "task_text": task_text,
            "benchmark_name": benchmark_name,
            "token_usage": {},
            "tool_calls": [],
            "trace_events": [],
        }
    )
    wall_ms = int((time.perf_counter() - start) * 1000)

    agent_state = AgentState.bootstrap(task_id=task_id, task_text=task_text, benchmark_name=benchmark_name)
    agent_state.set_round(1, "single_agent_react", ["single_agent"])
    agent_state.append_message("single_agent", result.get("thought", ""))

    bb = BlackboardStore()
    subgoal = bb.add_item(
        item_type="subgoal",
        content=task_text,
        source_agent="single_agent",
        round_id=1,
        confidence=1.0,
        status="verified",
    )
    bb.add_item(
        item_type="result",
        content=result.get("final_answer", ""),
        source_agent="single_agent",
        round_id=1,
        confidence=1.0 if result.get("final_answer") else 0.0,
        status="verified" if result.get("final_answer") else "rejected",
        parent_id=subgoal.item_id,
    )
    agent_state.set_blackboard_records(bb.as_records())

    if result.get("thought"):
        agent_state.add_claim(
            {
                "claim_id": str(uuid4()),
                "round_id": 1,
                "agent_id": "single_agent",
                "text": result["thought"],
                "confidence": 0.8,
            }
        )
    if result.get("observation"):
        agent_state.add_evidence(
            {
                "evidence_id": str(uuid4()),
                "round_id": 1,
                "agent_id": "single_agent",
                "text": result["observation"],
                "confidence": 0.9,
            }
        )

    final_answer = result.get("final_answer", "")
    agent_state.add_candidate_answer(
        {
            "agent_id": "single_agent",
            "answer": final_answer,
            "confidence": 1.0 if final_answer else 0.0,
            "round_id": 1,
        }
    )
    agent_state.set_vote_result(
        {
            "method": "single_agent_direct",
            "winner": final_answer,
            "votes": {final_answer: 1} if final_answer else {},
            "round_id": 1,
        }
    )
    agent_state.set_verifier_result(
        {
            "status": "not_applied",
            "conflicts": [],
            "notes": "single-agent baseline skips verifier stage",
            "round_id": 1,
        }
    )
    agent_state.token_usage = dict(result.get("token_usage", {}))
    agent_state.tool_calls = list(result.get("tool_calls", []))
    agent_state.set_final_answer(final_answer, "single_agent_completed" if final_answer else "single_agent_failed")
    agent_state.set_trace_events(list(result.get("trace_events", [])))

    recorder = TraceRecorder()
    for event in result.get("trace_events", []):
        recorder.events.append(TraceEvent(**event))
    recorder.write_jsonl(trace_output_path, state_snapshot=agent_state.to_record())

    total_trace_latency = sum(int(e.get("latency_ms", 0)) for e in result.get("trace_events", []))
    return {
        "task_id": task_id,
        "task_text": task_text,
        "final_answer": final_answer,
        "used_tool": bool(result.get("need_tool", False)),
        "tool_name": result.get("tool_name"),
        "token_usage": result.get("token_usage", {}),
        "trace_event_count": len(result.get("trace_events", [])),
        "trace_latency_ms": total_trace_latency,
        "wall_ms": wall_ms,
        "trace_path": str(trace_output_path),
        "status": "ok" if final_answer else "error",
        "error": result.get("error", ""),
    }


def run_batch(
    *,
    benchmark_name: str,
    output_dir: Path,
    tasks: List[str],
    runtime_config: RuntimeConfig,
) -> Dict[str, Any]:
    api_key = (runtime_config and runtime_config.api_key_env and __import__("os").environ.get(runtime_config.api_key_env)) or ""
    if not api_key:
        raise RuntimeError(f"missing API key env: {runtime_config.api_key_env}")

    client = OpenAI(api_key=api_key, base_url=runtime_config.base_url)
    app = build_graph(client=client, model=runtime_config.model)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"stage4_single_agent_{ts}"
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        task_id = f"single-{idx:03d}"
        trace_path = traces_dir / f"{task_id}.jsonl"
        one = run_single_task(
            app=app,
            task_id=task_id,
            task_text=task,
            benchmark_name=benchmark_name,
            trace_output_path=trace_path,
        )
        results.append(one)

    results_path = run_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    ok_count = sum(1 for r in results if r["status"] == "ok")
    used_tool_count = sum(1 for r in results if r["used_tool"])
    total_tokens = sum(int(r["token_usage"].get("total_tokens", 0)) for r in results)
    total_wall_ms = sum(int(r["wall_ms"]) for r in results)
    summary = {
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "task_count": len(results),
        "ok_count": ok_count,
        "success_rate": round(ok_count / len(results), 4) if results else 0.0,
        "used_tool_count": used_tool_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_task": round(total_tokens / len(results), 2) if results else 0.0,
        "total_wall_ms": total_wall_ms,
        "avg_wall_ms_per_task": round(total_wall_ms / len(results), 2) if results else 0.0,
        "model": runtime_config.model,
        "base_url": runtime_config.base_url,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage4 single-agent ReAct baseline runner")
    p.add_argument("--benchmark-name", default="stage4_local_smoke")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--output-dir", default=str(ROOT / "runs"))
    p.add_argument("--runtime-config", default=str(DEFAULT_RUNTIME_CONFIG_PATH))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = load_runtime_config(Path(args.runtime_config))
    tasks = default_tasks()[: args.samples]
    summary = run_batch(
        benchmark_name=args.benchmark_name,
        output_dir=Path(args.output_dir),
        tasks=tasks,
        runtime_config=runtime_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
