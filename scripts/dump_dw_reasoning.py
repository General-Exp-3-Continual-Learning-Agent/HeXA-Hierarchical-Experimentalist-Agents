"""Extract per-agent reasoning traces from a Workflow transcript into an eval dir.

The Claude Code workflow runner writes one `agent-<id>.jsonl` per subagent under
its transcript dir, containing that agent's thinking blocks, tool calls and tool
results. This pulls those out into readable markdown + a machine-readable JSON
so a dynamic-workflow run's reasoning is archived alongside its trajectory.

    python scripts/dump_dw_reasoning.py \
        --transcript-dir ~/.claude/projects/<proj>/<session>/subagents/workflows/wf_xxx \
        --out-dir "Direct Results/eval_catapult_dw_opus5_seed4"
"""

import argparse
import glob
import json
import os


def _blocks(content):
    """Normalise a message `content` field to a list of blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def _label_from_prompt(prompt) -> str:
    """The runner's meta.json carries no label, so name the agent from its prompt."""
    if not prompt:
        return "agent"
    if "SINGLE SHOT" in prompt:
        return "direct-oneshot"
    if "arbiter" in prompt:
        return "commit-verify"
    for line in prompt.splitlines():
        if line.startswith("HYPOTHESIS FAMILY:"):
            return "hyp: " + line.split(":", 1)[1].strip()[:40]
    return next((l.strip()[:40] for l in prompt.splitlines() if l.strip()), "agent")


def read_agent(path: str) -> dict:
    meta_path = path.replace(".jsonl", ".meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    steps, prompt = [], None
    # Streaming writes the same assistant message several times; dedupe by msg id
    # so output_tokens is not multiple-counted.
    # Streaming rewrites the same assistant message repeatedly and the EARLY
    # snapshots carry a partial output_tokens count, so keep the largest usage
    # seen per message id rather than the first.
    per_msg, models = {}, set()
    TOK_KEYS = ("output_tokens", "input_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens")

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msg = rec.get("message") or {}
            role = msg.get("role")

            u, mid = msg.get("usage"), msg.get("id")
            if u and mid:
                prev = per_msg.get(mid)
                if prev is None or (u.get("output_tokens") or 0) > (prev.get("output_tokens") or 0):
                    per_msg[mid] = u
                if msg.get("model"):
                    models.add(msg["model"])
            for b in _blocks(msg.get("content")):
                t = b.get("type")
                if role == "user" and t == "text" and prompt is None:
                    prompt = b["text"]
                elif t == "thinking":
                    steps.append({"kind": "thinking", "text": b.get("thinking", "")})
                elif t == "text" and role == "assistant":
                    steps.append({"kind": "text", "text": b.get("text", "")})
                elif t == "tool_use":
                    steps.append({"kind": "tool_use", "name": b.get("name"),
                                  "input": b.get("input")})
                elif t == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c
                                      if isinstance(x, dict))
                    steps.append({"kind": "tool_result", "text": str(c)})

    tok = {k: sum(u.get(k) or 0 for u in per_msg.values()) for k in TOK_KEYS}
    seen_msgs = per_msg

    return {
        "agent_id": os.path.basename(path).replace("agent-", "").replace(".jsonl", ""),
        "label": meta.get("label") or meta.get("name") or _label_from_prompt(prompt),
        "phase": meta.get("phase") or "?",
        "meta": meta,
        "prompt": prompt,
        "models": sorted(models),
        "tokens": tok,
        "assistant_messages": len(seen_msgs),
        "steps": steps,
        # NOTE: the runner stores thinking blocks with their text stripped (only a
        # signature survives), so thinking_chars is usually 0 — the recoverable
        # reasoning is the visible output, the tool calls, and the structured result.
        "thinking_blocks": sum(1 for s in steps if s["kind"] == "thinking"),
        "thinking_chars": sum(len(s["text"]) for s in steps if s["kind"] == "thinking"),
        "tool_calls": sum(1 for s in steps if s["kind"] == "tool_use"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--transcript-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-result-chars", type=int, default=4000,
                   help="Truncate long tool results in the markdown view.")
    a = p.parse_args()

    tdir = os.path.expanduser(a.transcript_dir)
    os.makedirs(a.out_dir, exist_ok=True)

    agents = [read_agent(fp) for fp in sorted(glob.glob(os.path.join(tdir, "agent-*.jsonl")))]

    journal = []
    jp = os.path.join(tdir, "journal.jsonl")
    if os.path.exists(jp):
        with open(jp) as f:
            for line in f:
                if line.strip():
                    journal.append(json.loads(line))

    totals = {k: sum(ag["tokens"][k] for ag in agents) for k in
              ("output_tokens", "input_tokens",
               "cache_creation_input_tokens", "cache_read_input_tokens")}

    out_json = os.path.join(a.out_dir, "dw_reasoning.json")
    with open(out_json, "w") as f:
        json.dump({"transcript_dir": tdir, "token_totals": totals,
                   "agents": agents, "journal": journal}, f, indent=2)

    with open(os.path.join(a.out_dir, "dw_tokens.json"), "w") as f:
        json.dump({
            "transcript_dir": tdir,
            "totals": totals,
            "per_agent": [
                {"agent_id": ag["agent_id"], "label": ag["label"],
                 "models": ag["models"], "assistant_messages": ag["assistant_messages"],
                 "tool_calls": ag["tool_calls"], "thinking_blocks": ag["thinking_blocks"],
                 **ag["tokens"]}
                for ag in agents
            ],
        }, f, indent=2)

    out_md = os.path.join(a.out_dir, "dw_reasoning.md")
    with open(out_md, "w") as f:
        f.write("# Dynamic-workflow reasoning trace\n\n")
        f.write(f"Transcript: `{tdir}`\n\n")
        for ag in agents:
            title = ag["label"] or ag["agent_id"]
            f.write(f"\n---\n\n## {title}  (phase: {ag['phase']}, "
                    f"agent `{ag['agent_id']}`)\n\n")
            f.write(f"- thinking blocks: {ag['thinking_blocks']} "
                    f"({ag['thinking_chars']} chars retained) | "
                    f"tool calls: {ag['tool_calls']}\n\n")
            if ag["prompt"]:
                f.write("<details><summary>prompt</summary>\n\n```\n"
                        f"{ag['prompt']}\n```\n\n</details>\n\n")
            for s in ag["steps"]:
                if s["kind"] == "thinking":
                    f.write("### 🧠 thinking\n\n"
                            f"{s['text'] or '_(text not retained by the runner — signature only)_'}\n\n")
                elif s["kind"] == "text":
                    f.write(f"### 💬 output\n\n{s['text']}\n\n")
                elif s["kind"] == "tool_use":
                    f.write(f"### 🔧 {s['name']}\n\n```json\n"
                            f"{json.dumps(s['input'], indent=2)[:2000]}\n```\n\n")
                else:
                    txt = s["text"]
                    if len(txt) > a.max_result_chars:
                        txt = txt[:a.max_result_chars] + "\n… [truncated]"
                    f.write(f"### 📋 result\n\n```\n{txt}\n```\n\n")

    total_think = sum(ag["thinking_chars"] for ag in agents)
    print(f"agents: {len(agents)} | thinking blocks: "
          f"{sum(ag['thinking_blocks'] for ag in agents)} "
          f"({total_think} chars retained) | tool calls: "
          f"{sum(ag['tool_calls'] for ag in agents)}")
    for ag in agents:
        print(f"  {(ag['label'] or ag['agent_id'])[:40]:<42} "
              f"out_tok={ag['tokens']['output_tokens']:>7} "
              f"cache_read={ag['tokens']['cache_read_input_tokens']:>9} "
              f"msgs={ag['assistant_messages']:>4} "
              f"tools={ag['tool_calls']:>3} "
              f"think_blocks={ag['thinking_blocks']:>3}")
    print(f"  {'TOTAL':<42} out_tok={totals['output_tokens']:>7} "
          f"cache_read={totals['cache_read_input_tokens']:>9}")
    print(f"  models: {sorted({m for ag in agents for m in ag['models']})}")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
