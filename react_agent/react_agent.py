"""
Core ReAct loop that orchestrates model <-> tool interaction for solving
Interphyre physics puzzles.
"""

import re
import json
import time
from typing import Optional, Tuple, List, Dict, Any

from react_agent.level_prompts import (
    build_system_prompt, build_initial_user_message, format_observation,
    OSS_FORMAT_ADDENDUM, OSS_RETRY_NUDGE, get_oss_tools,
)
from react_agent.tools import InterphyreToolkit

# Control tokens used by gpt-oss-20b's native channel format.
# These are NOT registered as special tokens, so skip_special_tokens=True won't
# strip them.  We also include <|constrain|> and <|endofprompt|>.
_OSS_CONTROL_TOKENS = re.compile(
    r"<\|(?:start|end|channel|message|call|return|constrain|endofprompt)\|>"
)

# Matches the analysis channel body:
#   <|channel|>analysis<|message|>THOUGHT<|end|>
# After _OSS_CONTROL_TOKENS replaces tokens with spaces, the pattern becomes:
#   analysis  THOUGHT
# We also accept "assistantanalysis" / "analysis" tags that leak through.
_OSS_ANALYSIS_RE = re.compile(
    r"(?:assistant)?analysis\s+(.*?)(?=\s+(?:assistant)?(?:commentary|final)|"
    r"\s+to=functions\.|$)",
    re.DOTALL,
)

# Matches a tool call:
#   to=functions.TOOL_NAME ... commentary json  {ARGS}
# where ARGS may contain nested braces.
_OSS_TOOL_CALL_RE = re.compile(
    r"to=\s*functions\.(\w+).*?(\{(?:[^{}]|\{[^{}]*\})*\})",
    re.DOTALL,
)


def _extract_coords_from_thought(thought: str) -> str:
    """
    When the OSS model generates empty {} args for a tool call but mentions
    coordinates in its analysis/thought, extract x, y, radius from the thought.

    Returns a JSON string like '{"x": 3.5, "y": 4.0, "radius": 0.7}' or '{}'
    if no coordinates are found.
    """
    import json as _json

    # Look for explicit x=..., y=..., radius=... patterns
    # Use findall to get the LAST mentioned values (most recent in reasoning)
    _NUM = r'(-?\d+(?:\.\d+)?)'
    x_all = re.findall(r'\bx\s*[=:]\s*' + _NUM, thought)
    y_all = re.findall(r'\by\s*[=:]\s*' + _NUM, thought)
    r_all = re.findall(r'\bradius\s*[=:]\s*' + _NUM, thought)
    x_m = x_all[-1] if x_all else None
    y_m = y_all[-1] if y_all else None
    r_m = r_all[-1] if r_all else None

    if x_m and y_m:
        coords = {"x": float(x_m), "y": float(y_m)}
        coords["radius"] = float(r_m) if r_m else 0.5
        return _json.dumps(coords)

    # Try "at (x, y)" or "(x, y, radius)" pattern
    _NUM2 = r'(-?\d+(?:\.\d+)?)'
    coord_m = re.search(
        r'at\s*\(\s*' + _NUM2 + r'\s*,\s*' + _NUM2 + r'(?:\s*,\s*' + _NUM2 + r')?\s*\)',
        thought,
    )
    if coord_m:
        coords = {"x": float(coord_m.group(1)), "y": float(coord_m.group(2))}
        coords["radius"] = float(coord_m.group(3)) if coord_m.group(3) else 0.5
        return _json.dumps(coords)

    return "{}"


# ── Tool-name sets for regex-based fallback detection ──────────────────────
_KNOWN_TOOLS_NO_ARGS = frozenset({
    "get_level_state", "get_contact_log", "describe_scene_geometry",
})
_KNOWN_TOOLS_WITH_COORDS = frozenset({
    "simulate_action", "validate_action", "predict_first_contact", "finish",
})
_KNOWN_TOOLS_WITH_TRACE = frozenset({"simulate_with_trace"})
_ALL_KNOWN_TOOLS = _KNOWN_TOOLS_NO_ARGS | _KNOWN_TOOLS_WITH_COORDS | _KNOWN_TOOLS_WITH_TRACE

# Sorted longest-first so longer names match before shorter prefixes
_TOOL_MENTION_RE = re.compile(
    r"\b(" + "|".join(sorted(_ALL_KNOWN_TOOLS, key=len, reverse=True)) + r")\b"
)

# Matches JSON-encoded function calls: {"name": "tool", "arguments": {...}}
_JSON_FN_CALL_RE = re.compile(
    r'"(?:name|function|tool)"\s*:\s*"(\w+)"[^}]*?"(?:arguments?|parameters?|input)"\s*:\s*(\{[^}]*\})',
    re.DOTALL,
)


def _truncate_messages(messages: List[Dict[str, Any]], keep_recent: int = 8) -> List[Dict[str, Any]]:
    """
    Keep system prompt + initial user message + most recent `keep_recent` turns.

    A "turn" here is one assistant message (with optional tool result that
    follows it).  Used to bound prefill context length on long ReAct chains
    where eager attention's O(seq²) activation matrix can OOM the GPU.
    """
    if len(messages) <= 2:
        return messages
    # Always keep the first system + first user message
    head = messages[:2]
    tail = messages[2:]
    # Each ReAct turn for OSS = 2 messages (assistant tool_call + tool result).
    # For non-OSS = 2 messages (assistant + user observation).  So multiply by 2.
    keep_msgs = keep_recent * 2
    if len(tail) <= keep_msgs:
        return messages
    return head + tail[-keep_msgs:]


def _compact_old_observations(
    messages: List[Dict[str, Any]],
    keep_recent: int = 4,
    max_chars_when_old: int = 600,
) -> List[Dict[str, Any]]:
    """Return a copy of `messages` with OLD observations head-and-tail truncated.

    For Claude (non-OSS) ReAct chains, the prompt grows linearly with iteration
    count because every prior Thought / Action / Observation triple is kept in
    full. The `Observation` slot is the chunky one — a `simulate_with_trace`
    output can be 1k+ tokens. Past iteration ~5 this dominates the prompt.

    Strategy: keep the system prompt + initial user message untouched, keep the
    `keep_recent` most recent (assistant, observation) pairs verbatim, and for
    every OLDER observation truncate the content to a head + tail snippet so
    the model still sees the important markers (SUCCESS / INVALID / final
    positions sit at the start; any tail summary stays at the end). The
    assistant turns (Thought / Action / Action Input lines) are NEVER compacted
    — they're short and load-bearing for the model's reasoning chain.

    This is purely a prompt-builder transform; the in-memory `messages` list
    and the saved trajectory are untouched, so the per-iteration JSON dump
    retains full fidelity for offline analysis.
    """
    if len(messages) <= 2:
        return messages
    head = messages[:2]
    tail = messages[2:]
    keep_msgs = keep_recent * 2  # one assistant + one user per turn
    if len(tail) <= keep_msgs:
        return messages
    old, recent = tail[:-keep_msgs], tail[-keep_msgs:]

    compacted: List[Dict[str, Any]] = []
    for msg in old:
        # Only compact the OBSERVATION slot (role=user). Assistant Thought/Action
        # lines stay verbatim — they're the reasoning chain the model relies on.
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            content = msg["content"]
            if len(content) > max_chars_when_old:
                head_len = int(max_chars_when_old * 0.67)
                tail_len = max_chars_when_old - head_len - 60  # marker overhead
                elided = len(content) - head_len - tail_len
                content = (
                    content[:head_len]
                    + f"\n  ...[elided {elided} chars to save context]...\n"
                    + content[-tail_len:]
                )
                compacted.append({**msg, "content": content})
                continue
        compacted.append(msg)
    return head + compacted + recent


def _detect_tool_mention(text: str, thought: str) -> Optional[str]:
    """
    Last-resort tool detection: scan text for known tool names and reconstruct
    a valid ReAct action.  Called only when all structured patterns have failed.
    Returns a ReAct-formatted string or None.
    """
    # Check for JSON-encoded function call: {"name": "tool", "arguments": {...}}
    json_fn_m = _JSON_FN_CALL_RE.search(text)
    if json_fn_m:
        tool = json_fn_m.group(1)
        args = json_fn_m.group(2)
        if tool in _ALL_KNOWN_TOOLS:
            return f"Thought: {thought}\nAction: {tool}\nAction Input: {args}"

    # Scan text for known tool names — use the *last* match (post-reasoning intent)
    matches = list(_TOOL_MENTION_RE.finditer(text))
    if not matches:
        return None

    tool_name = matches[-1].group(1)

    if tool_name in _KNOWN_TOOLS_NO_ARGS:
        return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {{}}"

    if tool_name in _KNOWN_TOOLS_WITH_COORDS:
        # Search nearby text + thought for coordinates
        nearby = text[max(0, matches[-1].start() - 150):matches[-1].end() + 300]
        coords_json = _extract_coords_from_thought(nearby + " " + thought)
        return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {coords_json}"

    if tool_name in _KNOWN_TOOLS_WITH_TRACE:
        obj_names_m = re.search(r'"object_names"\s*:\s*(\[[^\]]*\])', text)
        coords_json = _extract_coords_from_thought(thought)
        try:
            coords = json.loads(coords_json) if coords_json != "{}" else {}
        except json.JSONDecodeError:
            coords = {}
        obj_part = f'"object_names": {obj_names_m.group(1)}' if obj_names_m else '"object_names": []'
        coord_part = (
            f', "x": {coords["x"]}, "y": {coords["y"]}, "radius": {coords.get("radius", 0.5)}'
            if coords.get("x") is not None else ""
        )
        return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {{{obj_part}{coord_part}}}"

    return None


def _parse_oss_native(raw: str) -> str:
    """
    Convert gpt-oss-20b's native channel output to standard ReAct format.

    The model generates structured output using control tokens:
      <|channel|>analysis<|message|>THOUGHT<|end|>
      <|start|>assistant to=functions.TOOL<|channel|>commentary json<|message|>ARGS<|call|>
      <|channel|>final<|message|>RESPONSE<|end|>

    We strip control tokens and parse what remains.
    """
    # Strip control tokens — they appear as literal text since they're not "special"
    cleaned = _OSS_CONTROL_TOKENS.sub(" ", raw)

    # Strip repetitive scratchpad loops:
    #   "assistantanalysis to=assistantcommentary" (or partial variants)
    cleaned = re.sub(
        r"(?:ant|assistant)?(?:analysis|commentary)\s*to=\s*(?:assistant)?(?:analysis|commentary)\s*",
        " ", cleaned
    )

    # Collapse whitespace (preserve newlines for ReAct format)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned).strip()

    # ── Pass-through if model produced ReAct format ──
    if re.search(r"Thought\s*:", cleaned):
        m = re.search(r"Thought\s*:", cleaned)
        return cleaned[m.start():]

    # ── Extract thought from "analysis" channel ──
    thought_text = ""
    thought_match = _OSS_ANALYSIS_RE.search(cleaned)
    if thought_match:
        thought_text = thought_match.group(1).strip()
        thought_text = re.sub(r"\b(commentary|final|assistant|json)\b", "", thought_text).strip()

    # ── Extract tool call ──
    tool_match = _OSS_TOOL_CALL_RE.search(cleaned)
    if tool_match:
        tool_name = tool_match.group(1)
        tool_args = tool_match.group(2)
        # If args are empty {} and the tool needs coordinates, try extracting
        # from the thought text (OSS model often reasons about coordinates in
        # the analysis channel but generates empty args in the function call).
        if tool_args.strip() == "{}" and tool_name in (
            "simulate_action", "validate_action", "simulate_partial",
            "predict_first_contact",
        ) and thought_text:
            extracted = _extract_coords_from_thought(thought_text)
            if extracted != "{}":
                tool_args = extracted
        return f"Thought: {thought_text}\nAction: {tool_name}\nAction Input: {tool_args}"

    # ── Fallback: bare "action=TOOL" or "action: TOOL" patterns ──
    bare_action = re.search(
        r"action[=:]\s*(\w+)\s+(?:Input[=:]\s*)?(\{[^}]*\})",
        cleaned, re.IGNORECASE,
    )
    if bare_action:
        return (
            f"Thought: {thought_text}\n"
            f"Action: {bare_action.group(1)}\n"
            f"Action Input: {bare_action.group(2)}"
        )

    # ── "final" channel (no tool call, just a response / finish) ──
    final_match = re.search(r"final\s+(.*)", cleaned, re.DOTALL)
    if final_match:
        content = final_match.group(1).strip()
        json_match = re.search(r"(\{[^}]*\})", content)
        if json_match:
            return f"Thought: {thought_text}\nAction: finish\nAction Input: {json_match.group(1)}"

    # Last resort: if there is a JSON blob anywhere, pair it with the thought
    json_anywhere = re.search(r"(\{[^}]*\"x\"[^}]*\})", cleaned)
    if json_anywhere and thought_text:
        return f"Thought: {thought_text}\nAction: simulate_action\nAction Input: {json_anywhere.group(1)}"

    # Regex scan for any known tool name mentioned in the cleaned text
    detected = _detect_tool_mention(cleaned, thought_text)
    if detected:
        return detected

    # Absolute fallback: return a safe ReAct that re-inspects the scene rather
    # than raw text that will cause action_name=None and a wasted retry iteration.
    safe_thought = thought_text or "(unable to parse model output — re-inspecting scene)"
    return f"Thought: {safe_thought}\nAction: get_level_state\nAction Input: {{}}"


def load_qwen_model(model_name: str, temperature: float = 0.3, max_new_tokens: int = 800):
    """
    Load a Qwen/GPT-OSS-20B text-only LLM from HuggingFace.
    Returns a callable that takes a list of messages and returns generated text.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Qwen model: {model_name} ...")

    # Resolve HF model path
    if "/" not in model_name:
        hf_name = f"Qwen/{model_name}"
    else:
        hf_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    is_oss = "gpt-oss" in model_name.lower()
    if is_oss:
        # GptOssForCausalLM does not support SDPA; eager is required.
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype="auto",
            device_map="auto",
        )
    print(f"Model loaded successfully: {hf_name}")

    # For OSS models, <|call|> (200012) marks the end of a tool call in the
    # native channel format.  Adding it as an EOS token prevents the model
    # from hallucinating observations after a tool call.
    _oss_eos_ids = [tokenizer.eos_token_id]  # <|return|> (200002)
    if is_oss:
        _CALL_TOKEN_ID = 200012   # <|call|>
        _oss_eos_ids.append(_CALL_TOKEN_ID)

    # Cap OSS generation to prevent massive analysis dumps that exhaust tokens
    # before reaching a tool call.  The native format typically needs ~200 tokens
    # (analysis + function call + JSON args).
    _oss_max_tokens = min(max_new_tokens, 400)

    def generate(messages: List[Dict[str, str]], temp: float = temperature, max_tokens: int = max_new_tokens) -> str:
        """Generate a response from the model given a message list."""
        if is_oss:
            max_tokens = _oss_max_tokens
            patched = list(messages)
            # Inject the brief format addendum into system prompt
            if patched and patched[0]["role"] == "system":
                patched[0] = {"role": "system", "content": patched[0]["content"] + OSS_FORMAT_ADDENDUM}
            # Keep system + first user message fixed; also preserve early
            # informational tool exchanges (get_level_state, gap_analysis,
            # relative_positions) so the model never forgets the scene layout.
            # Then retain only the last MAX_TURNS exchanges.
            if len(patched) > 2:
                MAX_TURNS = 10
                _INFO_TOOLS = {"get_level_state", "compute_gap_analysis", "compute_relative_positions"}
                # Collect pinned early messages: system + initial user + first info tool calls
                pinned = patched[:2]
                seen_info = set()
                for idx in range(2, min(len(patched), 16)):
                    msg = patched[idx]
                    # Pin assistant messages that call info tools (+ their tool responses)
                    tool_calls = msg.get("tool_calls", [])
                    fname = tool_calls[0].get("function", {}).get("name", "") if tool_calls else ""
                    # Also check content string for non-tool_calls format
                    content = msg.get("content", "")
                    is_info_call = fname in _INFO_TOOLS or (
                        msg.get("role") == "assistant" and any(t in content for t in _INFO_TOOLS)
                    )
                    if is_info_call and fname not in seen_info:
                        seen_info.add(fname or content[:30])
                        pinned.append(msg)
                    elif msg.get("role") == "tool" and idx > 0 and patched[idx - 1] in pinned:
                        pinned.append(msg)
                pinned_count = len(pinned)
                tail = patched[-(MAX_TURNS * 2):]
                # Avoid duplicates if tail overlaps with pinned section
                if len(patched) - (MAX_TURNS * 2) > pinned_count:
                    patched = pinned + tail
                # else: conversation is short enough, keep as-is
            # Use proper tool-calling template — the model gets TypeScript
            # function defs and knows how to call them natively.
            oss_tools = getattr(generate, "_oss_tools", None)
            text = tokenizer.apply_chat_template(
                patched,
                tools=oss_tools,
                tokenize=False,
                add_generation_prompt=True,
                model_identity="You are a physics puzzle solving assistant.",
                reasoning_effort="medium",
            )
        else:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        eos_ids = _oss_eos_ids if is_oss else tokenizer.eos_token_id
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        if is_oss:
            # Decode WITHOUT stripping special tokens — the model's control
            # tokens (<|channel|>, <|start|>, etc.) are NOT registered as
            # special, so skip_special_tokens=True silently drops them.
            raw = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            response = _parse_oss_native(raw)
            # Final cleanup: normalize any remaining format variants
            response = re.sub(r"\bcodeInput\s*:", "Action Input:", response)
            response = re.sub(r"\bcodeAction\s*:", "Action:", response)
            response = re.sub(
                r"(Action\s*:\s*)(\w+)\}?\((\{.*?\})\)",
                lambda m: f"{m.group(1)}{m.group(2)}\nAction Input: {m.group(3)}",
                response,
                flags=re.DOTALL,
            )
            response = re.sub(r"(Action\s*:\s*\w+)[\}\(\)\s]+$", r"\1", response, flags=re.MULTILINE)
        else:
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response.strip()

    return generate

def _strip_harmony_tags(text: str) -> str:
    """
    Strip gpt-oss harmony response format tags from generated text.
    The model emits tags like: assistantcommentary, assistantanalysis,
    <|channel|>...<|message|>, etc. that pollute the ReAct output.
    """
    # Remove harmony channel/message markers
    text = re.sub(r"<\|channel\|>[^<]*<\|message\|>", "", text)
    # Remove harmony role tags that leak into content
    text = re.sub(r"assistant(?:commentary|analysis|reasoning)\s*", "", text, flags=re.IGNORECASE)
    # Remove any remaining special tokens
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def _truncate_at_observation(text: str) -> str:
    """
    Truncate model output at the first 'Observation:' line.
    gpt-oss tends to hallucinate entire multi-turn conversations,
    imagining observations before they happen. We only want
    the first Thought + Action + Action Input block.
    """
    match = re.search(r"\nObservation:", text)
    if match:
        text = text[:match.start()]
    return text.strip()


def load_gpt_oss_model(model_name: str, temperature: float = 0.3, max_new_tokens: int = 800):
    """
    Load gpt-oss via Transformers with device_map="auto" across all available GPUs (BF16).
    Parses the harmony response format to extract the final response.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading gpt-oss model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"Model loaded successfully: {model_name}")

    # Pre-compute stop token IDs for "Observation:" to prevent multi-turn hallucination
    obs_token_ids = tokenizer.encode("\nObservation:", add_special_tokens=False)

    def generate(messages: List[Dict[str, str]], temp: float = temperature, max_tokens: int = max_new_tokens) -> str:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False)
        # Extract final message from harmony format
        if "<|channel|>final<|message|>" in response:
            response = response.split("<|channel|>final<|message|>")[-1]
        response = tokenizer.decode(
            tokenizer.encode(response, add_special_tokens=False),
            skip_special_tokens=True,
        ).strip()
        # Strip remaining harmony tags and truncate at hallucinated Observation
        response = _strip_harmony_tags(response)
        response = _truncate_at_observation(response)
        return response

    return generate

def load_openai_compatible_model(
    model_name: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.3,
    max_new_tokens: int = 800,
):
    """
    Load a model via an OpenAI-compatible API endpoint (e.g. a vLLM server).
    Returns a callable compatible with ReactAgent: generate(messages, temp, max_tokens) -> str.

    Typical usage: point base_url at a vLLM server started with:
      python -m vllm.entrypoints.openai.api_server \\
          --model <hf_path> --tensor-parallel-size N --port <port>
    No real API key needed — pass api_key="EMPTY" for local vLLM.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required for OpenAI-compatible models: pip install openai"
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"OpenAI-compatible model ready: {model_name} @ {base_url}")

    is_oss = "gpt-oss" in model_name.lower()

    def generate(
        messages: List[Dict[str, str]],
        temp: float = temperature,
        max_tokens: int = max_new_tokens,
    ) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        if is_oss:
            content = _strip_harmony_tags(content)
            content = _truncate_at_observation(content)
        return content

    return generate


def is_thinking_model(model_name: str) -> bool:
    """Check if a model is a Qwen3 Thinking model."""
    return "Thinking" in model_name


def load_qwen3_thinking_model(model_name: str, max_new_tokens: int = 32768):
    """
    Load a Qwen3 Thinking model from HuggingFace.
    Returns a callable that takes a list of messages and returns a dict
    with 'thinking' and 'content' keys.
    
    Per Qwen3 best practices:
    - Temperature=0.6, TopP=0.95, TopK=20
    - Thinking content must NOT be included in conversation history
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Qwen3 Thinking model: {model_name} ...")

    if "/" not in model_name:
        hf_name = f"Qwen/{model_name}"
    else:
        hf_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded successfully: {hf_name}")

    # </think> token id for Qwen3
    THINK_END_TOKEN_ID = 151668

    def generate(messages: List[Dict[str, str]], temp: float = 0.6, max_tokens: int = max_new_tokens) -> dict:
        """Generate a response, returning both thinking and content."""
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = output[0][inputs["input_ids"].shape[1]:].tolist()

        # Parse thinking vs content using </think> token
        try:
            index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
        except ValueError:
            index = 0

        thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return {"thinking": thinking, "content": content}

    return generate

class MockModel:
    """Mock model for testing the ReAct loop without a real LLM.

    Each episode runs up to _EPISODE_LEN simulate_action calls with random
    placements.  If any attempt returns SUCCESS the model immediately calls
    finish with those winning coordinates, mirroring how a real LLM would
    behave.  Otherwise the last call is converted to finish.

    Constraints (from world bounds): x, y in [-5, 5]; radius in [0.1, 1.5];
    ball must fit fully inside the world, so x in [-5+r, 5-r], y in [-5+r, 5-r].
    """

    _EPISODE_LEN = 25  # max calls per episode (up to 24 simulate + 1 finish)

    def __init__(self, level_name: str = "down_to_earth"):
        self.call_count = 0

    def __call__(self, messages, **kwargs):
        import random
        rng = random.SystemRandom()
        r = round(rng.uniform(0.1, 1.5), 2)
        x = round(rng.uniform(-5.0 + r, 5.0 - r), 2)
        y = round(rng.uniform(-5.0 + r, 5.0 - r), 2)
        self.call_count += 1
        args = f'{{"x": {x}, "y": {y}, "radius": {r}}}'

        # Check if any previous simulate_action returned SUCCESS.
        # Scan message pairs: assistant (action) followed by user (observation).
        # If found, immediately commit those winning coordinates via finish.
        winning_args = None
        for i in range(len(messages) - 1, 0, -1):
            msg = messages[i]
            if msg.get("role") == "user" and "SUCCESS" in msg.get("content", ""):
                prev = messages[i - 1]
                if prev.get("role") == "assistant":
                    m = re.search(r"Action Input:\s*(\{[^}]+\})", prev.get("content", ""))
                    if m:
                        winning_args = m.group(1)
                break

        if winning_args is not None:
            return (
                "Thought: A previous placement succeeded. Submitting it as the final answer.\n"
                "Action: finish\n"
                f"Action Input: {winning_args}"
            )

        if self.call_count % self._EPISODE_LEN == 0:
            return (
                "Thought: Tried enough placements. Submitting the last one.\n"
                "Action: finish\n"
                f"Action Input: {args}"
            )
        return (
            "Thought: I will simulate a random valid placement to evaluate it.\n"
            "Action: simulate_action\n"
            f"Action Input: {args}"
        )


def parse_react_output(text: str) -> Tuple[str, Optional[str], Optional[dict]]:
    """
    Parse the model output to extract Thought, Action, and Action Input.

    Returns:
        (thought, action_name, action_args)
        action_name is None if no action was found.
        action_args is a dict (possibly empty) or None.
    """
    thought = ""
    action_name = None
    action_args = None

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract Action
    action_match = re.search(r"Action:\s*(\S+)", text)
    if action_match:
        action_name = action_match.group(1).strip()

    # Extract Action Input (JSON)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if input_match:
        try:
            action_args = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            action_args = {}
    else:
        action_args = {}

    return thought, action_name, action_args


class ReactAgent:
    """ReAct agent for solving Interphyre physics puzzles."""

    def __init__(
        self,
        model_fn,
        toolkit: InterphyreToolkit,
        level_name: str = "down_to_earth",
        max_iterations: int = 10,
        verbose: bool = True,
        temperature: float = 0.3,
        max_new_tokens: int = 800,
        is_oss: bool = False,
    ):
        self.model_fn = model_fn
        self.toolkit = toolkit
        self.level_name = level_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.is_oss = is_oss

    @staticmethod
    def _make_tool_call_id(iteration: int) -> str:
        return f"call_{iteration}"

    def solve(self) -> dict:
        """
        Run the ReAct loop to solve the puzzle.

        Returns:
            dict with keys:
                - success: bool
                - action: tuple (x, y, radius) or None
                - iterations: int
                - trajectory: list of (thought, action, observation) tuples
                - final_observation: str
        """
        system_prompt = build_system_prompt(self.level_name, is_oss=self.is_oss)
        initial_message = build_initial_user_message(self.level_name, is_oss=self.is_oss)

        # For OSS models, register tool schemas on the generate function so
        # the template can render TypeScript definitions the model recognises.
        if self.is_oss:
            self.model_fn._oss_tools = get_oss_tools(self.level_name)

        # Build conversation as a list of messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_message},
        ]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  SYSTEM PROMPT")
            print(f"{'='*60}")
            print(system_prompt)
            print(f"\n{'='*60}")
            print(f"  USER PROMPT")
            print(f"{'='*60}")
            print(initial_message)
            print(f"{'='*60}\n")

        trajectory = []
        final_action = None
        final_observation = ""
        # Track last valid coordinates for OSS empty-args recovery
        _last_valid_coords = {}

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  Iteration {iteration}/{self.max_iterations}")
                print(f"{'='*60}")

            # Truncate older turns for OSS to bound prefill context.  Prevents
            # OOM on gpt-oss-120b where eager attention's O(seq²) activation
            # exceeds the ~20GB headroom left after model weights.
            #
            # For Claude (non-OSS), do a softer compaction: keep all turns in
            # the conversation but truncate OBSERVATION bodies older than the
            # `keep_recent` most recent turns. This holds the prompt size
            # roughly flat past iteration ~5 instead of growing linearly with
            # each verbose simulate_with_trace return. The saved trajectory is
            # untouched.
            if self.is_oss:
                messages_for_model = _truncate_messages(messages, keep_recent=6)
            else:
                messages_for_model = _compact_old_observations(
                    messages, keep_recent=4, max_chars_when_old=600
                )

            # Generate model response
            start_time = time.time()
            raw_response = self.model_fn(
                messages_for_model,
                temp=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            gen_time = time.time() - start_time

            # Handle thinking models (return dict) vs standard models (return str)
            if isinstance(raw_response, dict):
                thinking_content = raw_response.get("thinking", "")
                response = raw_response.get("content", "")
                if self.verbose and thinking_content:
                    print(f"\n[Thinking] ({gen_time:.1f}s)")
                    print(thinking_content[:500] + ("..." if len(thinking_content) > 500 else ""))
                    print(f"\n[Response]")
                    print(response)
                elif self.verbose:
                    print(f"\n[Model Response] ({gen_time:.1f}s)")
                    print(response)
            else:
                response = raw_response
                if self.verbose:
                    print(f"\n[Model Response] ({gen_time:.1f}s)")
                    print(response)

            # Parse the response
            thought, action_name, action_args = parse_react_output(response)

            # OSS empty-args recovery: when the model calls simulate_action
            # or validate_action with {} but previously used valid coords,
            # re-use the last known coordinates.
            if self.is_oss and action_name in (
                "simulate_action", "validate_action", "simulate_partial",
                "predict_first_contact",
            ) and (not action_args or not action_args.get("x")):
                if _last_valid_coords:
                    action_args = dict(_last_valid_coords)
                    if self.verbose:
                        print(f"[OSS Recovery] Empty args -> reusing last coords: {action_args}")

            # Track last valid coordinates for OSS
            if self.is_oss and action_name in (
                "simulate_action", "validate_action", "simulate_partial",
                "predict_first_contact",
            ) and action_args and action_args.get("x") is not None and action_args.get("y") is not None:
                _last_valid_coords = {
                    "x": action_args["x"],
                    "y": action_args["y"],
                    "radius": action_args.get("radius", 0.5),
                }

            if action_name is None:
                # Model didn't produce a valid action — append response and retry
                if self.verbose:
                    print("\n[Warning] No action found in model output. Retrying...")
                messages.append({"role": "assistant", "content": response})
                if self.is_oss:
                    retry_msg = OSS_RETRY_NUDGE
                else:
                    retry_msg = (
                        "You must follow the format: Thought: <reasoning>\n"
                        "Action: <tool_name>\nAction Input: <json args>\n\n"
                        "Please try again."
                    )
                messages.append({"role": "user", "content": retry_msg})
                trajectory.append((thought, None, "No action parsed — retrying"))
                continue

            # Handle finish
            if action_name.lower() == "finish":
                if action_args and "x" in action_args:
                    final_action = (
                        float(action_args["x"]),
                        float(action_args["y"]),
                        float(action_args["radius"]),
                    )
                if self.verbose:
                    print(f"\n[Finish] Final action: {final_action}")

                # Actually run the final simulation to check success
                if final_action:
                    final_observation = self.toolkit.simulate_action(*final_action)
                    success = "SUCCESS" in final_observation
                else:
                    final_observation = "No action provided."
                    success = False

                trajectory.append((thought, f"finish({action_args})", final_observation))

                return {
                    "success": success,
                    "action": final_action,
                    "iterations": iteration,
                    "trajectory": trajectory,
                    "final_observation": final_observation,
                }

            # Execute the tool
            if self.verbose:
                print(f"\n[Tool Call] {action_name}({action_args})")

            observation = self.toolkit.dispatch_tool(action_name, action_args or {})

            if self.verbose:
                print(f"\n[Observation]")
                print(observation)

            trajectory.append((thought, f"{action_name}({action_args})", observation))

            # Auto-finish: when a simulation tool reports success and we have valid
            # coords, submit immediately rather than waiting for a new model turn.
            # Prevents losing solutions found on the last iteration.
            # simulate_with_trace → "Success: True"; simulate_action → "SUCCESS! ..."
            _sim_success = (
                action_name in ("simulate_with_trace", "simulate_action")
                and ("Success: True" in observation or "SUCCESS" in observation)
                and action_args
                and action_args.get("x") is not None
                and action_args.get("y") is not None
            )
            if _sim_success:
                final_action = (
                    float(action_args["x"]),
                    float(action_args["y"]),
                    float(action_args.get("radius", 0.5)),
                )
                if self.verbose:
                    print(f"\n[Auto-Finish] Simulation returned Success:True → submitting {final_action}")
                final_observation = self.toolkit.simulate_action(*final_action)
                success = "SUCCESS" in final_observation
                trajectory.append(("(auto-finish after Success:True)", f"finish({action_args})", final_observation))
                return {
                    "success": success,
                    "action": final_action,
                    "iterations": iteration,
                    "trajectory": trajectory,
                    "final_observation": final_observation,
                }

            # Append to conversation history — use the model's native tool
            # message format for OSS so the chat template formats it correctly.
            if self.is_oss:
                call_id = self._make_tool_call_id(iteration)
                args_str = json.dumps(action_args or {})
                messages.append({
                    "role": "assistant",
                    "thinking": thought,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": action_name, "arguments": args_str},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "content": observation,
                    "tool_call_id": call_id,
                })
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": format_observation(observation)})

        # Max iterations reached without finish
        if self.verbose:
            print(f"\n[Max iterations reached ({self.max_iterations})]")

        return {
            "success": False,
            "action": final_action,
            "iterations": self.max_iterations,
            "trajectory": trajectory,
            "final_observation": "Max iterations reached without calling finish.",
        }
