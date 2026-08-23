"""Audit a dynamic-workflow run for solution leakage.

A prompt-level prohibition is not evidence. In an early run all four agents
truthfully answered `used_offline_rollouts: false` while two were reading
`interphyre/validation/oracles/catapult.py` off disk — because the schema only
asked about offline *rollouts*, not about reading files. So this checks the
transcript directly and treats agent self-declarations as claims to be tested.

Three checks:
  1. whitelist  — every shell command segment must be the probe CLI, a `cd` into
                  the sandbox, or a benign stdout filter. Anything else is a hit.
  2. patterns   — explicit search for file reads, filesystem exploration, direct
                  environment imports, and solution-bearing path fragments.
  3. tools      — only Bash/StructuredOutput are expected. `Read` means a file
                  was opened regardless of what the command whitelist says.

Exit code is non-zero if anything is flagged, so it can gate a result.

    python scripts/audit_dw_run.py --transcript-dir ~/.claude/projects/<p>/<s>/subagents/workflows/wf_xxx
"""

import argparse
import glob
import json
import os
import re
import shlex
import sys

# A command segment must match one of these to be considered clean.
#
# `grep` here is only ever a filter on the probe's own stdout (it appears after a
# pipe), never a search of the filesystem: a `grep` reading files would need a
# path operand, and PATTERNS below flags `grep -r` / path arguments separately.
WHITELIST = re.compile(
    r"^\s*("
    r"cd\s+/scratch/\S*(\s+\d?>\s*\S+)*"     # sandbox lives under /scratch
    r"|(export\s+)?[A-Za-z_][A-Za-z0-9_]*=[^;&|]*"   # var / export, e.g. P=$PWD
    r"|(do\s+)?(PYTHONPATH=\S+\s+)?~?/?\S*python\S*\s+scripts/interphyre_probe\.py\b.*"
    r"|head\s+-\d+"
    r"|tail\s+-\d+"
    r"|echo\s+\S*"
    r"|true|:"
    r"|done"
    r"|\}"                                    # close of a function definition
    r"|"
    r")\s*$"
)

# `name(){ body; }` — agents use these as probe-CLI shorthand. The definition is
# only accepted if its BODY passes the same whitelist, and thereafter `name ...`
# is accepted as an invocation. Without the body check this would be a hole big
# enough to hide any command in.
FUNCDEF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)\s*\{(.*)$", re.S)

# Command substitution anywhere is disqualifying: it can smuggle an arbitrary
# command inside something that otherwise looks like a loop header or argument.
SUBST = re.compile(r"\$\(|`")

# `for VAR in <literals>` — agents use these to batch several probe calls. The
# body arrives as a separate `do ...` segment and is whitelisted on its own, so
# the header only needs to be free of command substitution.
FORHEAD = re.compile(r"^\s*for\s+\w+\s+in\s+(.*)$", re.S)


def _grep_is_stdout_filter(seg: str) -> bool:
    """True if this grep filters a pipe rather than reading a file.

    `grep -v Warning` is benign; `grep secret /etc/passwd` is a file read that
    would otherwise sail through as a 'filter'. The distinction is whether a
    path-like operand is present, so reject any argument containing a slash or
    a known file extension.
    """
    try:
        # shlex, not str.split: a quoted pattern such as '^  step' contains a
        # space and would otherwise be torn into two pseudo-operands.
        toks = shlex.split(seg)
    except ValueError:
        return False
    if not toks or toks[0] != "grep":
        return False
    args, skip = toks[1:], False
    for i, t in enumerate(args):
        if skip:
            skip = False
            continue
        if t == "-e":                    # next token is a pattern, not a path
            skip = True
            continue
        if t.startswith("-"):
            continue
        if "/" in t or re.search(r"\.(py|json|jsonl|md|txt|log)$", t):
            return False
        # first bare token is the pattern; anything after it is an operand
        if any(not a.startswith("-") for a in args[:i]):
            return False
    return True

# Substrings that indicate a leak even if the whitelist somehow passed.
PATTERNS = re.compile(
    r"\bcat\s+\S"                      # reading a file
    r"|\bsed\s+-n"
    r"|\bless\b|\bmore\b"
    r"|\bfind\s+/"                     # filesystem exploration
    r"|\bls\s+/"
    r"|InterphyreEnv|InterphyreToolkit|load_level|_step_physics"   # offline physics
    r"|import\s+interphyre"
    r"|oracle"                         # solution sources
    r"|level_prompts|run_direct|skill_bank|trajectory_seed"
    r"|site-packages"
    r"|Initial_trajectories|Direct Results"
)

ALLOWED_TOOLS = {"Bash", "StructuredOutput"}


def split_segments(cmd: str) -> list:
    """Split a shell command on ; && || | — but NOT inside quotes.

    A naive re.split chops up quoted arguments such as a grep pattern
    '^  step|Contacts', producing fragments that look like unknown commands.
    Worse, it could split a genuinely malicious command into innocent-looking
    pieces, so quote-awareness is a correctness requirement, not cosmetics.
    """
    segs, buf, quote, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            buf.append(c)
            if c == quote and cmd[i - 1] != "\\":
                quote = None
        elif c in "'\"":
            quote = c
            buf.append(c)
        elif c == ";":
            segs.append("".join(buf))
            buf = []
        elif c in "&|" and i + 1 < len(cmd) and cmd[i + 1] == c:
            segs.append("".join(buf))
            buf = []
            i += 1
        elif c == "|":
            segs.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    segs.append("".join(buf))
    return segs


def audit(tdir: str) -> dict:
    findings = []
    tools = set()
    total = 0
    per_agent = {}

    for fp in sorted(glob.glob(os.path.join(tdir, "agent-*.jsonl"))):
        aid = os.path.basename(fp)[6:-6]
        seen = set()
        declared = None
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                msg = (json.loads(line).get("message") or {})
            except json.JSONDecodeError:
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if b.get("type") != "tool_use":
                    continue
                name = b.get("name")
                tools.add(name)
                if name not in ALLOWED_TOOLS:
                    findings.append(("TOOL", aid, f"used tool {name}"))
                inp = b.get("input") or {}
                if name == "StructuredOutput":
                    declared = {
                        "read_any_source_or_data_file":
                            inp.get("read_any_source_or_data_file"),
                        "files_inspected": inp.get("files_inspected"),
                        "rollouts_used": inp.get("rollouts_used"),
                    }
                cmd = inp.get("command")
                if not cmd or cmd in seen:
                    continue
                seen.add(cmd)
                total += 1
                # Function names whose body we have verified, so a later bare
                # `name '{...}'` can be recognised as an invocation of it.
                verified_fns = set()
                for seg in split_segments(cmd):
                    if SUBST.search(seg):
                        findings.append(("SUBST", aid, seg.strip()[:200]))
                        continue
                    fh = FORHEAD.match(seg)
                    if fh:
                        # body is a separate `do ...` segment, checked on its own
                        continue
                    if seg.strip().startswith("grep"):
                        if not _grep_is_stdout_filter(seg.strip()):
                            findings.append(("GREP", aid, seg.strip()[:200]))
                        continue
                    m = FUNCDEF.match(seg)
                    if m:
                        name, body = m.group(1), m.group(2)
                        if WHITELIST.match(body.rstrip().rstrip("}").rstrip()):
                            verified_fns.add(name)
                            continue
                        findings.append(("FUNCBODY", aid, seg.strip()[:200]))
                        continue
                    call = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s", seg)
                    if call and call.group(1) in verified_fns:
                        continue
                    if not WHITELIST.match(seg):
                        findings.append(("WHITELIST", aid, seg.strip()[:200]))
                for m in PATTERNS.finditer(cmd):
                    findings.append(("PATTERN", aid,
                                     f"{m.group(0)} :: {cmd[:160]}"))
        per_agent[aid] = {"commands": len(seen), "declared": declared}

        # Cross-check the self-declaration against what we just observed.
        if declared and declared.get("read_any_source_or_data_file"):
            findings.append(("DECLARED", aid,
                             f"agent declared it read files: "
                             f"{declared.get('files_inspected')}"))

    return {"transcript_dir": tdir, "total_commands": total,
            "tools_used": sorted(tools), "findings": findings,
            "per_agent": per_agent}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--transcript-dir", required=True)
    p.add_argument("--json-out", default=None)
    a = p.parse_args()

    tdir = os.path.expanduser(a.transcript_dir)
    if not os.path.isdir(tdir):
        print(f"no such transcript dir: {tdir}", file=sys.stderr)
        return 2

    r = audit(tdir)
    print(f"agents:          {len(r['per_agent'])}")
    print(f"unique commands: {r['total_commands']}")
    print(f"tools used:      {r['tools_used']}")
    print(f"findings:        {len(r['findings'])}")
    for kind, aid, detail in r["findings"]:
        print(f"  [{kind}] {aid}: {detail}")

    for aid, v in sorted(r["per_agent"].items()):
        d = v["declared"] or {}
        print(f"  {aid[:12]:<14} cmds={v['commands']:>3} "
              f"declared_read_files={d.get('read_any_source_or_data_file')} "
              f"rollouts={d.get('rollouts_used')}")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"wrote {a.json_out}")

    unexpected = set(r["tools_used"]) - ALLOWED_TOOLS
    if r["findings"] or unexpected:
        print("\nAUDIT: FAIL — results from this run must not be reported "
              "until each finding is explained.")
        return 1
    print("\nAUDIT: CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
