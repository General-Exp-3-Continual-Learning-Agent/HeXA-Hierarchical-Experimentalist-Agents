"""The injection join: skill bank -> CORE's guidelines slot -> a measured prompt.

`PHASE3_DESIGN.md` §4b records this as Known issue 4: `retrieve_within_token_budget`
produces a `RetrievedBlock`, `Problem.build_prompt` accepts a guidelines string, and
**no code has ever joined them**. Consequently `Attempt.added_tokens` has never been
populated from a real injection, and the added-token figure -- the x-axis of the
Pareto comparison against CORE (0.92k), GEPA (1.29k), Episodic RAG (33.6k) and
MemRL (32.7k) -- is unverified. This module is that join.

Two things it must get right, both of which fail silently otherwise.

**1. Silent degradation to no-memory.** Every CORE prompt template gates its
guidelines slot on truthiness, so an empty block makes ``build_prompt(block)``
byte-identical to ``build_prompt(None)``. A budget too small for any skill therefore
produces a *no-memory run wearing a with-memory label* -- and the accuracy it
reports is real, just not measuring what the column header claims. `Injection`
carries `injected` and `starved` so a caller can assert rather than hope, and
`assert_injection_effective` makes that assertion available to the loop.

**2. Parent-problem contamination (§5b).** With `|D_train| = 5` and ~400 attempts
each, nothing stops the evolver writing a training problem's answer into a skill's
``example`` and that skill then being retrieved *for that same problem*. Confidence
rises; it looks like learning; it is memorization. It does not leak to eval -- eval is
a disjoint problem set -- but it corrupts every training-time utility estimate, which
is what confidence ranking, pruning and admission all consume.

CORE has this switch (`memory.can_apply_lesson_to_problem`, gated by
``--leave-parent-problems-out-in-train``, default **False**). HeXA has none at all.
Here it defaults **on**, which is deliberately *stricter* than CORE -- a deviation to
declare in the paper, not to hide, hence `Injection.excluded_parent` recording when
it actually fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from hexa_core.adapter import Problem
from hexa_core.hexa_bridge import RetrievedBlock, retrieve_within_token_budget
from hexa_core.tokens import added_tokens, counter_name, is_exact

# CORE's measured added-context figure is 0.92k tokens; 1000 keeps the Pareto
# comparison like-for-like (PHASE3_DESIGN.md §4c).
DEFAULT_INJECTION_BUDGET = 1000

#: How many task-tier skills reach the prompt. **16, not 8, and that is deliberate:** it
#: reproduces what every completed campaign actually did. The nominal default was
#: ``max_specific=8``, but the parent-exclusion over-fetch doubled it to 16 and nothing
#: trimmed it back (see `build_injected_prompt`), so a 10-skill bank was injected whole.
#: 16 exceeds any bank this project has built (`max_skills` is 10), so this default is
#: "no cap" -- historical behaviour, bit-for-bit, which keeps the completed results
#: reproducible.
#:
#: It is a knob because `WEAKNESSES.md` §11 measured the cost of a *wide* block: the
#: injected text is not merely context, it is a list of things the solver believes it
#: must do per candidate. On Matchstick a 10-skill / 1017-token block ran 18-20%
#: truncation, and truncation there is worth 0.176 of accuracy -- more than every
#: remaining reasoning error combined. Lowering this trims the block from its
#: low-confidence tail, which is where §11 found the weakest skills sit. §5 (retrieval
#: cost) is the open item this exists to test rather than argue about.
DEFAULT_MAX_INJECTED_SKILLS = 16


@dataclass
class Injection:
    """A prompt built from the bank, with everything needed to audit it.

    ``prompt`` is what the model sees. Everything else exists because a number in a
    results table needs provenance: how many tokens were added, whether the budget
    truncated the block, whether anything was injected at all, and whether the token
    count is exact or heuristic.
    """

    problem: Problem
    prompt: str
    baseline_prompt: str
    block: RetrievedBlock
    added_tokens: int
    token_counter: str
    tokens_exact: bool
    excluded_parent: int = 0
    excluded_skill_ids: list[str] = field(default_factory=list)

    @property
    def injected(self) -> bool:
        """True iff the prompt actually differs from the no-memory prompt.

        Compares the rendered prompts rather than trusting the block to be
        non-empty: the truthiness gate in CORE's templates is the thing that
        matters, and this is a direct observation of it.
        """
        return self.prompt != self.baseline_prompt

    @property
    def starved(self) -> bool:
        """The bank offered something, but nothing survived to the prompt.

        The single most dangerous state in this module: a with-memory run that is
        silently a no-memory run. Distinct from an empty bank, which is `not
        self.block.skills and not self.block.mistakes` with no drops.
        """
        candidates = (
            len(self.block.skills)
            + len(self.block.mistakes)
            + self.block.dropped_skills
            + self.block.dropped_mistakes
            + self.excluded_parent
        )
        return candidates > 0 and not self.injected

    @property
    def budget_bound(self) -> bool:
        """The token budget dropped at least one retrieved candidate."""
        return self.block.budget_bound

    def to_dict(self) -> dict[str, Any]:
        """Flat record for `validation.jsonl` / the run manifest."""
        return {
            "problem_id": self.problem.id,
            "task_name": self.problem.task_name,
            "added_tokens": self.added_tokens,
            "token_counter": self.token_counter,
            "tokens_exact": self.tokens_exact,
            "injected": self.injected,
            "starved": self.starved,
            "budget_bound": self.budget_bound,
            "n_skills": len(self.block.skills),
            "n_mistakes": len(self.block.mistakes),
            "dropped_skills": self.block.dropped_skills,
            "dropped_mistakes": self.block.dropped_mistakes,
            "excluded_parent": self.excluded_parent,
            "excluded_skill_ids": list(self.excluded_skill_ids),
            "block_tokens": self.block.tokens,
        }


def _skill_parents(skill: Any) -> set[str]:
    """Problem ids a skill was distilled from.

    ``Skill.source_seeds`` is HeXA's field name; `attempt_to_trajectory` puts CORE
    problem ids there (HeXA's "seed" is our "problem id"). Values are coerced to
    ``str`` because HeXA's InterPhyre path stores ints and ours stores strings, and a
    silent type mismatch here would make exclusion never fire -- which looks exactly
    like exclusion being unnecessary.
    """
    return {str(seed) for seed in (getattr(skill, "source_seeds", None) or [])}


def filter_parent_problem(
    skills: Sequence[Any], problem_id: str
) -> tuple[list[Any], list[str]]:
    """Drop skills distilled from *problem_id*. Returns (kept, dropped_ids)."""
    kept, dropped = [], []
    for skill in skills:
        if problem_id in _skill_parents(skill):
            dropped.append(getattr(skill, "skill_id", "<unknown>"))
        else:
            kept.append(skill)
    return kept, dropped


def build_injected_prompt(
    problem: Problem,
    bank: Any,
    *,
    budget_tokens: int = DEFAULT_INJECTION_BUDGET,
    hierarchy_level: str | None = None,
    max_general: int = 0,
    max_specific: int = 8,
    max_mistakes: int = 4,
    parent_exclusion: bool = True,
    skip_oversized: bool = False,
) -> Injection:
    """Retrieve, exclude, render, and measure -- the whole injection path.

    Order matters and is not arbitrary: **parent exclusion happens before the token
    budget walk**. Excluding afterwards would let a parent skill consume budget that a
    usable skill could have taken, so the block would be smaller than it needed to be
    and the drop counts would misattribute the loss to the budget.

    ``hierarchy_level`` selects the retrieval key. §3 maps HeXA's ``general -> level``
    hierarchy onto ``task -> type``, so this is ``"matchstick"`` for the cross-type
    tier and ``"matchstick:operator"`` for a type-specific one. Defaults to the task
    name, which is the flat behaviour.
    """
    baseline_prompt = problem.build_prompt(None)
    level = hierarchy_level or problem.task_name
    # Under `task:type`, retrieval must also see the cross-type `task` tier -- which is
    # where `evolve_skill_bank` files everything. See `_TieredBank`.
    bank = tiered_bank(bank, level, problem.task_name)

    # Retrieve with headroom when exclusion is on: parent skills removed after the
    # top-N cut would otherwise shrink the effective candidate pool below max_specific.
    over_fetch = max_specific * 2 if parent_exclusion else max_specific
    block = retrieve_within_token_budget(
        bank,
        level,
        budget_tokens=budget_tokens,
        max_general=max_general,
        max_specific=over_fetch,
        max_mistakes=max_mistakes,
        skip_oversized=skip_oversized,
    )

    excluded_ids: list[str] = []
    if parent_exclusion and block.skills:
        kept, excluded_ids = filter_parent_problem(block.skills, problem.id)
        # Trim the over-fetch back down. Without this, `max_specific` is not a cap at
        # all whenever exclusion removes nothing (the usual case -- `excluded_parent` is
        # 0 in every completed run's summary): the block keeps all `max_specific * 2`
        # candidates. That is why a bank of 10 skills was injected whole under a nominal
        # `max_specific=8`, and why lowering the knob appeared to do nothing.
        trimmed = len(kept) > max_specific
        if trimmed:
            kept = kept[:max_specific]
        if excluded_ids or trimmed:
            # Re-render from the surviving candidates so block_text, skills and the
            # token count cannot disagree -- the invariant retrieve_within_token_budget
            # maintains by returning its last accepted render.
            block = _rerender(
                bank, level, kept, block, budget_tokens, max_mistakes, skip_oversized
            )

    prompt = problem.build_prompt(block.block_text or None)
    return Injection(
        problem=problem,
        prompt=prompt,
        baseline_prompt=baseline_prompt,
        block=block,
        added_tokens=added_tokens(baseline_prompt, prompt),
        token_counter=counter_name(),
        tokens_exact=is_exact(),
        excluded_parent=len(excluded_ids),
        excluded_skill_ids=excluded_ids,
    )


def _rerender(
    bank: Any,
    level: str,
    kept_skills: Sequence[Any],
    original: RetrievedBlock,
    budget_tokens: int,
    max_mistakes: int,
    skip_oversized: bool,
) -> RetrievedBlock:
    """Re-run the budget walk over an explicitly reduced skill list.

    Uses a throwaway bank view holding only ``kept_skills`` so the existing budget
    logic is reused verbatim rather than reimplemented -- the same "import, never
    reimplement" rule the adapter follows for CORE.
    """
    # `level` is passed explicitly: the survivors may span both hierarchy tiers, so
    # inferring the key from each skill's `source_level` would file the cross-type ones
    # under a key `retrieve_skills` is not asking about and drop them from the render.
    view = _BankView(bank, kept_skills, level)
    reduced = retrieve_within_token_budget(
        view,
        level,
        budget_tokens=budget_tokens,
        max_general=0,
        max_specific=len(kept_skills) or 1,
        max_mistakes=max_mistakes,
        skip_oversized=skip_oversized,
    )
    # Preserve the original drop counts: those candidates were lost to the *budget*,
    # and attributing them to exclusion (or losing them) would misreport why.
    return reduced._replace(
        dropped_skills=original.dropped_skills,
        dropped_mistakes=original.dropped_mistakes,
    )


class _BankView:
    """A read-only bank exposing a restricted skill list, delegating the rest.

    Deliberately not a `SkillBank` subclass: it must not be storable, evolvable or
    persistable, and duck-typing keeps it obviously a view.

    ``level`` is the key the surviving skills are served under. It must be passed
    explicitly rather than inferred from each skill's ``source_level``: under the
    ``task:type`` hierarchy the survivors legitimately come from two tiers
    (``matchstick`` and ``matchstick:operator``), so grouping by ``source_level``
    would file the cross-type ones under a key the retriever is not asking about and
    silently drop them from the re-render.
    """

    def __init__(self, bank: Any, skills: Sequence[Any], level: str | None = None) -> None:
        self._bank = bank
        self._skills = list(skills)
        self._level = level

    def _for(self, level: str) -> list[Any]:
        if self._level is not None:
            return list(self._skills) if level == self._level else []
        # Legacy flat behaviour: source_level is the retrieval key.
        return [s for s in self._skills if getattr(s, "source_level", None) == level]

    def get_skills_for_level(self, level: str) -> list[Any]:
        return self._for(level)

    def get_mistakes_for_level(self, level: str) -> list[Any]:
        return self._bank.get_mistakes_for_level(level)

    def format_skills_as_prompt(self, *args: Any, **kwargs: Any) -> str:
        return self._bank.format_skills_as_prompt(*args, **kwargs)

    @property
    def general_skills(self) -> list[Any]:
        return []

    @property
    def level_skills(self) -> dict[str, list[Any]]:
        if self._level is not None:
            return {self._level: list(self._skills)}
        grouped: dict[str, list[Any]] = {}
        for skill in self._skills:
            grouped.setdefault(getattr(skill, "source_level", ""), []).append(skill)
        return grouped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bank, name)


class _TieredBank:
    """Serve a ``task:type`` retrieval key as *type-specific + cross-type* skills.

    §3 maps HeXA's ``general -> level`` hierarchy onto ``task -> type``: the bare task
    key (``"tower-of-hanoi"``) is the cross-type tier, and ``"tower-of-hanoi:5"`` is
    the type-specific one. Nothing was joining them.

    That gap was fatal and silent. `evolve_skill_bank` is called with the bare task
    name, so every distilled skill lands under ``level_skills["tower-of-hanoi"]``,
    while retrieval under ``hierarchy="task:type"`` looked only in
    ``level_skills["tower-of-hanoi:5"]`` -- empty -- with ``general_skills`` empty too
    (`skip_general` for single-level runs). **No skill ever reached a prompt.** The
    Stage 3a smoke pilot proved it: round 1's prompt was byte-identical in length to
    round 0's no-memory prompt with a 9-skill bank in hand.

    It could not be caught by the existing guards. ``Injection.starved`` fires when the
    bank offered candidates and the budget dropped them all; here the bank offered
    nothing *at the level asked for*, which is indistinguishable from an empty bank.
    A full pilot would have reported no-memory accuracy under a with-memory label and
    read as "the method does not help".

    Skills are ordered specific-tier first so the top-N cut prefers the more
    particular ones, and deduplicated by ``skill_id`` in case a skill is filed in both.
    """

    def __init__(self, bank: Any, level: str, general_level: str) -> None:
        self._bank = bank
        self._level = level
        self._general_level = general_level

    def _merged(self) -> list[Any]:
        seen: set[Any] = set()
        merged: list[Any] = []
        for tier in (self._level, self._general_level):
            for skill in self._bank.level_skills.get(tier, []) or []:
                key = getattr(skill, "skill_id", None) or id(skill)
                if key not in seen:
                    seen.add(key)
                    merged.append(skill)
        return merged

    @property
    def level_skills(self) -> dict[str, list[Any]]:
        combined = dict(self._bank.level_skills)
        combined[self._level] = self._merged()
        return combined

    def get_skills_for_level(self, level: str) -> list[Any]:
        if level != self._level:
            return self._bank.get_skills_for_level(level)
        return self._merged()

    def get_mistakes_for_level(self, level: str) -> list[Any]:
        if level != self._level:
            return self._bank.get_mistakes_for_level(level)
        seen: set[Any] = set()
        merged: list[Any] = []
        for tier in (self._level, self._general_level):
            for mistake in self._bank.get_mistakes_for_level(tier) or []:
                key = getattr(mistake, "mistake_id", None) or id(mistake)
                if key not in seen:
                    seen.add(key)
                    merged.append(mistake)
        return merged

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bank, name)


def tiered_bank(bank: Any, level: str, task_name: str) -> Any:
    """Wrap *bank* so retrieval at *level* also sees the cross-type ``task_name`` tier.

    Returns *bank* unchanged for flat retrieval (``level == task_name``) or when there
    is no bank, so the common path stays free of the view entirely.
    """
    if bank is None or level == task_name:
        return bank
    return _TieredBank(bank, level, task_name)


def assert_injection_effective(injections: Iterable[Injection]) -> None:
    """Raise if any injection silently degraded to the no-memory prompt.

    Call this on a round's injections before trusting its numbers. The failure it
    catches produces no exception and no wrong-looking output on its own -- only a
    with-memory column that quietly contains no-memory results.
    """
    starved = [inj for inj in injections if inj.starved]
    if not starved:
        return
    first = starved[0]
    raise AssertionError(
        f"{len(starved)} injection(s) degraded to the no-memory prompt: the bank "
        f"offered candidates but none reached the prompt. First: problem "
        f"{first.problem.id!r} -- {len(first.block.skills)} skills and "
        f"{len(first.block.mistakes)} mistakes survived retrieval, "
        f"{first.block.dropped_skills}/{first.block.dropped_mistakes} dropped for "
        f"budget, {first.excluded_parent} excluded as parent. Because CORE's "
        f"templates gate the guidelines slot on truthiness, build_prompt(block) is "
        f"byte-identical to build_prompt(None) here, so this run would report "
        f"no-memory accuracy under a with-memory label. Raise "
        f"--injection-budget-tokens, or lower max_specific so shorter candidates fit."
    )
