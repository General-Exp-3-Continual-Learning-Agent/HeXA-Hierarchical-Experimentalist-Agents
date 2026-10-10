"""
level_prompts package — split from react_agent/prompts.py.

Each level has its own submodule. This __init__ re-exports everything so that:
    from react_agent.level_prompts import X
works for every X that was previously importable via:
    from react_agent.prompts import X
"""

# ─── shared ────────────────────────────────────────────────
from .shared import (
    REACT_FORMAT_INSTRUCTIONS,
    SHARED_TOOLS,
    FINISH_TOOL,
    FINISH_AS_5,
    OSS_FORMAT_ADDENDUM,
    OSS_RETRY_NUDGE,
    _XYR_PARAMS,
    _EMPTY_PARAMS,
    OSS_TOOLS_SHARED,
    format_observation,
)

# ─── down_to_earth ─────────────────────────────────────────
from .down_to_earth import (
    DOWN_TO_EARTH_ANALYSIS_TOOL,
    DOWN_TO_EARTH_TOOL_DESCRIPTIONS,
    DOWN_TO_EARTH_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    DOWN_TO_EARTH_SYSTEM,
    DOWN_TO_EARTH_INITIAL,
    OSS_DOWN_TO_EARTH_SYSTEM,
    OSS_DOWN_TO_EARTH_INITIAL,
    _OSS_DTE_STRATEGY_BLOCK,
    _OSS_DTE_PHYSICS_LAST_TWO,
    _DTE_STRATEGY_BLOCK,
)

# ─── two_body_problem ──────────────────────────────────────
from .two_body_problem import (
    TWO_BODY_ANALYSIS_TOOL,
    TWO_BODY_TOOL_DESCRIPTIONS,
    TWO_BODY_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    TWO_BODY_SYSTEM,
    TWO_BODY_INITIAL,
    OSS_TWO_BODY_SYSTEM,
    OSS_TWO_BODY_INITIAL,
    _OSS_TBP_HOW_TO_PUSH_BLOCK,
    _OSS_TBP_PRACTICAL_FULL,
    _OSS_TBP_PRACTICAL_GEOMETRY_ONLY,
    _OSS_TBP_PRACTICAL_RADIUS_ONLY,
    _OSS_TBP_CORE_FACT_BLOCK,
    _OSS_TBP_CRITICAL_COLLISION_BLOCK,
    _OSS_TBP_STRATEGY_BLOCK,
)

# ─── catapult ──────────────────────────────────────────────
from .catapult import (
    CATAPULT_SCENE_TOOL,
    CATAPULT_TRACE_TOOL,
    CATAPULT_PREDICT_TOOL,
    CATAPULT_TOOL_DESCRIPTIONS,
    CATAPULT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    CATAPULT_SYSTEM,
    CATAPULT_INITIAL,
    OSS_CATAPULT_SYSTEM,
    OSS_CATAPULT_INITIAL,
)

# ─── falling_into_place ────────────────────────────────────
from .falling_into_place import (
    FALLING_INTERCEPT_TOOL,
    FALLING_TOOL_DESCRIPTIONS,
    FALLING_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    FALLING_INTO_PLACE_SYSTEM,
    FALLING_INTO_PLACE_INITIAL,
    OSS_FALLING_INTO_PLACE_SYSTEM,
    OSS_FALLING_INTO_PLACE_INITIAL,
    _OSS_FIP_HOW_TO_PLACE_BLOCK,
)

# ─── pass_the_parcel ───────────────────────────────────────
from .pass_the_parcel import (
    PASS_THE_PARCEL_ANALYSIS_TOOL,
    PASS_THE_PARCEL_TOOL_DESCRIPTIONS,
    PASS_THE_PARCEL_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    PASS_THE_PARCEL_SYSTEM,
    PASS_THE_PARCEL_INITIAL,
    OSS_PASS_THE_PARCEL_SYSTEM,
    OSS_PASS_THE_PARCEL_INITIAL,
)

# ─── basket_case ───────────────────────────────────────────
from .basket_case import (
    BASKET_CASE_ANALYSIS_TOOL,
    BASKET_CASE_TOOL_DESCRIPTIONS,
    BASKET_CASE_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    BASKET_CASE_SYSTEM,
    BASKET_CASE_INITIAL,
    OSS_BASKET_CASE_SYSTEM,
    OSS_BASKET_CASE_INITIAL,
)

# ─── cliffhanger ───────────────────────────────────────────
from .cliffhanger import (
    CLIFFHANGER_TOOL_DESCRIPTIONS,
    CLIFFHANGER_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    CLIFFHANGER_SYSTEM,
    CLIFFHANGER_INITIAL,
    OSS_CLIFFHANGER_SYSTEM,
    OSS_CLIFFHANGER_INITIAL,
)

# ─── tipping_point ─────────────────────────────────────────
from .tipping_point import (
    TIPPING_POINT_ANALYSIS_TOOL,
    TIPPING_POINT_TOOL_DESCRIPTIONS,
    TIPPING_POINT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL,
    TIPPING_POINT_SYSTEM,
    TIPPING_POINT_INITIAL,
    OSS_TIPPING_POINT_SYSTEM,
    OSS_TIPPING_POINT_INITIAL,
)

# ─── oss_tools ─────────────────────────────────────────────
from .oss_tools import (
    OSS_TOOL_GAP_ANALYSIS,
    OSS_TOOL_RELATIVE_POS,
    OSS_TOOL_CATAPULT_SCENE,
    OSS_TOOL_CATAPULT_TRACE,
    OSS_TOOL_CATAPULT_PREDICT,
    OSS_TOOL_CATAPULT_ANALYSIS,
    OSS_TOOL_FALLING_ANALYSIS,
    OSS_TOOL_BASKET_ANALYSIS,
    OSS_TOOL_PASS_THE_PARCEL_ANALYSIS,
    OSS_TOOL_TIPPING_POINT_ANALYSIS,
    get_oss_tools,
)


# ─── Dispatch functions ─────────────────────────────────────

def build_system_prompt(level_name: str = "down_to_earth", is_oss: bool = False) -> str:
    """Build the full system prompt for the given level."""
    if is_oss:
        if level_name == "two_body_problem":
            return OSS_TWO_BODY_SYSTEM
        elif level_name == "catapult":
            return OSS_CATAPULT_SYSTEM
        elif level_name == "falling_into_place":
            return OSS_FALLING_INTO_PLACE_SYSTEM
        elif level_name == "basket_case":
            return OSS_BASKET_CASE_SYSTEM
        elif level_name == "pass_the_parcel":
            return OSS_PASS_THE_PARCEL_SYSTEM
        elif level_name == "cliffhanger":
            return OSS_CLIFFHANGER_SYSTEM
        elif level_name == "tipping_point":
            return OSS_TIPPING_POINT_SYSTEM
        else:
            return OSS_DOWN_TO_EARTH_SYSTEM
    if level_name == "two_body_problem":
        return TWO_BODY_SYSTEM.format(
            tool_descriptions=TWO_BODY_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "catapult":
        return CATAPULT_SYSTEM.format(
            tool_descriptions=CATAPULT_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "falling_into_place":
        return FALLING_INTO_PLACE_SYSTEM.format(
            tool_descriptions=FALLING_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "basket_case":
        return BASKET_CASE_SYSTEM.format(
            tool_descriptions=BASKET_CASE_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "pass_the_parcel":
        return PASS_THE_PARCEL_SYSTEM.format(
            tool_descriptions=PASS_THE_PARCEL_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "cliffhanger":
        return CLIFFHANGER_SYSTEM.format(
            tool_descriptions=CLIFFHANGER_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    elif level_name == "tipping_point":
        return TIPPING_POINT_SYSTEM.format(
            tool_descriptions=TIPPING_POINT_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )
    else:
        return DOWN_TO_EARTH_SYSTEM.format(
            tool_descriptions=DOWN_TO_EARTH_TOOL_DESCRIPTIONS,
            react_format=REACT_FORMAT_INSTRUCTIONS,
        )


def build_initial_user_message(
    level_name: str = "down_to_earth",
    is_oss: bool = False,
    # NOTE: every non-OSS level's INITIAL message states its own iteration
    # budget in prose ("You have {max_iterations} iterations..."). This used
    # to be a literal number (25, or 12 for pass_the_parcel) with NO connection
    # to the --max-iterations actually enforced by ReactAgent -- so a run
    # configured for e.g. 12 iterations (Reflexion's per-trial cap) told the
    # model it had 25, and the model had no way to know it was about to be cut
    # off. Default stays 25 so any caller that doesn't pass this explicitly
    # gets byte-identical output to before.
    #
    # The OSS variants were originally written with no stated count, but
    # catapult/tipping_point/cliffhanger/two_body_problem later picked up the
    # same "{max_iterations}" placeholder as the non-OSS templates without
    # this branch ever being updated to .format() it in -- every prior
    # gpt-oss-120b run on those 4 levels sent the model the literal text
    # "You have {max_iterations} iterations" verbatim. All 8 OSS templates
    # format safely with this kwarg (verified: the 4 without the placeholder
    # round-trip unchanged), so .format() is applied unconditionally here.
    max_iterations: int = 25,
) -> str:
    """Build the initial user message for the given level."""
    if is_oss:
        if level_name == "two_body_problem":
            return OSS_TWO_BODY_INITIAL.format(max_iterations=max_iterations)
        elif level_name == "catapult":
            return OSS_CATAPULT_INITIAL.format(max_iterations=max_iterations)
        elif level_name == "falling_into_place":
            return OSS_FALLING_INTO_PLACE_INITIAL.format(max_iterations=max_iterations)
        elif level_name == "basket_case":
            return OSS_BASKET_CASE_INITIAL.format(max_iterations=max_iterations)
        elif level_name == "pass_the_parcel":
            return OSS_PASS_THE_PARCEL_INITIAL.format(max_iterations=max_iterations)
        elif level_name == "cliffhanger":
            return OSS_CLIFFHANGER_INITIAL.format(max_iterations=max_iterations)
        elif level_name == "tipping_point":
            return OSS_TIPPING_POINT_INITIAL.format(max_iterations=max_iterations)
        else:
            return OSS_DOWN_TO_EARTH_INITIAL.format(max_iterations=max_iterations)
    if level_name == "two_body_problem":
        return TWO_BODY_INITIAL.format(max_iterations=max_iterations)
    elif level_name == "catapult":
        return CATAPULT_INITIAL.format(max_iterations=max_iterations)
    elif level_name == "falling_into_place":
        return FALLING_INTO_PLACE_INITIAL.format(max_iterations=max_iterations)
    elif level_name == "basket_case":
        return BASKET_CASE_INITIAL.format(max_iterations=max_iterations)
    elif level_name == "pass_the_parcel":
        return PASS_THE_PARCEL_INITIAL.format(max_iterations=max_iterations)
    elif level_name == "cliffhanger":
        return CLIFFHANGER_INITIAL.format(max_iterations=max_iterations)
    elif level_name == "tipping_point":
        return TIPPING_POINT_INITIAL.format(max_iterations=max_iterations)
    else:
        return DOWN_TO_EARTH_INITIAL.format(max_iterations=max_iterations)


__all__ = [
    # shared
    "REACT_FORMAT_INSTRUCTIONS",
    "SHARED_TOOLS",
    "FINISH_TOOL",
    "FINISH_AS_5",
    "OSS_FORMAT_ADDENDUM",
    "OSS_RETRY_NUDGE",
    "_XYR_PARAMS",
    "_EMPTY_PARAMS",
    "OSS_TOOLS_SHARED",
    "format_observation",
    # down_to_earth
    "DOWN_TO_EARTH_ANALYSIS_TOOL",
    "DOWN_TO_EARTH_TOOL_DESCRIPTIONS",
    "DOWN_TO_EARTH_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "DOWN_TO_EARTH_SYSTEM",
    "DOWN_TO_EARTH_INITIAL",
    "OSS_DOWN_TO_EARTH_SYSTEM",
    "OSS_DOWN_TO_EARTH_INITIAL",
    "_OSS_DTE_STRATEGY_BLOCK",
    "_OSS_DTE_PHYSICS_LAST_TWO",
    "_DTE_STRATEGY_BLOCK",
    # two_body_problem
    "TWO_BODY_ANALYSIS_TOOL",
    "TWO_BODY_TOOL_DESCRIPTIONS",
    "TWO_BODY_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "TWO_BODY_SYSTEM",
    "TWO_BODY_INITIAL",
    "OSS_TWO_BODY_SYSTEM",
    "OSS_TWO_BODY_INITIAL",
    "_OSS_TBP_HOW_TO_PUSH_BLOCK",
    "_OSS_TBP_PRACTICAL_FULL",
    "_OSS_TBP_PRACTICAL_GEOMETRY_ONLY",
    "_OSS_TBP_PRACTICAL_RADIUS_ONLY",
    "_OSS_TBP_CORE_FACT_BLOCK",
    "_OSS_TBP_CRITICAL_COLLISION_BLOCK",
    "_OSS_TBP_STRATEGY_BLOCK",
    # catapult
    "CATAPULT_SCENE_TOOL",
    "CATAPULT_TRACE_TOOL",
    "CATAPULT_PREDICT_TOOL",
    "CATAPULT_TOOL_DESCRIPTIONS",
    "CATAPULT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "CATAPULT_SYSTEM",
    "CATAPULT_INITIAL",
    "OSS_CATAPULT_SYSTEM",
    "OSS_CATAPULT_INITIAL",
    # falling_into_place
    "FALLING_INTERCEPT_TOOL",
    "FALLING_TOOL_DESCRIPTIONS",
    "FALLING_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "FALLING_INTO_PLACE_SYSTEM",
    "FALLING_INTO_PLACE_INITIAL",
    "OSS_FALLING_INTO_PLACE_SYSTEM",
    "OSS_FALLING_INTO_PLACE_INITIAL",
    "_OSS_FIP_HOW_TO_PLACE_BLOCK",
    # pass_the_parcel
    "PASS_THE_PARCEL_ANALYSIS_TOOL",
    "PASS_THE_PARCEL_TOOL_DESCRIPTIONS",
    "PASS_THE_PARCEL_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "PASS_THE_PARCEL_SYSTEM",
    "PASS_THE_PARCEL_INITIAL",
    "OSS_PASS_THE_PARCEL_SYSTEM",
    "OSS_PASS_THE_PARCEL_INITIAL",
    # basket_case
    "BASKET_CASE_ANALYSIS_TOOL",
    "BASKET_CASE_TOOL_DESCRIPTIONS",
    "BASKET_CASE_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "BASKET_CASE_SYSTEM",
    "BASKET_CASE_INITIAL",
    "OSS_BASKET_CASE_SYSTEM",
    "OSS_BASKET_CASE_INITIAL",
    # cliffhanger
    "CLIFFHANGER_TOOL_DESCRIPTIONS",
    "CLIFFHANGER_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "CLIFFHANGER_SYSTEM",
    "CLIFFHANGER_INITIAL",
    "OSS_CLIFFHANGER_SYSTEM",
    "OSS_CLIFFHANGER_INITIAL",
    # tipping_point
    "TIPPING_POINT_ANALYSIS_TOOL",
    "TIPPING_POINT_TOOL_DESCRIPTIONS",
    "TIPPING_POINT_TOOL_DESCRIPTIONS_NO_LEVEL_TOOL",
    "TIPPING_POINT_SYSTEM",
    "TIPPING_POINT_INITIAL",
    "OSS_TIPPING_POINT_SYSTEM",
    "OSS_TIPPING_POINT_INITIAL",
    # oss_tools
    "OSS_TOOL_GAP_ANALYSIS",
    "OSS_TOOL_RELATIVE_POS",
    "OSS_TOOL_CATAPULT_SCENE",
    "OSS_TOOL_CATAPULT_TRACE",
    "OSS_TOOL_CATAPULT_PREDICT",
    "OSS_TOOL_CATAPULT_ANALYSIS",
    "OSS_TOOL_FALLING_ANALYSIS",
    "OSS_TOOL_BASKET_ANALYSIS",
    "OSS_TOOL_PASS_THE_PARCEL_ANALYSIS",
    "OSS_TOOL_TIPPING_POINT_ANALYSIS",
    "get_oss_tools",
    # dispatch functions
    "build_system_prompt",
    "build_initial_user_message",
]
