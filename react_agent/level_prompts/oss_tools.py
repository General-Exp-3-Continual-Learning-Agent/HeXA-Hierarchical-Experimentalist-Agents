"""
OSS tool schemas (OpenAI function-calling format) for each level.
"""

from .shared import _XYR_PARAMS, _EMPTY_PARAMS, OSS_TOOLS_SHARED

# ─── OSS tool schemas (OpenAI function-calling format) ──────

OSS_TOOL_GAP_ANALYSIS = {"type": "function", "function": {
    "name": "compute_gap_analysis",
    "description": "Analyze gaps on each side of the platform. Returns whether green ball can fit through each gap.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_RELATIVE_POS = {"type": "function", "function": {
    "name": "compute_relative_positions",
    "description": "Analyze positions of green and blue balls. Returns distance, relative side, and recommended placement direction.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_CATAPULT_SCENE = {"type": "function", "function": {
    "name": "describe_scene_geometry",
    "description": "Strategy-neutral scene geometry: every ball (pos, radius, dynamic, mass), every bar as endpoints + angle + thickness, every basket with floor center, rim corners, opening direction, angle; plus a few useful pairwise distances. No prescriptive advice.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_CATAPULT_TRACE = {"type": "function", "function": {
    "name": "trace_green_ball",
    "description": "Place a red ball and simulate, then return sampled waypoints and kinematic summary for YOUR chosen objects. Use object_names=[\"green_ball\"] to see an arc, [\"basket\",\"blue_ball\"] to see if the basket was disturbed, [\"catapult_arm\",\"green_ball\"] to see lever behavior, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "radius": {"type": "number"},
            "object_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of objects to trace (e.g. ['green_ball','catapult_arm'])."
            },
            "n_samples": {"type": "integer", "description": "Number of waypoints per object (default 12, max 20)."},
            "stop_step": {"type": "integer", "description": "Optional step cap; default = run to completion."},
        },
        "required": ["x", "y", "radius", "object_names"],
    }}}

OSS_TOOL_CATAPULT_PREDICT = {"type": "function", "function": {
    "name": "predict_first_contact",
    "description": "Cheap pre-sim (<=90 steps): returns the first object your red ball touches, the step, approach speed, contact point, and surface normal. Use BEFORE a full simulate_with_trace to confirm the red ball hits the intended object.",
    "parameters": _XYR_PARAMS}}

# Back-compat alias so any external caller that imports the old constant keeps
# working (though its target is now the new scene tool, not the legacy one).
OSS_TOOL_CATAPULT_ANALYSIS = OSS_TOOL_CATAPULT_SCENE

OSS_TOOL_FALLING_ANALYSIS = {"type": "function", "function": {
    "name": "compute_intercept_setup",
    "description": "Computes intercept geometry: which platform green is on, direction it must travel, platform edge to cross, gap center, and estimated jar fall time to platform height.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_BASKET_ANALYSIS = {"type": "function", "function": {
    "name": "compute_basket_analysis",
    "description": "Analyze the basket setup: green ball position, basket position and scale, and recommended direction to push the green ball to avoid the basket.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_PASS_THE_PARCEL_ANALYSIS = {"type": "function", "function": {
    "name": "get_ramp_center",
    "description": "Analyze the pass_the_parcel setup: top basket position (inverted, trapping green ball), bottom basket position (holding blue ball), ramp angle and bounds, platform position, and recommended red ball placement.",
    "parameters": _EMPTY_PARAMS}}

OSS_TOOL_TIPPING_POINT_ANALYSIS = {"type": "function", "function": {
    "name": "compute_tipping_point_analysis",
    "description": "Analyze tipping_point geometry. Returns the green bar's centre, length, angle, top/bottom endpoints; the basket's centre and floor; the purple wall's x and top/bottom; whether the wall is on the LEFT or RIGHT of the bar; the horizontal distance from the bar's centre to the wall; and the suggested tip direction (LEFT or RIGHT).",
    "parameters": _EMPTY_PARAMS}}


def get_oss_tools(level_name: str = "down_to_earth", include_level_tool: bool = True) -> list:
    """Return the list of tool schemas for the OSS model's chat template.
    When include_level_tool is False (ablation), the level-specific
    analysis tool is omitted.
    """
    tools = list(OSS_TOOLS_SHARED)
    if include_level_tool:
        if level_name == "down_to_earth":
            tools.append(OSS_TOOL_GAP_ANALYSIS)
        elif level_name == "two_body_problem":
            tools.append(OSS_TOOL_RELATIVE_POS)
        elif level_name == "catapult":
            tools.append(OSS_TOOL_CATAPULT_SCENE)
            tools.append(OSS_TOOL_CATAPULT_TRACE)
            tools.append(OSS_TOOL_CATAPULT_PREDICT)
        elif level_name == "falling_into_place":
            tools.append(OSS_TOOL_FALLING_ANALYSIS)
        elif level_name == "basket_case":
            tools.append(OSS_TOOL_BASKET_ANALYSIS)
        elif level_name == "pass_the_parcel":
            tools.append(OSS_TOOL_PASS_THE_PARCEL_ANALYSIS)
        elif level_name == "tipping_point":
            tools.append(OSS_TOOL_TIPPING_POINT_ANALYSIS)
    return tools
