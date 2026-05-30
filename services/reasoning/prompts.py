"""
Prompt builders for LLM-based surveillance reasoning.
"""


# ---------------------------------------------------------------------------
# Scene Graph Prompt Integration
# ---------------------------------------------------------------------------

from services.reasoning.scene_graph import SceneGraph


from services.reasoning.scene_graph import SceneGraph


def build_reasoning_prompt(*args) -> str:
    """
    Backward-compatible prompt builder.

    Supports:
    1. build_reasoning_prompt(event_description, scene_graph)
    2. build_reasoning_prompt(summary, captions, camera_id, zone_name, dwell_time)
    """

    # New API
    if len(args) == 2:
        event_description, scene_graph = args

        graph_context = scene_graph.to_prompt_str()

        return f"""{graph_context}

Event description:
{event_description}

Based on the scene graph and event above, analyze whether this activity is suspicious.
Consider spatial relationships, zone access, and object interactions.
Be concise and structured in your response.
"""

    # Legacy API used by tests
    if len(args) == 5:
        summary, captions, camera_id, zone_name, dwell_time = args

        return f"""
Summary:
{summary}

Captions:
{captions}

Camera:
{camera_id}

Zone:
{zone_name}

Dwell Time:
{dwell_time}
"""

    raise TypeError(
        f"build_reasoning_prompt expected 2 or 5 arguments, got {len(args)}"
    )