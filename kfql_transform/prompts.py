from textwrap import dedent

# Minimal, high-signal query DSL so the model doesn’t need your entire engine source.
KFQL_QUERY_DSL = dedent("""
You are generating a Python KeyframeQL query that uses this DSL:

# === KeyframeQL query DSL ===

# Objects
obj1 = Obj('<class_name>', idx=0)   # e.g., Obj('car', idx=0), Obj('person', idx=0)
obj2 = Obj('<class_name>', idx=1)   # e.g., Obj('car', idx=1), Obj('bike', idx=0)

# Keyframes
k1 = KF('k1').where(<Predicate> [& <Predicate> ...])
k2 = KF('k2').where(<Predicate> [& <Predicate> ...])

# Predicates (single-frame)
objX.velocity_above(float)         # "driving fast"
objX.velocity_below(float)
objX.heading_diff_to(objY, deg, tol_deg=15.0)  # relative heading in a frame

# Inter-frame comparators (k1 -> k2)
objX.heading_diff(expected_deg, tol_deg=15.0)  # change in heading between k1 and k2
objX.distance_within(max_dist)
objX.distance_apart(min_dist)

# Temporal constraints
Query(k1, k2, ...)
  .always(anchor=None, dur_sec=3.0, target=k1, tol=0.01)
  .always(anchor=None, dur_sec=3.0, target=k2, tol=0.01)
  .interframe(anchor=k1, target=k2, time_shift=<seconds>, comparators=[...])
  .build(objects={'car': 2, 'person': 1})

MAPPING RULES (apply consistently):
- "opposite directions" → heading_diff_to(other, 180.0)
- "perpendicular" / "in a perpendicular way" → heading_diff_to(other, 90.0)
- "keep going straight" (inter-frame) → heading_diff(0 ± 5°) ⇒ heading_diff(5.0)
- "turn left" (inter-frame) → heading_diff(~90°) ⇒ heading_diff(90.0)
- "turn right" (inter-frame) → heading_diff(~90°) ⇒ heading_diff(90.0)  # sign ignored; use magnitude
- "driving fast" → velocity_above({FAST_VEL})
- “at the intersection” → If an explicit UDF/ROI is not provided, keep as a comment.

OUTPUT RULES:
- Output ONLY Python code for the query, wrapped in a single ```python fence.
- Define object proxies (obj1, obj2, …) deterministically following labels in the JSON (stable ordering).
- Create KFs in the order of appearance in the JSON.
- Compute time_shift as (to_timestamp - from_timestamp) using first inter-frame pair.
- Name the query constant meaningfully (e.g., SCENE_<VIDEOID>_QUERY with safe characters).
- Do not include extra commentary outside the code fence.
""").strip()

# You can drop your gold pair (JSON → Query) here. Kept compact but faithful.
EXAMPLE_PAIR = dedent("""
# === EXAMPLE: JSON annotations ===
<EXAMPLE_JSON>
{
  "video_id": "EURO_NCAP_CMCscp_normal_Car-to-Car_Crossing_Straight_Crossing_Path",
  "keyframes": [
    {
      "id": "keyframe_1757423189369",
      "timestamp": 6.584278,
      "duration": 0,
      "objects": [
        {
          "id": "obj_1757423231098",
          "label": "car1",
          "timestamp": 6.584278,
          "constraints": [
            "moving in a perpendicular way to car2",
            "driving fast"
          ],
          "type": "bounding_box",
          "coordinates": {
            "x": 497,
            "y": 315,
            "width": 47,
            "height": 75
          }
        },
        {
          "id": "obj_1757423237177",
          "label": "car2",
          "timestamp": 6.584278,
          "constraints": [
            "driving fast"
          ],
          "type": "bounding_box",
          "coordinates": {
            "x": 244,
            "y": 205,
            "width": 81,
            "height": 38
          }
        }
      ],
      "constraints": {
        "frame_level": [
          "at the intersection"
        ],
        "inter_frame": []
      }
    },
    {
      "id": "keyframe_1757423194507",
      "timestamp": 12.430398,
      "duration": 0,
      "objects": [
        {
          "id": "obj_1757423246088",
          "label": "car1",
          "timestamp": 12.430398,
          "constraints": [
            "moving in a perpendicular way to car2",
            "driving fast"
          ],
          "type": "bounding_box",
          "coordinates": {
            "x": 497,
            "y": 27,
            "width": 29,
            "height": 42
          }
        },
        {
          "id": "obj_1757423251168",
          "label": "car2",
          "timestamp": 12.430398,
          "constraints": [
            "driving fast"
          ],
          "type": "bounding_box",
          "coordinates": {
            "x": 462,
            "y": 204,
            "width": 79,
            "height": 40
          }
        }
      ],
      "constraints": {
        "frame_level": [
          "at the intersection"
        ],
        "inter_frame": []
      }
    }
  ],
  "inter_frame_constraints": [
    {
      "id": "constraint_1757423444909",
      "from_keyframe_id": "keyframe_1757423189369",
      "to_keyframe_id": "keyframe_1757423194507",
      "constraint_type": "car1 keep going straight",
      "from_timestamp": 6.584278,
      "to_timestamp": 12.430398
    },
    {
      "id": "constraint_1757423450988",
      "from_keyframe_id": "keyframe_1757423189369",
      "to_keyframe_id": "keyframe_1757423194507",
      "constraint_type": "car2 keep going straight",
      "from_timestamp": 6.584278,
      "to_timestamp": 12.430398
    }
  ]
}
</EXAMPLE_JSON>

# === EXAMPLE: Target Query ===
<EXAMPLE_QUERY>
# ── object proxy ──────────────────────────────────────────────────────
car1 = Obj('car', idx=0)
car2 = Obj('car', idx=1)

# ── keyframes ----------------------------------------------------------
k1 = KF('k1').where(car1.heading_diff_to(car2, 90.0) & car1.velocity_above(2.0) & car2.velocity_above(2.0))
k2 = KF('k2').where(car1.heading_diff_to(car2, 90.0) & car1.velocity_above(2.0) & car2.velocity_above(2.0))

# ── build query --------------------------------------------------------
EURO_NCAP_CMCscp_normal_QUERY = (
    Query(k1, k2)
      .always(anchor=None, dur_sec=3.0, target=k1, tol=0.01)
      .always(anchor=None, dur_sec=3.0, target=k2, tol=0.01)
      .interframe(
          anchor=k1, target=k2, time_shift=3.0,
          comparators=[car1.heading_diff(5.0), car2.heading_diff(5.0)]
      )
      .build(objects={'car': 2})
)
</EXAMPLE_QUERY>
""").strip()

def build_prompt(
    user_command: str,
    json_annotations: str,
    include_query_dsl: bool = True,
    include_examples: bool = True,
    # domain-specific default parameters
    fast_velocity: float = 2.0,
) -> str:
    parts = []
    if include_query_dsl:
        parts.append(KFQL_QUERY_DSL.replace("{FAST_VEL}", f"{fast_velocity:.1f}"))
    if include_examples:
        parts.append(EXAMPLE_PAIR)
    parts.append("# === USER COMMAND ===\n" + user_command.strip())
    parts.append("# === INPUT JSON ANNOTATIONS ===\n<INPUT_JSON>\n" + json_annotations.strip() + "\n</INPUT_JSON>")
    parts.append("# === REQUIRED OUTPUT FORMAT ===\nReturn ONLY a single Python code block, nothing else.")
    return "\n\n".join(parts)
