# KeyframeQL Transformer

Turn end-user JSON annotations into an executable KeyframeQL query using an LLM.
Ships with a clean prompt pipeline, few-shot examples, a concise "query dsl" of the KeyframeQL DSL, and pluggable providers (OpenAI GPT-5 and Google Gemini).


## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# Core deps (choose the providers you plan to use)
pip install openai google-generativeai
```

If your environment already has the providers’ SDKs, you can skip installing both.

## Authentication
```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Google Gemini
export GEMINI_API_KEY=...
```

## Quick start (CLI)

Put your annotations in a file, e.g. examples/example_annotations.json.

Run the transformer:
```bash
python -m main \
  --input input/<input>.json \
  --provider openai \
  --gpt-model gpt-5 \
  --temperature 0.2
```

Switch to Gemini:
```bash
python -m main \
  --input input/<input>.json \
  --provider gemini \
  --gpt-model gemini-1.5-pro
```

Useful flags:
```bash
--no-examples       # drop few-shot to test generalization
--no-query-dsl      # drop DSL query DSL (not recommended)
--fast-vel 3.0      # domain-specific default parameters; change "driving fast" threshold for velocity_above()
``` 


## Input JSON format

Your annotations should contain:
- video_id: string
- keyframes: list of keyframes:
    - id, timestamp, duration
    - objects: list:
        - label (e.g., "car1", "car2")
        - constraints: list of NL strings (e.g., "driving fast", "moving in a perpendicular way to car2")
        - type, coordinates (optional; used by your runtime)
    - constraints.frame_level: list of NL strings (e.g., "at the intersection")
- inter_frame_constraints: list:
    - from_keyframe_id, to_keyframe_id
    - constraint_type (e.g., "car1 taking a turn left", "car2 keep going straight"
    - from_timestamp, to_timestamp

The pipeline does not enforce a full schema; it passes JSON through validation to catch gross errors and then prompts the LLM with mapping rules.

## Output format

The pipeline returns only a Python code block implementing a KeyframeQL query:
- Object proxies: car1 = Obj('car', idx=0), car2 = Obj('car', idx=1), …
- Keyframes with .where(...) predicates.
- Temporal constraints: .always(...), .interframe(...), etc.
- .build(objects={'car': 2}) as a minimal object spec (adjust as needed).

You paste the code into your KeyframeQL environment and execute it.

## Prompting

The prompting pipeline constructs a single string prompt composed of:

```bash
You are generating a Python KeyframeQL query that uses this Query DSL.

# === KeyframeQL Query DSL ===
<...>
# === EXAMPLE: JSON annotations 
# === EXAMPLE: Target Query ===
<...>
# === USER COMMAND === 
“Transform the JSON annotations into a KeyframeQL query.”
# === INPUT JSON ANNOTATIONS === 
<input/Euro_NCAP-CCFtap_normal.json>
# === REQUIRED OUTPUT FORMAT === : 
... 
```

### Command

A concise instruction, default:

> “Transform the JSON annotations into a KeyframeQL query.”

(Override via TransformConfig.command.)

### Query DSL

A compact, high-signal summary of the KeyframeQL DSL and UDFs the model should use (e.g., heading_diff_to, velocity_above, interframe(..., comparators=[...])). It also includes mapping rules and output constraints (single Python fence).

#### Mapping rules

These rules are embedded in the query DSL and applied by the model:

| Natural phrase                             | Query expression                                                                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| “opposite directions”                      | `heading_diff_to(other, 180.0)`                                                                                                                   |
| “perpendicular” / “in a perpendicular way” | `heading_diff_to(other, 90.0)`                                                                                                                    |
| “keep going straight” (inter-frame)        | `heading_diff(5.0)`  *(\~0° change within ±5°)*                                                                                                   |
| “turn left” / “turn right” (inter-frame)   | `heading_diff(90.0)` *(magnitude only; sign ignored in current UDF)*                                                                              |
| “driving fast”                             | `velocity_above(FAST_VEL)` (default `FAST_VEL = 2.0`)                                                                                             |
| “at the intersection”                      | By default: comment. If you provide a UDF (e.g., `maps.is_intersection(...)`) or ROI (`within(BBox(...))`), we can wire it into the prompt/rules. |

Time shift: computed as the difference between to_timestamp and from_timestamp from the first inter-frame constraint pair. (You can change the strategy—e.g., median over all pairs.)

### Examples (few-shot)

A paired example of JSON → Query to ground the model’s formatting and conventions (you can extend/rotate these).

### Provider call

We send the built prompt to your chosen provider and extract the fenced Python block.
