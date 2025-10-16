# DSPy Pipeline (NL → QuerySpec)

This directory mirrors the LangChain-based workflow under `NL/`, but replaces the prompt/parse logic with [DSPy](https://github.com/stanfordnlp/dspy).

## Components

- `signatures.py`: DSPy signature classes defining input/output fields.
- `pipeline.py`: Lightweight `dspy.Module` composition that
  - prompts the LM with instructional few-shot context,
  - parses output into the shared `QuerySpec` pydantic model,
  - reuses the existing semantic checker (auto-registers new predicates).
- `tests/` (TODO): We should port/extend the regression cases from `NL/test_fixed_system.py` once we finalize evaluation data.

## Usage

```python
import dspy
from NL_dspy import configure_lm, build_pipeline

configure_lm("openai/gpt-4o-mini", api_key="...", temperature=0.2)
pipeline = build_pipeline()

nl = "Find cases where a car approaches a pedestrian within 15m, comes to a stop, then drives away."
spec = pipeline(nl_request=nl)
```

The returned object is an instance of `NL.specs.QuerySpec`, ready for compilation by `NL/compiler.py`.

## TODOs / Gaps

- **Validation set**: the existing LangChain flow relies on manual scripts; we should curate a held-out NL → spec dataset for regression.
- **Structured evaluation**: no automatic JSON validation loop yet—currently errors bubble up as exceptions.
- **UDF alignment**: still depends on `GLOBAL_UDF_REGISTRY`; long-term we may consolidate registry definitions before runtime.  only knows the predicate names; the actual executable UDFs still live in the per-run UDFRegistry(df) that main.py / QueryCompiler instantiate once they have a dataframe.

