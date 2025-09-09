import re

_CODE_FENCE_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_python_block(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: return everything (caller may still format)
    return text.strip()
