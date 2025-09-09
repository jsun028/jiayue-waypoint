import json

def ensure_json_str(doc: str | dict) -> str:
    if isinstance(doc, str):
        # sanity parse to catch obvious errors but return original string
        json.loads(doc)
        return doc
    return json.dumps(doc, ensure_ascii=False, indent=2)
