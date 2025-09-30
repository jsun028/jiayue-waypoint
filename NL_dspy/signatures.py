"""DSPy signature definitions for the NL → query-spec pipeline."""

import dspy


class NLToSpecSignature(dspy.Signature):
    """Translate a natural language description into a structured query spec JSON."""

    nl_request = dspy.InputField("Natural language description of the target scenario.")
    available_udfs = dspy.InputField(
        "Comma-separated list of predicate names available in the execution engine."
    )
    spec_json = dspy.OutputField(
        "JSON string strictly matching the query spec schema (no commentary)."
    )


class SemanticValidationSignature(dspy.Signature):
    """Validate or enrich a structured spec, returning a corrected JSON string if needed."""

    spec_json = dspy.InputField("Candidate JSON spec produced by the generator.")
    available_udfs = dspy.InputField(
        "Comma-separated list of predicate names available in the execution engine."
    )
    validated_json = dspy.OutputField(
        "Semantically valid JSON spec (may match input if already valid)."
    )


