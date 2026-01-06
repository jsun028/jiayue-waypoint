import json
from keyframeql.specs import (
    QuerySpec,
    PredicateAtom,
    PredicateExpr,
    KeyframeSpec,
    ObjectsSpec,
    AlwaysSpec,
    InterframeSpec,
    TrajectorySpec,
    ComputationSpec)
from typing import Any, Union

class SpecParser:
    """Parse JSON into `QuerySpec`, raising helpful errors."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[DSPy][Parse] {message}")

    def _check_unknown_fields(
        self, data: Any, model_class: type, path: str = "root"
    ) -> None:
        """Recursively check for unknown fields in nested model structures."""
        if not isinstance(data, dict):
            return

        if not hasattr(model_class, "model_fields"):
            return

        allowed_fields = set(model_class.model_fields.keys())
        unknown_fields = set(data.keys()) - allowed_fields

        if unknown_fields:
            raise ValueError(
                f"Unknown fields in {path}: {sorted(unknown_fields)}. "
                f"Allowed fields are: {sorted(allowed_fields)}"
            )

        # Recursively check nested structures based on model fields
        for field_name, field_info in model_class.model_fields.items():
            if field_name not in data:
                continue

            field_value = data[field_name]
            field_type = field_info.annotation

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Get the non-None type from Optional[Type] = Union[Type, None]
                non_none_types = [
                    t for t in field_type.__args__ if t is not type(None)
                ]
                if non_none_types:
                    field_type = non_none_types[0]
                else:
                    continue

            # Handle List types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                if isinstance(field_value, list):
                    item_type = (
                        field_type.__args__[0]
                        if hasattr(field_type, "__args__") and field_type.__args__
                        else None
                    )
                    # Handle specific list types
                    if field_name == "keyframes":
                        for i, kf in enumerate(field_value):
                            if isinstance(kf, dict):
                                self._check_unknown_fields(
                                    kf, KeyframeSpec, f"{path}.{field_name}[{i}]"
                                )
                    elif field_name == "constraints":
                        for i, constraint in enumerate(field_value):
                            if isinstance(constraint, dict):
                                kind = constraint.get("kind")
                                if kind == "always":
                                    self._check_unknown_fields(
                                        constraint,
                                        AlwaysSpec,
                                        f"{path}.{field_name}[{i}]",
                                    )
                                elif kind == "interframe":
                                    self._check_unknown_fields(
                                        constraint,
                                        InterframeSpec,
                                        f"{path}.{field_name}[{i}]",
                                    )
                                elif kind == "trajectory":
                                    self._check_unknown_fields(
                                        constraint,
                                        TrajectorySpec,
                                        f"{path}.{field_name}[{i}]",
                                    )
                    elif field_name == "args":
                        # Recursive PredicateExpr args
                        for i, arg in enumerate(field_value):
                            if isinstance(arg, dict):
                                self._check_unknown_fields(
                                    arg, PredicateExpr, f"{path}.{field_name}[{i}]"
                                )
                    elif item_type:
                        # Generic list handling
                        for i, item in enumerate(field_value):
                            if isinstance(item, dict):
                                self._check_unknown_fields(
                                    item, item_type, f"{path}.{field_name}[{i}]"
                                )
                continue

            # Handle BaseModel types
            if isinstance(field_value, dict):
                try:
                    # Try to identify the model class based on field name and structure
                    if field_name == "where" and field_value.get("op") is not None:
                        self._check_unknown_fields(
                            field_value, PredicateExpr, f"{path}.{field_name}"
                        )
                    elif field_name == "atom" and field_value.get("type") is not None:
                        self._check_unknown_fields(
                            field_value, PredicateAtom, f"{path}.{field_name}"
                        )
                    elif field_name == "computation" and field_value.get("type") is not None:
                        self._check_unknown_fields(
                            field_value, ComputationSpec, f"{path}.{field_name}"
                        )
                    elif field_name == "objects":
                        self._check_unknown_fields(
                            field_value, ObjectsSpec, f"{path}.{field_name}"
                        )
                except (AttributeError, TypeError):
                    # If we can't determine the type, skip deeper validation
                    pass

    def __call__(self, spec_json: str) -> QuerySpec:
        self._log("Parsing JSON into QuerySpec …")
        try:
            raw = json.loads(spec_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generated spec is not valid JSON: {exc}") from exc

        try:
            # Recursively check for unknown fields at all levels
            if isinstance(raw, dict):
                self._check_unknown_fields(raw, QuerySpec, "QuerySpec")

            spec = QuerySpec.model_validate(raw)
        except ValueError as exc:
            # Re-raise ValueError as-is (our custom errors)
            raise exc
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Generated spec failed schema validation: {exc}") from exc

        self._log("Schema validation succeeded")
        return spec