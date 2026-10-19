"""Export of the contract schemas to the strict JSON-schema subset (Technical Implementation 3.7):
every field required, ``additionalProperties: false``, closed vocabularies as ``enum``, no ``format``,
``pattern`` or numeric bounds. A contract test checks every exported schema against this subset."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .tasks import TASKS

FORBIDDEN_KEYS = {
    "format",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "multipleOf",
    "default",
    "title",
}


def _expand_vocab(name: str, vocab: dict[str, list[str]]) -> list[str]:
    if name.endswith("+any"):
        return [*vocab[name[:-4]], "any"]
    return vocab[name]


def _walk(node: Any, vocab: dict[str, list[str]]) -> Any:
    if isinstance(node, list):
        return [_walk(x, vocab) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in FORBIDDEN_KEYS and not isinstance(v, dict):
            continue
        out[k] = _walk(v, vocab)
    if "x-vocab" in out:
        out = {"type": "string", "enum": _expand_vocab(out.pop("x-vocab"), vocab)}
    if "x-vocab-items" in out:
        name = out.pop("x-vocab-items")
        out["items"] = {"type": "string", "enum": _expand_vocab(name, vocab)}
    if out.get("type") == "object" and "properties" in out:
        out["required"] = list(out["properties"].keys())
        out["additionalProperties"] = False
    return out


def export_schema(model: type[BaseModel], vocab: dict[str, list[str]]) -> dict[str, Any]:
    raw = model.model_json_schema()
    schema = _walk(raw, vocab)
    # inline $defs refs are kept (OpenAI strict supports $ref/$defs)
    return schema


def export_all(vocab: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, spec in TASKS.items():
        out[f"{name}.input"] = export_schema(spec.input_model, vocab)
        out[f"{name}.output"] = export_schema(spec.output_model, vocab)
    return out


def check_strict_subset(schema: Any, path: str = "$") -> list[str]:
    """Return violations of the strict structured-output subset (empty list = conforming)."""
    errs: list[str] = []
    if isinstance(schema, list):
        for i, x in enumerate(schema):
            errs += check_strict_subset(x, f"{path}[{i}]")
        return errs
    if not isinstance(schema, dict):
        return errs
    for k in schema:
        if k in FORBIDDEN_KEYS and k not in ("title", "default"):
            errs.append(f"{path}: forbidden keyword '{k}'")
        if k in ("default",):
            errs.append(f"{path}: 'default' is not allowed (every field must be required)")
    if schema.get("type") == "object" and "properties" in schema:
        props = set(schema["properties"])
        if set(schema.get("required", [])) != props:
            errs.append(f"{path}: not every property is required")
        if schema.get("additionalProperties") is not False:
            errs.append(f"{path}: additionalProperties must be false")
    for k, v in schema.items():
        if k in ("properties", "$defs"):
            for pk, pv in v.items():
                errs += check_strict_subset(pv, f"{path}.{k}.{pk}")
        elif isinstance(v, (dict, list)):
            errs += check_strict_subset(v, f"{path}.{k}")
    return errs


def check_vocab(obj: BaseModel, vocab: dict[str, list[str]], path: str = "") -> list[str]:
    """Validate vocabulary-bound fields of a parsed output (Pydantic checks run after parsing)."""
    errs: list[str] = []
    for fname, finfo in type(obj).model_fields.items():
        val = getattr(obj, fname)
        extra = finfo.json_schema_extra or {}
        if isinstance(extra, dict):
            if "x-vocab" in extra:
                allowed = _expand_vocab(extra["x-vocab"], vocab)
                if val not in allowed:
                    errs.append(f"{path}{fname}: '{val}' not in vocabulary {extra['x-vocab']}")
            if "x-vocab-items" in extra:
                allowed = _expand_vocab(extra["x-vocab-items"], vocab)
                bad = [v for v in val if v not in allowed]
                if bad:
                    errs.append(f"{path}{fname}: {bad[:3]} not in vocabulary {extra['x-vocab-items']}")
        if isinstance(val, BaseModel):
            errs += check_vocab(val, vocab, f"{path}{fname}.")
        elif isinstance(val, list):
            for i, it in enumerate(val):
                if isinstance(it, BaseModel):
                    errs += check_vocab(it, vocab, f"{path}{fname}[{i}].")
    return errs
