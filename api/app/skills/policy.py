"""Declared-value refusal and required-input checks for skill inputs.

Pure functions over :class:`~app.skills.schema.SkillInputs`; no I/O. The
chat send path (``app.api.chats.send_message``) resolves each attached
catalogue skill's declared inputs, then:

* :func:`missing_required_inputs` names the required inputs the caller
  left missing or empty (the send is refused with 422
  ``skill_input_missing``);
* :func:`find_refused_input` finds the first bound value listed in an
  input's ``refuse_values`` (the send is answered with
  :func:`render_refusal` and the model is never called).

Which required inputs are enforced is a policy choice made by the
caller: an input that declares ``refuse_values`` is always enforced
(omitting it must not bypass the refusal); the rest only when the
deployment turns full enforcement on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from app.skills.schema import SkillInputDef, SkillInputs

# Same placeholder grammar as the gateway's assembler (ADR 0006), so a
# refusal template reads like a skill body.
_VARIABLE_RE: Final[re.Pattern[str]] = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

FALLBACK_REFUSAL: Final[str] = (
    "This request is outside the scope of the skill {skill}: the input "
    "{input} was given the value {value}, which the skill does not handle. "
    "No model was run."
)


@dataclass(frozen=True)
class RefusedInput:
    """The input whose bound value the skill declares out of scope."""

    skill: str
    input: str
    value: str


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def missing_required_inputs(
    inputs: SkillInputs, bindings: dict[str, Any], *, enforce_all: bool
) -> list[str]:
    """Return the names of required inputs that are missing or empty.

    With ``enforce_all=False`` only required inputs that declare
    ``refuse_values`` are checked.
    """

    missing: list[str] = []
    for spec in inputs.required:
        if not enforce_all and not spec.refuse_values:
            continue
        if _is_empty(bindings.get(spec.name)):
            missing.append(spec.name)
    return missing


def _bound_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    if value is None:
        return []
    return [str(value)]


def find_refused_input(
    skill: str, inputs: SkillInputs, bindings: dict[str, Any]
) -> RefusedInput | None:
    """Return the first bound value that an input lists in ``refuse_values``.

    Matching is exact on the text form of the value; a list binding is
    refused when any of its items is listed.
    """

    specs: list[SkillInputDef] = [*inputs.required, *inputs.optional]
    for spec in specs:
        if not spec.refuse_values or spec.name not in bindings:
            continue
        refused = set(spec.refuse_values)
        for value in _bound_values(bindings[spec.name]):
            if value in refused:
                return RefusedInput(skill=skill, input=spec.name, value=value)
    return None


def render_refusal(template: str | None, bindings: dict[str, Any], refused: RefusedInput) -> str:
    """Render the skill's refusal text, or the fixed fallback.

    Placeholders bound from the declared inputs are substituted; unknown
    placeholders are left in place, as the gateway does for skill bodies.
    """

    if template is None:
        return FALLBACK_REFUSAL.format(
            skill=refused.skill, input=refused.input, value=refused.value
        )

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in bindings:
            return match.group(0)
        value = bindings[name]
        return "" if value is None else str(value)

    return _VARIABLE_RE.sub(_replace, template)


__all__ = [
    "FALLBACK_REFUSAL",
    "RefusedInput",
    "find_refused_input",
    "missing_required_inputs",
    "render_refusal",
]
