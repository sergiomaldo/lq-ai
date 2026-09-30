"""Unit tests for fork decision S1: ``refuse_values`` / ``refusal_template``.

Covers frontmatter parsing in both input locations (top-level ``inputs``
and ``lq_ai.inputs``) and the pure helpers in :mod:`app.skills.policy`.
No database; the send-path behaviour is in
``tests/integration/test_skill_input_refusal_send.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from app.skills.policy import (
    FALLBACK_REFUSAL,
    RefusedInput,
    find_refused_input,
    missing_required_inputs,
    render_refusal,
)
from app.skills.schema import (
    SkillFrontmatter,
    SkillInputs,
    extract_inputs,
    extract_refusal_template,
)

TOP_LEVEL = """
name: dpa-scope
description: Test skill.
refusal_template: "{{regime}} is outside this skill's scope."
inputs:
  required:
    - name: regime
      type: enum
      enum: [gdpr, ccpa, other]
      refuse_values: [other]
    - name: document
      type: document
  optional:
    - name: tone
      type: enum
      enum: [formal, plain, no]
      refuse_values: [no]
"""

NESTED = """
name: dpa-scope
description: Test skill.
lq_ai:
  refusal_template: Nested refusal for {{regime}}.
  inputs:
    required:
      - name: regime
        type: enum
        enum: [gdpr, ccpa, other]
        refuse_values: [other]
"""


def _fm(text: str) -> SkillFrontmatter:
    return SkillFrontmatter.model_validate(yaml.safe_load(text))


@pytest.mark.parametrize("source", [TOP_LEVEL, NESTED], ids=["top-level", "lq_ai"])
def test_refuse_values_parsed_from_both_locations(source: str) -> None:
    inputs = extract_inputs("dpa-scope", _fm(source))
    regime = inputs.required[0]
    assert regime.name == "regime"
    assert regime.enum == ["gdpr", "ccpa", "other"]
    assert regime.refuse_values == ["other"]


def test_refuse_values_absent_is_none() -> None:
    inputs = extract_inputs("dpa-scope", _fm(TOP_LEVEL))
    assert inputs.required[1].name == "document"
    assert inputs.required[1].refuse_values is None


def test_refuse_values_yaml_scalars_compared_as_text() -> None:
    # YAML reads the bare ``no`` as False; the value must still match "False".
    inputs = extract_inputs("dpa-scope", _fm(TOP_LEVEL))
    assert inputs.optional[0].refuse_values == ["False"]


def test_refusal_template_top_level() -> None:
    assert extract_refusal_template(_fm(TOP_LEVEL)) == "{{regime}} is outside this skill's scope."


def test_refusal_template_nested() -> None:
    assert extract_refusal_template(_fm(NESTED)) == "Nested refusal for {{regime}}."


def test_refusal_template_top_level_wins_over_nested() -> None:
    data: dict[str, Any] = yaml.safe_load(NESTED)
    data["refusal_template"] = "Top."
    assert extract_refusal_template(SkillFrontmatter.model_validate(data)) == "Top."


def test_refusal_template_absent_or_blank_is_none() -> None:
    assert extract_refusal_template(_fm("name: x\ndescription: y\n")) is None
    assert (
        extract_refusal_template(_fm("name: x\ndescription: y\nrefusal_template: '  '\n")) is None
    )


# --- policy helpers ---------------------------------------------------------


def _inputs() -> SkillInputs:
    return extract_inputs("dpa-scope", _fm(TOP_LEVEL))


def test_missing_only_policy_inputs_when_not_enforcing_all() -> None:
    assert missing_required_inputs(_inputs(), {}, enforce_all=False) == ["regime"]


def test_missing_all_required_when_enforcing_all() -> None:
    assert missing_required_inputs(_inputs(), {}, enforce_all=True) == ["regime", "document"]


@pytest.mark.parametrize("empty", [None, "", "   ", [], {}])
def test_empty_values_count_as_missing(empty: Any) -> None:
    assert missing_required_inputs(_inputs(), {"regime": empty}, enforce_all=False) == ["regime"]


def test_bound_required_inputs_are_not_missing() -> None:
    bindings = {"regime": "gdpr", "document": "text"}
    assert missing_required_inputs(_inputs(), bindings, enforce_all=True) == []


def test_refused_value_found() -> None:
    refused = find_refused_input("dpa-scope", _inputs(), {"regime": "other"})
    assert refused == RefusedInput(skill="dpa-scope", input="regime", value="other")


def test_in_scope_value_not_refused() -> None:
    assert find_refused_input("dpa-scope", _inputs(), {"regime": "gdpr"}) is None


def test_refused_value_matched_in_list_binding() -> None:
    refused = find_refused_input("dpa-scope", _inputs(), {"regime": ["gdpr", "other"]})
    assert refused is not None and refused.value == "other"


def test_refused_optional_input() -> None:
    refused = find_refused_input("dpa-scope", _inputs(), {"regime": "gdpr", "tone": False})
    assert refused == RefusedInput(skill="dpa-scope", input="tone", value="False")


def test_render_refusal_substitutes_bindings_and_keeps_unknown() -> None:
    refused = RefusedInput(skill="dpa-scope", input="regime", value="other")
    text = render_refusal(
        "{{regime}} not handled; see {{ counsel }}.", {"regime": "other"}, refused
    )
    assert text == "other not handled; see {{ counsel }}."


def test_render_refusal_fallback_when_no_template() -> None:
    refused = RefusedInput(skill="dpa-scope", input="regime", value="other")
    text = render_refusal(None, {"regime": "other"}, refused)
    assert text == FALLBACK_REFUSAL.format(skill="dpa-scope", input="regime", value="other")
    assert "No model was run." in text
