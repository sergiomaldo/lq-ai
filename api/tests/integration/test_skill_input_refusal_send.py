"""Fork decision S1 — skill-input policy on ``POST /chats/{id}/messages``.

* A required input that declares ``refuse_values`` and is missing → 422
  ``skill_input_missing``; nothing persisted, gateway not called.
* Other required inputs are enforced only with
  ``LQ_AI_ENFORCE_REQUIRED_SKILL_INPUTS`` on.
* A bound value listed in ``refuse_values`` → the skill's refusal text as
  the assistant turn (JSON and SSE), user + assistant rows persisted, one
  ``chat.skill_input_refused`` audit row, gateway never called.
* An in-scope value → the normal gateway path, unchanged.

The skill registry is built from a temporary folder so the fixtures do
not leak into other suites; the gateway is mocked with respx.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
import respx
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import app.api.chats as chats_module
from app.clients.gateway import GatewayClient, set_gateway_client
from app.config import get_settings
from app.db.session import get_db
from app.main import app
from app.models.audit import AuditLog
from app.models.chat import Message
from app.models.user import User
from app.security import create_access_token, hash_password
from app.skills import load_registry
from app.skills.registry import MutableSkillRegistry

GATEWAY_BASE = "http://test-gateway"
GATEWAY_KEY = "test-gw-key"

SCOPED_SKILL = """---
name: scoped-review
description: Test skill with an out-of-scope regime value.
lq_ai:
  refusal_template: "This review does not cover {{regime}} agreements. Ask counsel."
  inputs:
    required:
      - name: regime
        type: enum
        enum: [gdpr, ccpa, other]
        refuse_values: [other]
---

# Scoped review

Review under {{regime}}.
"""

DOC_SKILL = """---
name: doc-review
description: Test skill with a plain required input and no refusal policy.
inputs:
  required:
    - name: document
      type: document
---

# Doc review
"""


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    for name, body in (("scoped-review", SCOPED_SKILL), ("doc-review", DOC_SKILL)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "SKILL.md").write_text(body, encoding="utf-8")
    return tmp_path


@pytest_asyncio.fixture
async def client(db_session: AsyncSession, skills_dir: Path) -> AsyncIterator[AsyncClient]:
    async def _override() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_db] = _override
    prior_holder = getattr(app.state, "skill_registry", None)
    app.state.skill_registry = MutableSkillRegistry(load_registry(skills_dir))

    gw = GatewayClient(base_url=GATEWAY_BASE, gateway_key=GATEWAY_KEY)
    set_gateway_client(gw)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    set_gateway_client(None)
    await gw.aclose()
    if prior_holder is None:
        delattr(app.state, "skill_registry")
    else:
        app.state.skill_registry = prior_holder
    app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def db_user(db_session: AsyncSession) -> User:
    user = User(
        email=f"skill-refusal-{uuid.uuid4().hex[:8]}@example.com",
        display_name="Skill Refusal Test User",
        hashed_password=hash_password("correct-horse-battery-staple"),
        is_admin=False,
        mfa_enabled=False,
        must_change_password=False,
    )
    db_session.add(user)
    await db_session.flush()
    return user


def _h(user: User) -> dict[str, str]:
    token = create_access_token(user.id, user.email, is_admin=user.is_admin)
    return {"Authorization": f"Bearer {token}"}


def _success_payload() -> dict[str, object]:
    return {
        "id": "chatcmpl-skill-refusal",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "claude-sonnet-4-6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "model answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "routed_inference_tier": 3,
        "routed_provider": "anthropic-prod",
        "cost_estimate": 0.0001,
        "lq_ai_applied_skills": ["scoped-review"],
    }


async def _new_chat(client: AsyncClient, headers: dict[str, str]) -> str:
    resp = await client.post("/api/v1/chats", headers=headers, json={"title": "x"})
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


async def _messages(db: AsyncSession, chat_id: str) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.chat_id == uuid.UUID(chat_id))
        .order_by(Message.created_at, Message.role.desc())
    )
    return list((await db.execute(stmt)).scalars().all())


# ---------------------------------------------------------------------------
# (a) missing required input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_missing_policy_input_returns_422_and_persists_nothing(
    client: AsyncClient, db_user: User, db_session: AsyncSession
) -> None:
    route = respx.post(f"{GATEWAY_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_success_payload())
    )
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={"content": "review this", "attached_skills": [{"slug": "scoped-review"}]},
    )

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "skill_input_missing"
    assert detail["details"] == {"skill": "scoped-review", "missing": ["regime"]}
    assert not route.called
    assert await _messages(db_session, chat_id) == []


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_empty_policy_input_returns_422(client: AsyncClient, db_user: User) -> None:
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={
            "content": "review this",
            "skills": ["scoped-review"],
            "skill_inputs": {"scoped-review": {"regime": "  "}},
        },
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["details"]["missing"] == ["regime"]


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_plain_required_input_not_enforced_by_default(
    client: AsyncClient, db_user: User
) -> None:
    route = respx.post(f"{GATEWAY_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_success_payload())
    )
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={"content": "review this", "skills": ["doc-review"]},
    )

    assert resp.status_code == 200, resp.text
    assert route.called


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_plain_required_input_enforced_when_setting_on(
    client: AsyncClient, db_user: User, monkeypatch: pytest.MonkeyPatch
) -> None:
    enforcing = get_settings().model_copy(update={"lq_ai_enforce_required_skill_inputs": True})
    monkeypatch.setattr(chats_module, "get_settings", lambda: enforcing)
    route = respx.post(f"{GATEWAY_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_success_payload())
    )
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={"content": "review this", "skills": ["doc-review"]},
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["details"] == {"skill": "doc-review", "missing": ["document"]}
    assert not route.called


# ---------------------------------------------------------------------------
# (b) declared-value refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_refused_value_answers_without_gateway(
    client: AsyncClient, db_user: User, db_session: AsyncSession
) -> None:
    route = respx.post(f"{GATEWAY_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_success_payload())
    )
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={
            "content": "review this",
            "attached_skills": [
                {"slug": "scoped-review", "inputs": {"regime": "other"}, "source": "picker"}
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    assert not route.called
    body = resp.json()
    expected = "This review does not cover other agreements. Ask counsel."
    assert body["message"]["content"] == expected
    assert body["message"]["role"] == "assistant"
    assert body["message"]["kind"] == "refusal"
    assert body["message"]["prompt_tokens"] is None
    assert body["message"]["cost_estimate"] is None
    assert body["routed_provider"] == "policy"
    assert body["applied_skills"] == ["scoped-review"]
    assert resp.headers["X-LQ-AI-Routed-Provider"] == "policy"

    rows = await _messages(db_session, chat_id)
    assert [(m.role, m.kind) for m in rows] == [("user", "user"), ("assistant", "refusal")]
    assert rows[0].content == "review this"
    assistant = rows[1]
    assert str(assistant.id) == body["message"]["id"]
    assert assistant.content == expected
    assert assistant.applied_skills == ["scoped-review"]
    assert assistant.routed_provider == "policy"
    assert assistant.completion_tokens is None
    assert assistant.cost_estimate_micros is None

    audits = (
        (
            await db_session.execute(
                select(AuditLog).where(AuditLog.action == "chat.skill_input_refused")
            )
        )
        .scalars()
        .all()
    )
    assert len(audits) == 1
    audit = audits[0]
    assert audit.user_id == db_user.id
    assert audit.resource_id == str(assistant.id)
    assert audit.routed_provider == "policy"
    assert audit.details is not None
    assert audit.details["reason"] == "refuse_values"
    assert audit.details["skill"] == "scoped-review"
    assert audit.details["input"] == "regime"
    assert audit.details["value"] == "other"
    assert audit.details["user_message_id"] == str(rows[0].id)


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_refused_value_streaming_takes_same_path(
    client: AsyncClient, db_user: User, db_session: AsyncSession
) -> None:
    route = respx.post(f"{GATEWAY_BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_success_payload())
    )
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={
            "content": "review this",
            "skills": ["scoped-review"],
            "skill_inputs": {"scoped-review": {"regime": "other"}},
            "stream": True,
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert not route.called
    data_lines = [
        line[len("data: ") :] for line in resp.text.split("\n") if line.startswith("data: ")
    ]
    assert data_lines[-1] == "[DONE]"
    frames = [json.loads(line) for line in data_lines[:-1]]
    assert [f["type"] for f in frames] == ["start", "delta", "complete"]
    expected = "This review does not cover other agreements. Ask counsel."
    assert frames[1]["delta"] == expected
    assert frames[2]["message"]["content"] == expected
    assert frames[2]["routed_provider"] == "policy"
    assert frames[2]["applied_skills"] == ["scoped-review"]

    rows = await _messages(db_session, chat_id)
    assert [(m.role, m.kind) for m in rows] == [("user", "user"), ("assistant", "refusal")]


# ---------------------------------------------------------------------------
# normal path unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
@respx.mock
async def test_in_scope_value_calls_gateway_as_before(
    client: AsyncClient, db_user: User, db_session: AsyncSession
) -> None:
    captured: dict[str, object] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_success_payload())

    route = respx.post(f"{GATEWAY_BASE}/v1/chat/completions").mock(side_effect=_capture)
    headers = _h(db_user)
    chat_id = await _new_chat(client, headers)

    resp = await client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=headers,
        json={
            "content": "review this",
            "skills": ["scoped-review"],
            "skill_inputs": {"scoped-review": {"regime": "gdpr"}},
        },
    )

    assert resp.status_code == 200, resp.text
    assert route.called
    fwd = captured["body"]
    assert isinstance(fwd, dict)
    assert fwd["lq_ai_skills"] == ["scoped-review"]
    assert fwd["lq_ai_skill_inputs"] == {"scoped-review": {"regime": "gdpr"}}
    body = resp.json()
    assert body["message"]["content"] == "model answer"
    assert body["message"]["kind"] == "ai"
    assert body["routed_provider"] == "anthropic-prod"

    audits = (
        (
            await db_session.execute(
                select(AuditLog).where(AuditLog.action == "chat.skill_input_refused")
            )
        )
        .scalars()
        .all()
    )
    assert audits == []
