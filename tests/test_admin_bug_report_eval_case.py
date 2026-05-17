from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


def _admin_override():
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    return app, require_admin_jwt


def test_generate_eval_case_from_bug_report_returns_valid_jsonl(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [{
                "id": "bug-report-123456",
                "error_types": ["记忆编造"],
                "reason": "AI 编造了用户喜欢的歌手",
                "message_id": "msg-assistant",
                "message_role": "assistant",
                "assistant_reply": "我记得你最喜欢周兴哲。",
                "created_at": datetime(2026, 5, 17, 10, 0, tzinfo=timezone.utc),
                "conversation_id": "conv-1",
            }],
            [{"role": "user", "content": "你还记得我最喜欢的歌手是谁吗？"}],
        ],
    )

    try:
        with patch("app.api.admin.bug_reports.db", fake_db):
            response = api_client.post("/admin-api/bug-reports/bug-report-123456/eval-case", json={})

        assert response.status_code == 200
        data = response.json()
        assert data["appended"] is False
        assert data["path"] is None
        case = data["case"]
        assert case["id"].startswith("bug_bug-repo_")
        assert case["category"] == "memory_safety"
        assert case["priority"] == "P0"
        assert case["turns"] == [{"role": "user", "content": "你还记得我最喜欢的歌手是谁吗？"}]
        assert "must_not_contain_any" in data["jsonl"]
        assert case["source"]["bug_report_id"] == "bug-report-123456"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_generate_eval_case_can_append_when_explicit(api_client, tmp_path):
    app, require_admin_jwt = _admin_override()
    cases_path = tmp_path / "cases.jsonl"
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [{
                "id": "bug-report-abcdef",
                "error_types": ["tone"],
                "reason": "回复像机器人",
                "message_id": "msg-assistant",
                "message_role": "assistant",
                "assistant_reply": "作为AI，我可以帮你。",
                "created_at": "2026-05-17T10:00:00",
                "conversation_id": "conv-1",
            }],
            [{"role": "user", "content": "像朋友一样回我一句。"}],
        ],
    )

    try:
        with (
            patch("app.api.admin.bug_reports.db", fake_db),
            patch("app.api.admin.bug_reports._EVAL_CASES_PATH", cases_path),
        ):
            response = api_client.post(
                "/admin-api/bug-reports/bug-report-abcdef/eval-case",
                json={"append_to_cases": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["appended"] is True
        assert data["path"] == str(cases_path)
        written = cases_path.read_text(encoding="utf-8").strip()
        assert written == data["jsonl"]
        assert '"category":"tone"' in written
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


def test_generate_eval_case_rejects_invalid_assertion_override(api_client):
    app, require_admin_jwt = _admin_override()
    fake_db = MagicMock()
    fake_db.query_raw = AsyncMock(
        side_effect=[
            [{
                "id": "bug-report-invalid",
                "error_types": ["tone"],
                "reason": "bad assertion override",
                "message_id": "msg-assistant",
                "message_role": "assistant",
                "assistant_reply": "ok",
                "created_at": "2026-05-17T10:00:00",
                "conversation_id": "conv-1",
            }],
            [{"role": "user", "content": "你好"}],
        ],
    )

    try:
        with patch("app.api.admin.bug_reports.db", fake_db):
            response = api_client.post(
                "/admin-api/bug-reports/bug-report-invalid/eval-case",
                json={"assertions": [{"type": "unknown"}]},
            )

        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "invalid_eval_case"
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)
