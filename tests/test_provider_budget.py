from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError

from app.services.ops import provider_budget as b, service_health as h


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(b, '_active', set())
    monkeypatch.setattr(b, '_pending', {})
    monkeypatch.setattr(b, '_overflow', False)


def error(code, status=403):
    return APIStatusError('private provider text', response=httpx.Response(status, request=httpx.Request('POST', 'https://example.invalid')),
                          body={'error': {'code': code, 'message': 'private provider text'}})


@pytest.mark.parametrize('provider,code,expected', [
    ('ark', 'AccountOverdueError', 'AccountOverdueError'),
    ('ark', 'OperationDenied.ServiceOverdue', 'OperationDenied.ServiceOverdue'),
    ('ark', 'SetLimitExceeded', 'SetLimitExceeded'),
    ('dashscope', 'Arrearage', 'Arrearage'),
    ('dashscope', 'insufficient_quota', 'insufficient_quota'),
    ('dashscope', 'AllocationQuota.FreeTierOnly', 'AllocationQuota.FreeTierOnly'),
    ('ark', 'APIAccountRpmRateLimitExceeded', None),
    ('dashscope', 'Throttling.RateQuota', None),
    ('ark', 'Arrearage', None), ('unknown', 'AccountOverdueError', None),
    ('ark', 'InvalidApiKey', None), ('ark', 'AccessDenied', None),
])
def test_only_exact_provider_billing_codes(provider, code, expected):
    assert b.classify(provider, error(code)) == expected
    assert b.classify(provider, ValueError(code)) is None


def test_nested_error_and_safe_route_identifiers():
    outer = RuntimeError('private'); outer.__cause__ = error('Arrearage')
    assert b.classify('dashscope', outer) == 'Arrearage'
    assert b.route('ark', 'ark/ep-20261028') == ('ark', 'ep-20261028')
    for model in ['', 'a'*161, 'https://secret.invalid/key', 'model\nforged']:
        assert b.route('ark', model) is None
    assert b.route('unknown', 'model') is None


async def test_healthy_traffic_does_not_touch_database(monkeypatch):
    database = SimpleNamespace(execute_raw=AsyncMock(side_effect=AssertionError('no healthy SQL')))
    monkeypatch.setattr(b, 'db', database)
    assert await b.observed_call(AsyncMock(return_value='ok'), 'ark', 'm') == 'ok'
    await b.record_failure('ark', 'm', error('APIAccountRpmRateLimitExceeded', 429))
    database.execute_raw.assert_not_called()


async def test_failure_persistence_is_retried_and_diagnostics_are_sanitized(monkeypatch, caplog):
    persist = AsyncMock(side_effect=RuntimeError('PRIVATE_CONFIGURATION'))
    monkeypatch.setattr(b, '_persist', persist)
    await b.record_failure('ark', 'ark/m', error('AccountOverdueError'))
    assert ('ark', 'm') in b._active and ('ark', 'm') in b._pending
    assert 'PRIVATE_CONFIGURATION' not in caplog.text and 'private provider text' not in caplog.text
    monkeypatch.setattr(b, 'db', SimpleNamespace(query_raw=AsyncMock(return_value=[])))
    await b.refresh_active()
    assert persist.await_count == 2 and ('ark', 'm') in b._active
    await b.record_success('ark', 'm', datetime.now(UTC))  # pending failure is not silently cleared
    assert ('ark', 'm') in b._pending


async def test_pending_limit_is_explicitly_unknown(monkeypatch):
    monkeypatch.setattr(b, 'LIMIT', 0)
    await b.record_failure('ark', 'm', error('AccountOverdueError'))
    monkeypatch.setattr(b, 'db', SimpleNamespace(query_raw=AsyncMock(return_value=[])))
    result = await b.measure(datetime.now(UTC), {})
    assert result['status'] == 'unknown' and result['alerting']


@pytest.mark.parametrize('complete', [True, False])
async def test_stream_recovery_requires_complete_success_and_closes_source(monkeypatch, complete):
    failure, success = AsyncMock(), AsyncMock()
    monkeypatch.setattr(b, 'record_failure', failure)
    monkeypatch.setattr(b, 'record_success', success)
    closed = []
    async def source():
        try:
            yield 'chunk'
            if not complete: raise error('SetLimitExceeded')
        finally: closed.append(True)
    if complete:
        assert [x async for x in b.observed_stream(source, 'ark', 'm')] == ['chunk']
        success.assert_awaited_once(); failure.assert_not_called()
    else:
        with pytest.raises(APIStatusError):
            [x async for x in b.observed_stream(source, 'ark', 'm')]
        failure.assert_awaited_once(); success.assert_not_called()
    assert closed == [True]


async def test_cancelled_stream_never_reports_recovery(monkeypatch):
    success = AsyncMock(); monkeypatch.setattr(b, 'record_success', success)
    async def source():
        yield 'one'
        yield 'two'
    stream = b.observed_stream(source, 'ark', 'm')
    assert await anext(stream) == 'one'
    await stream.aclose()
    success.assert_not_called()


def test_new_route_incident_alerts_without_repeating_old_or_misreporting_recovery():
    def sample(model, at='2026-10-28T01:00:00Z'):
        return {'checks': {'provider_budget': {'status': 'measured', 'alerting': True,
                'incidents': [{'provider': 'ark', 'model': model, 'first_failure_at': at, 'reason': 'overdue'}]}}}
    first = sample('primary')
    notify, notified = h.decide('warning', first, {}, 0)
    assert notify
    assert not h.decide('warning', first, notified, 0)[0]
    assert h.decide('warning', sample('secondary'), notified, 0)[0]
    assert h.decide('warning', sample('primary', '2026-10-28T02:00:00Z'), notified, 0)[0]
    assert 'ark/primary（欠费）' in h.message_for('warning', first)[1]
    assert h.level_for({'checks': {'provider_budget': {'status': 'unknown'}}}, 'warning') == 'warning'


def test_deepseek_402_is_billing_but_generic_code_alone_is_not():
    assert b.classify('deepseek', error('invalid_request_error', 402)) == 'HTTP402'
    assert b.classify('deepseek', error('invalid_request_error', 400)) is None
    assert b.classify('ark', error('invalid_request_error', 402)) is None


async def test_failed_recovery_write_keeps_incident_and_never_logs_sensitive_text(monkeypatch, caplog):
    b._active.add(('ark', 'm'))
    monkeypatch.setattr(b, 'db', SimpleNamespace(execute_raw=AsyncMock(side_effect=RuntimeError('PRIVATE_CONFIGURATION'))))
    await b.record_success('ark', 'm', datetime.now(UTC))
    assert ('ark', 'm') in b._active
    assert 'recovery persistence failed' in caplog.text and 'PRIVATE_CONFIGURATION' not in caplog.text


async def test_stream_cleanup_failure_does_not_replace_original_failure(monkeypatch):
    failure = AsyncMock(); monkeypatch.setattr(b, 'record_failure', failure)
    class Broken:
        def __aiter__(self): return self
        async def __anext__(self): raise error('AccountOverdueError')
        async def aclose(self): raise RuntimeError('close failed')
    with pytest.raises(APIStatusError):
        [x async for x in b.observed_stream(Broken, 'ark', 'm')]
    failure.assert_awaited_once()


async def test_retry_retains_first_failure_time_and_latest_error(monkeypatch):
    monkeypatch.setattr(b, '_persist', AsyncMock(side_effect=RuntimeError('synthetic SQL outage')))
    await b.record_failure('ark', 'm', error('AccountOverdueError'))
    first = b._pending[('ark', 'm')]
    await b.record_failure('ark', 'm', error('SetLimitExceeded'))
    latest = b._pending[('ark', 'm')]
    assert latest[1] == first[1] and latest[2] >= first[2] and latest[0] == 'SetLimitExceeded'
