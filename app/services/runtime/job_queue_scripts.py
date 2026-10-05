"""Fixed, bounded Redis transitions. Every accessed key is supplied in KEYS.

Check key types before writes: Lua isolates execution but does not roll back
writes when a later command raises. Queue leases fence queue results, not
external business effects. See docs/runtime-job-queue.md.
"""

_COMMON = """
local function check(key, expected)
    local kind = redis.call('TYPE', key).ok
    if kind ~= 'none' and kind ~= expected then
        error('Runtime queue key type mismatch: expected ' .. expected)
    end
end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local stamp = clock[1]
"""

ENQUEUE = _COMMON + """
check(KEYS[1], 'hash')
if ARGV[1] == '1' then check(KEYS[2], 'string') end
check(KEYS[3], 'list')
check(KEYS[4], 'zset')
if ARGV[1] == '1' then
    local current = redis.call('GET', KEYS[2]) or ''
    if current ~= ARGV[2] then return {'changed'} end
end
if redis.call('EXISTS', KEYS[1]) == 1 then
    if redis.call('HGET', KEYS[1], 'id') ~= ARGV[3]
       or not redis.call('HGET', KEYS[1], 'status') then
        return {'orphan', ARGV[3]}
    end
    return {'existing', ARGV[3]}
end
if ARGV[2] ~= '' then return {'orphan', ARGV[3]} end
local due = string.format('%.6f', now + tonumber(ARGV[7]))
redis.call('HSET', KEYS[1], 'id', ARGV[3], 'type', ARGV[4],
    'payload', ARGV[5], 'status', 'queued', 'attempts', '0',
    'max_attempts', ARGV[6], 'created_at', stamp, 'updated_at', stamp,
    'last_error', '', 'queue_version', '2', 'not_before', due)
redis.call('EXPIRE', KEYS[1], ARGV[8])
if tonumber(ARGV[7]) > 0 then
    redis.call('ZADD', KEYS[4], due, ARGV[3])
else
    redis.call('LPUSH', KEYS[3], ARGV[3])
end
if ARGV[1] == '1' then
    redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[8])
end
return {'created', ARGV[3]}
"""

CLAIM = _COMMON + """
check(KEYS[1], 'hash')
check(KEYS[2], 'list')
check(KEYS[3], 'zset')
check(KEYS[4], 'zset')
check(KEYS[5], 'string')
check(KEYS[6], 'list')
if redis.call('HGET', KEYS[1], 'status') ~= 'queued' then
    redis.call('LREM', KEYS[2], 0, ARGV[1])
    return {'skip'}
end
if ARGV[3] == '1' and not redis.call('LPOS', KEYS[2], ARGV[1]) then
    return {'skip'}
end
local due = redis.call('ZSCORE', KEYS[4], ARGV[1])
local recorded_due = tonumber(redis.call('HGET', KEYS[1], 'not_before') or '0')
if recorded_due and recorded_due > now then
    due = math.max(tonumber(due or '0'), recorded_due)
    redis.call('ZADD', KEYS[4], due, ARGV[1])
end
if due and tonumber(due) > now then
    redis.call('LREM', KEYS[2], 0, ARGV[1])
    return {'delayed'}
end
if redis.call('EXISTS', KEYS[5]) == 1 then
    redis.call('LREM', KEYS[2], 0, ARGV[1])
    redis.call('ZADD', KEYS[4], now + 5, ARGV[1])
    return {'busy'}
end
local attempts = tonumber(redis.call('HGET', KEYS[1], 'attempts') or '0')
local maximum = tonumber(redis.call('HGET', KEYS[1], 'max_attempts') or '3')
if not attempts or attempts < 0 or attempts ~= math.floor(attempts)
   or attempts > 1000000000 or not maximum or maximum < 1
   or maximum ~= math.floor(maximum) or maximum > 1000000000 then
    redis.call('HSET', KEYS[1], 'status', 'dead_letter', 'updated_at', stamp,
        'last_error', 'Invalid runtime job attempt counters')
    redis.call('LREM', KEYS[2], 0, ARGV[1])
    redis.call('ZREM', KEYS[3], ARGV[1])
    redis.call('ZREM', KEYS[4], ARGV[1])
    redis.call('LPUSH', KEYS[6], ARGV[1])
    return {'invalid'}
end
local expires = string.format('%.6f', now + tonumber(ARGV[4]) / 1000)
redis.call('SET', KEYS[5], ARGV[2], 'PX', ARGV[4])
redis.call('LREM', KEYS[2], 0, ARGV[1])
redis.call('ZREM', KEYS[4], ARGV[1])
redis.call('HSET', KEYS[1], 'status', 'running', 'attempts', attempts + 1,
    'updated_at', stamp, 'queue_version', '2', 'lease_token', ARGV[2],
    'lease_expires_at', tostring(expires), 'heartbeat_at', stamp)
redis.call('ZADD', KEYS[3], expires, ARGV[1])
return {'claimed', redis.call('HGETALL', KEYS[1])}
"""

RENEW = _COMMON + """
check(KEYS[1], 'hash')
check(KEYS[2], 'zset')
check(KEYS[3], 'string')
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'attempts') ~= ARGV[2]
   or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[3]
   or redis.call('GET', KEYS[3]) ~= ARGV[3]
   or redis.call('PTTL', KEYS[3]) <= 0
   or tonumber(redis.call('HGET', KEYS[1], 'lease_expires_at') or '0') <= now then
    return 0
end
local expires = string.format('%.6f', now + tonumber(ARGV[4]) / 1000)
redis.call('PEXPIRE', KEYS[3], ARGV[4])
redis.call('HSET', KEYS[1], 'updated_at', stamp,
    'heartbeat_at', stamp, 'lease_expires_at', tostring(expires))
redis.call('ZADD', KEYS[2], expires, ARGV[1])
return 1
"""

FINISH = _COMMON + """
check(KEYS[1], 'hash')
check(KEYS[2], 'zset')
check(KEYS[3], 'zset')
check(KEYS[4], 'list')
check(KEYS[5], 'list')
check(KEYS[6], 'string')
check(KEYS[7], 'list')
if redis.call('HGET', KEYS[1], 'status') ~= 'running'
   or redis.call('HGET', KEYS[1], 'attempts') ~= ARGV[2] then return 0 end
local token = redis.call('HGET', KEYS[1], 'lease_token') or ''
if token ~= ARGV[7] then return 0 end
if token ~= '' and (redis.call('GET', KEYS[6]) ~= token
   or redis.call('PTTL', KEYS[6]) <= 0
   or tonumber(redis.call('HGET', KEYS[1], 'lease_expires_at') or '0') <= now) then
    return 0
end
redis.call('HSET', KEYS[1], 'status', ARGV[3],
    'updated_at', stamp, 'last_error', ARGV[5])
redis.call('HDEL', KEYS[1], 'lease_token', 'lease_expires_at')
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call('LREM', KEYS[7], 0, ARGV[1])
if token ~= '' then redis.call('DEL', KEYS[6]) end
if ARGV[3] == 'queued' then
    redis.call('HSET', KEYS[1], 'not_before', ARGV[6])
    redis.call('ZADD', KEYS[3], ARGV[6], ARGV[1])
elseif ARGV[3] == 'dead_letter' then
    redis.call('LPUSH', KEYS[4], ARGV[1])
elseif ARGV[3] == 'succeeded' then
    redis.call('LPUSH', KEYS[5], ARGV[1])
    redis.call('LTRIM', KEYS[5], 0, 499)
end
return 1
"""

RECOVER = _COMMON + """
check(KEYS[1], 'hash')
check(KEYS[2], 'zset')
check(KEYS[3], 'list')
check(KEYS[4], 'zset')
check(KEYS[5], 'string')
check(KEYS[6], 'list')
local score = redis.call('ZSCORE', KEYS[2], ARGV[1])
if not score then return 0 end
if redis.call('HGET', KEYS[1], 'status') ~= 'running' then
    redis.call('ZREM', KEYS[2], ARGV[1])
    return 0
end
local token = redis.call('HGET', KEYS[1], 'lease_token') or ''
if token ~= '' then
    if tonumber(redis.call('HGET', KEYS[1], 'lease_expires_at') or '0') > now then
        return 0
    end
elseif tonumber(score) > tonumber(ARGV[2]) then return 0 end
-- A legacy worker or a freshly renewed owner must not be overlapped.
if redis.call('EXISTS', KEYS[5]) == 1 then return 0 end
local attempts = tonumber(redis.call('HGET', KEYS[1], 'attempts') or '0')
local maximum = tonumber(redis.call('HGET', KEYS[1], 'max_attempts') or '3')
local recovered = tonumber(redis.call('HGET', KEYS[1], 'recoveries') or '0') or 0
local exhausted = not attempts or not maximum or maximum < 1 or attempts >= maximum
redis.call('HDEL', KEYS[1], 'lease_token', 'lease_expires_at')
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[4], ARGV[1])
redis.call('LREM', KEYS[3], 0, ARGV[1])
redis.call('HSET', KEYS[1], 'status', exhausted and 'dead_letter' or 'queued',
    'updated_at', stamp, 'last_error', 'Execution interrupted: expired runtime lease',
    'recoveries', math.max(0, recovered) + 1, 'last_recovered_at', stamp,
    'not_before', stamp)
if exhausted then
    redis.call('LPUSH', KEYS[6], ARGV[1])
else
    redis.call('LPUSH', KEYS[3], ARGV[1])
end
return 1
"""

PROMOTE = _COMMON + """
check(KEYS[1], 'hash')
check(KEYS[2], 'zset')
check(KEYS[3], 'list')
local due = redis.call('ZSCORE', KEYS[2], ARGV[1])
if not due or tonumber(due) > now then return 0 end
if redis.call('HGET', KEYS[1], 'status') ~= 'queued' then
    redis.call('ZREM', KEYS[2], ARGV[1])
    return 0
end
redis.call('LREM', KEYS[3], 0, ARGV[1])
redis.call('LPUSH', KEYS[3], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
return 1
"""

RECONCILE = _COMMON + """
check(KEYS[1], 'hash')
check(KEYS[2], 'list')
check(KEYS[3], 'zset')
check(KEYS[4], 'zset')
check(KEYS[5], 'string')
local state = redis.call('HGET', KEYS[1], 'status')
if state == 'queued' then
    if redis.call('EXISTS', KEYS[5]) == 1
       or redis.call('LPOS', KEYS[2], ARGV[1])
       or redis.call('ZSCORE', KEYS[3], ARGV[1]) then return 0 end
    local recorded_due = redis.call('HGET', KEYS[1], 'not_before')
    if not recorded_due and ARGV[2] ~= '1' then return -1 end
    local due = tonumber(recorded_due or stamp)
    if not due then return -1 end
    redis.call('ZREM', KEYS[4], ARGV[1])
    if due > now then
        redis.call('ZADD', KEYS[3], due, ARGV[1])
    else
        redis.call('LPUSH', KEYS[2], ARGV[1])
    end
    return 1
elseif state == 'running' and not redis.call('ZSCORE', KEYS[4], ARGV[1]) then
    local score = tonumber(redis.call('HGET', KEYS[1], 'lease_expires_at')
        or redis.call('HGET', KEYS[1], 'updated_at')) or now
    redis.call('ZADD', KEYS[4], score, ARGV[1])
    return 1
end
return 0
"""

DIAGNOSE = _COMMON + """
check(KEYS[2], 'list')
check(KEYS[3], 'zset')
check(KEYS[4], 'zset')
check(KEYS[5], 'string')
local kind = redis.call('TYPE', KEYS[1]).ok
if kind == 'none' then return {'missing'} end
if kind ~= 'hash' then return {'malformed'} end
return {'ok', redis.call('HGET', KEYS[1], 'type') or '',
    redis.call('HGET', KEYS[1], 'status') or '',
    redis.call('HGET', KEYS[1], 'not_before') or '',
    redis.call('LPOS', KEYS[2], ARGV[1]) and '1' or '0',
    redis.call('ZSCORE', KEYS[3], ARGV[1]) or '',
    redis.call('ZSCORE', KEYS[4], ARGV[1]) or ''}
"""

DIAGNOSE_IDEMPOTENCY = _COMMON + """
check(KEYS[1], 'string')
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
local kind = redis.call('TYPE', KEYS[2]).ok
if kind ~= 'hash' then return 1 end
if redis.call('HGET', KEYS[2], 'id') ~= ARGV[1]
   or not redis.call('HGET', KEYS[2], 'status') then return 1 end
return 0
"""
