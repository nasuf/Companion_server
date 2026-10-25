"""Native read-only production probe. Sent over SSH; never imports application code."""
import datetime as dt
import hashlib
import json
import re
import subprocess

UUID = re.compile(r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
START = re.compile(r"\[DELAY-START\] conv=([a-f0-9-]{36}) message=([a-f0-9-]{36}) due_to_start_seconds=([\d.]+|None) reason=(\S+)")
LIMIT = 500


def native(args, stdin=None):
    result = subprocess.run(args, input=stdin, text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise RuntimeError("read-only native probe failed: " + args[0])
    return result.stdout


def reply_query(conversations, message_ids):
    """Only UUIDs and validated timestamps enter SQL; all statements are read-only."""
    if len(conversations) > LIMIT or len(message_ids) > LIMIT:
        raise ValueError("query scope exceeds evidence bound")
    if any(not UUID.fullmatch(x) for x in [*conversations, *message_ids]):
        raise ValueError("invalid evidence identity")
    scopes = []
    for cid, start in conversations.items():
        epoch = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
        if not 0 < epoch < 4102444800:
            raise ValueError("invalid evidence timestamp")
        scopes.append(f"(m.conversation_id='{cid}' AND m.created_at>=to_timestamp({epoch}))")
    ids = ",".join("'" + mid + "'" for mid in message_ids)
    if ids:
        scopes.append(f"m.id=ANY(ARRAY[{ids}]::text[])")
    if not scopes:
        return None
    return """BEGIN READ ONLY; SET LOCAL statement_timeout='8s';
SELECT json_build_object('id',m.id,'conversation_id',m.conversation_id,
 'created_at',m.created_at,'age_seconds',EXTRACT(EPOCH FROM(NOW()-m.created_at)),
 'reply_id',r.message_id,'reply_wait_seconds',r.wait_seconds)
FROM messages m LEFT JOIN LATERAL (
 SELECT t.message_id,EXTRACT(EPOCH FROM(a.created_at-m.created_at)) AS wait_seconds
 FROM turn_followup_events t JOIN messages a ON a.id=t.message_id
 WHERE t.scope->>'conversation_id'=m.conversation_id AND t.user_message_ids ? m.id
 AND a.role='assistant' AND a.conversation_id=m.conversation_id
 ORDER BY a.created_at LIMIT 1
) r ON true
WHERE m.role='user' AND (""" + " OR ".join(scopes) + """ )
ORDER BY m.created_at,m.id LIMIT 501;
COMMIT;"""


def collect(request):
    identity_format = '{{json .Id}}|{{json .Image}}|{{json .State.StartedAt}}'
    identity = native(['docker', 'inspect', '--format', identity_format, 'companion-server']).strip()
    container, image, started = map(json.loads, identity.split('|'))
    # Source hashes qualify the *running image*, not the host checkout or a mutable tag.
    files = request['source_files']
    allowed = re.compile(r'^(?:(?:app|jobs|prisma|scripts)/[a-zA-Z0-9_./-]+|pyproject.toml|runtime-constraints.txt)$')
    if not files or len(files) > 3000 or any(not allowed.fullmatch(p) or '..' in p.split('/') for p in files):
        raise ValueError('invalid source manifest')
    code = "import hashlib,json,pathlib; files=" + repr(files) + "; root=pathlib.Path('/app'); actual=sorted(str(p.relative_to(root)) for folder in ('app','jobs','prisma','scripts') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'); print(json.dumps({'hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() if (root/p).is_file() else None for p in files},'files':actual+['pyproject.toml','runtime-constraints.txt']}))"
    source = json.loads(native(['docker', 'exec', 'companion-server', 'python', '-c', code]))
    logs = subprocess.run(['docker', 'logs', '--timestamps', '--tail', '10000', 'companion-server'],
                          text=True, capture_output=True, timeout=25)
    if logs.returncode:
        raise RuntimeError('native log read failed')
    lines = sorted((logs.stdout + logs.stderr).splitlines())
    samples = []
    occurrences = {}
    for line in lines:
        match = START.search(line)
        if match:
            cid, mid, wait, reason = match.groups()
            if not UUID.fullmatch(cid) or not UUID.fullmatch(mid):
                raise ValueError('invalid delay log identity')
            digest = hashlib.sha256(line.encode()).hexdigest()
            occurrences[digest] = occurrences.get(digest, 0) + 1
            samples.append(dict(event=f'{container}:{digest}:{occurrences[digest]}',
                                conversation_id=cid, message_id=mid,
                                due_to_start_seconds=None if wait == 'None' else float(wait),
                                reason=reason))
    conversations = dict(request.get('conversations', {}))
    mids = set(request.get('message_ids', []))
    for sample in samples:
        cid = sample['conversation_id']
        conversations[cid] = min(conversations.get(cid, started), started)
        mids.add(sample['message_id'])
    truncated = len(lines) >= 10000 or len(samples) > LIMIT or len(mids) > LIMIT or len(conversations) > LIMIT
    rows = []
    if not truncated:
        sql = reply_query(conversations, sorted(mids))
        if sql:
            raw = native(['docker', 'exec', '-i', 'companion-postgres', 'sh', '-c',
                          'exec psql -X -qAt -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"'], sql)
            rows = [json.loads(line) for line in raw.splitlines() if line.startswith('{')]
            truncated = len(rows) > LIMIT
    queue = int(native(['docker', 'exec', 'companion-redis', 'redis-cli', '--raw', 'ZCARD', 'delayed:due']))
    after = native(['docker', 'inspect', '--format', identity_format, 'companion-server']).strip()
    if identity != after:
        raise RuntimeError('container changed during observation; snapshot not accepted')
    return dict(sampled_at=dt.datetime.now(dt.timezone.utc).isoformat(), container=container,
                image=image, started_at=started, source_hashes=source['hashes'], source_files=source['files'], samples=samples[:LIMIT],
                rows=rows[:LIMIT], truncated=truncated, log_lines=len(lines),
                # Only digests leave this function; arbitrary log contents are excluded.
                log_anchors=[hashlib.sha256(x.encode()).hexdigest() for x in lines],
                queue_conversations=queue)
