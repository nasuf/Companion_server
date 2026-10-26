"""Preregistered comparison of classifier dialogue budgets; never reads/writes live DB.

Prepare freezes prompts, cases, rendered inputs and scoring before any API calls.
Run consumes only that frozen file. No application runtime/model fallback is used.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from evals.utility_model.tasks import TASK_INTENT, TASK_MEMORY_RELEVANCE

TASKS = {t.key: t for t in (TASK_INTENT, TASK_MEMORY_RELEVANCE)}
VARIANTS = {'old': (3, 150, 900), 'wide': (5, 800, 3000), 'tight': (5, 150, 1500)}
SEED = 20261026


@dataclass(frozen=True)
class ContextCase:
    id: str
    task: str
    group: str
    message: str
    expected: str
    history: list[dict]
    query_terms: tuple[str, ...] = ()
    note: str = ''


def history(*turns: tuple[str, str | list[str]]) -> list[dict]:
    rows = []
    for n, (user, assistant) in enumerate(turns):
        for role, content in [('user', user)] + [('assistant', x) for x in ([assistant] if isinstance(assistant, str) else assistant)]:
            rows.append({'id': f'm{len(rows)}', 'role': role, 'content': content,
                         'createdAt': (datetime(2026, 10, 26, 1, tzinfo=UTC) + timedelta(seconds=n * 45 + len(rows))).isoformat()})
    return rows


def from_context(text: str) -> list[dict]:
    rows = []
    for i, line in enumerate(text.splitlines()):
        role, _, content = line.partition(':')
        if content:
            rows.append({'id': f'legacy{i}', 'role': 'assistant' if role.strip() == 'AI' else 'user',
                         'content': content.strip(), 'createdAt': '2026-10-26T01:00:00+00:00'})
    return rows


def schedule_guard_cases() -> list[ContextCase]:
    """Over-triggering guards for an intent wording change (2026-09-21 incident).

    Only added to the authorized intent experiment, so earlier frozen corpora
    and their evidence stay reproducible.
    """
    turns = lambda *pairs: history(*pairs)
    return [
        ContextCase('ack_unrelated_question', 'intent.unified', 'synthetic_multi_turn', '好呀', '日常交流',
                    turns(('今天有点累', '辛苦了'), ('想吃点好的', '想吃什么？'), ('还没想好', '嗯嗯'), ('你帮我选', '好'),
                          ('那你说', '那你明天想吃火锅还是烤肉？')), (), 'bare acknowledgement after a non-schedule question'),
        ContextCase('own_sleep_plan', 'intent.unified', 'synthetic_multi_turn', '我今晚要早点睡', '日常交流',
                    turns(('今天有点累', '辛苦了'), ('但还不想睡', '嗯'), ('想再聊聊', '我在'), ('你困吗', '有一点'),
                          ('那怎么办', '我可以晚点睡，继续陪你。')), (),
                    "the user's own schedule after an AI offer is not an AI schedule change"),
    ]


def synthetic_cases() -> list[ContextCase]:
    cases = []
    def add(task, name, message, expected, turns, terms=(), note=''):
        cases.append(ContextCase(name, task, 'synthetic_multi_turn', message, expected, history(*turns), terms, note))
    i = 'intent.unified'
    r = 'memory.relevance'
    # Latest-turn actions: earlier topics must not override explicit current intent.
    anchors = [
        ('reminder_to_invite', '提醒我明天八点吃药', '周末跟我一起去爬山吗', '日常交流'),
        ('delete_to_recall', '把我喜欢咖啡的事忘掉', '你还记得我喜欢喝什么吗', '日常交流'),
        ('crisis_to_third_person', '我今天压力很大', '电影里那个人说不想活了', '日常交流'),
        ('schedule_to_status', '你今晚能晚点睡吗', '你现在开心吗', '询问当前状态'),
    ]
    for name, anchor, message, label in anchors:
        add(i, name, message, label, [(anchor, '我听到了'), ('先不说这个', '好'), ('窗外下雨了', '听到雨声了'), ('有点凉', '披件衣服吧'), ('换个话题', '你说')], note='five turns; current explicit label wins')
    # Context-dependent schedule acknowledgement; bubble merging and assistant tail.
    for name, last, message in [
        ('schedule_multibubble', ['我可以晚点睡，继续陪你。', '你想让我再陪一会儿吗？'], '好，那你晚点睡'),
        ('schedule_earlybubble', ['要我晚一点睡，再陪你聊会儿吗？', '我也很喜欢这样聊天'], '好，那你再陪我十分钟'),
        ('schedule_longtail', '今天很平静。' * 50 + '要我再陪你一会儿，晚点睡吗？', '好，再陪一会儿'),
        ('close_longtail', '我们聊过许多小事。' * 35 + '那你早点休息，明天再聊吧。', '好，我先睡了，拜拜'),
    ]:
        add(i, name, message, '终结意图' if name.startswith('close') else '作息调整', [('今天有点累', '辛苦了'), ('但还不想睡', '嗯'), ('想再聊聊', '我在'), ('你困吗', '有一点'), ('那怎么办', last)], note='last assistant whole-turn tail and current request')
    # Older history should never downgrade current clear record/deletion/crisis intent.
    for name, message, label in [
        ('current_reminder', '改成后天九点提醒我', '记录请求'),
        ('current_delete', '把这件事从记忆里删掉', '删除'),
        ('current_apology', '对不起，我刚才不该冲你发火', '道歉承诺'),
        ('current_crisis', '我不想活了，想结束自己的生命', '危机求助'),
    ]:
        add(i, name, message, label, [('那件事让我不开心', '我听着'), ('暂时不想细说', '可以'), ('今天还下雨了', '嗯'), ('喝了杯茶', '暖一点'), ('我现在想告诉你', '你慢慢说')], note='latest explicit intent under unrelated multi-turn history')
    # Four-/five-turn referents are absent from old3 but present in both five-turn arms.
    entities = [
        ('film', '我昨天看了《星际穿越》', '那部电影你看过吗？', '中', ('星际穿越',)),
        ('band', '我最近在听落日飞车这个乐队', '那个乐队你听过吗？', '中', ('落日飞车',)),
        ('person', '我同事叫小野', '那他在哪家公司上班来着？', '强', ('小野',)),
        ('dog', '我家狗叫汤圆', '那它以前得过什么病来着？', '强', ('汤圆',)),
        ('trip', '我去过青海湖', '那个地方你去过吗？', '中', ('青海湖',)),
        ('book', '我刚看完《小王子》', '那本书你喜欢吗？', '中', ('小王子',)),
        ('friend', '我朋友小禾是程序员', '那她生日是什么时候来着？', '强', ('小禾',)),
        ('camera', '我前天买了富士相机', '那个相机我以前说过想买哪个颜色？', '强', ('富士',)),
    ]
    for name, anchor, message, level, terms in entities:
        add(r, f'referent_{name}', message, level, [(anchor, '听起来有意思'), ('说起这个话题我挺感兴趣', '你慢慢说'), ('刚才去接了杯水', '嗯'), ('回来了', '好'), ('继续刚才的话题吧', '你问')], terms, 'referent only in oldest of five turns; enhanced_query grounded noun required')
    add(r, 'latest_referent', '那本书你喜欢吗？', '中', [('我以前看《三体》', '嗯'), ('后来没看完', '有些长'), ('最近忙', '慢慢来'), ('换了一本', '哪本'), ('现在在看《活着》', '我知道这本')], ('活着',), 'latest explicit noun must replace old noun')
    add(r, 'long_user_head', '那个地方你去过吗？', '中', [('我打算去大理。' + '我做了一些路线功课，还想慢慢考虑。' * 25, '你想聊那里吗'), ('是啊', '好'), ('先喝口水', '嗯'), ('好了', '你说'), ('继续聊那里', '好')], ('大理',), 'user head preserved under tight cap')
    add(r, 'weak_ack', '好的', '弱', [('我昨天失眠', '辛苦了'), ('工作也有压力', '听起来累'), ('今晚会早点睡', '好'), ('喝了水', '嗯'), ('现在准备休息了', ['放下手机吧', '晚安'])], note='current simple acknowledgement remains weak despite emotional history')
    add(r, 'multibubble_target', '那个乐队你喜欢吗？', '中', [('听歌了', '什么歌'), ('想听摇滚', '好'), ('你有听过吗', '听过一些'), ('说一个名字', '可以'), ('你说吧', ['我最近听痛仰', '他们有几首歌挺有力量'])], ('痛仰',), 'referent lives in earlier assistant bubble')
    return cases


def cases() -> list[ContextCase]:
    result = []
    for task in TASKS.values():
        for n, case in enumerate(task.cases):
            result.append(ContextCase(f'{task.key}:{n:02}', task.key, 'existing_label_anchor', case.message,
                                      case.expected, from_context(case.context), note=case.note))
    return result + synthetic_cases()


def old_context(rows: list[dict]) -> str:
    # Frozen baseline data_fetch_phase.format_dialogue_lines/format_recent_context,
    # retaining its no-timestamp format and whole-line drop semantics.
    merged = []
    for row in rows:
        text = str(row.get('content') or '').strip()
        if not text:
            continue
        role = 'assistant' if row.get('role') == 'assistant' else 'user'
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + ' ' + text)
        else:
            merged.append((role, text))
    users = [n for n, row in enumerate(merged) if row[0] == 'user']
    window = merged[users[-3]:] if len(users) >= 3 else merged
    lines = []
    for role, text in window:
        if len(text) > 150:
            text = '…' + text[-150:] if role == 'assistant' else text[:150] + '…'
        lines.append(('AI' if role == 'assistant' else '用户') + ': ' + text)
    chosen, used = [], 0
    for line in reversed(lines):
        if chosen and used + len(line) + 1 > 900:
            break
        chosen.append(line)
        used += len(line) + 1
    return '\n'.join(reversed(chosen)) or '（无）'


def render_context(rows: list[dict], variant: str) -> str:
    if variant == 'old':
        return old_context(rows)
    from app.services.chat.dialogue_context import render_dialogue
    turns, line_chars, max_chars = VARIANTS[variant]
    return render_dialogue(rows, turns=turns, line_chars=line_chars,
                           max_chars=max_chars, purpose='utility_eval').text


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(snapshot: Path, target: Path, *, candidate_snapshot: Path | None = None,
            all_repeats: int | None = None, variants: tuple[str, ...] | None = None,
            allow_intent_change: bool = False):
    if target.exists():
        raise SystemExit('Refusing to overwrite preregistered manifest')
    prompts = {row['key']: row['content'] for row in json.loads(snapshot.read_text()) if row['key'] in TASKS}
    candidate_prompts = ({row['key']: row['content'] for row in json.loads(candidate_snapshot.read_text())
                          if row['key'] in TASKS} if candidate_snapshot else prompts)
    if set(candidate_prompts) != set(TASKS):
        raise SystemExit('Candidate prompt snapshot must contain both classifier keys')
    if candidate_prompts['intent.unified'] != prompts['intent.unified'] and not allow_intent_change:
        # An intent wording change is a separate, explicitly authorized experiment.
        raise SystemExit('Intent prompt changes are outside this comparison scope')
    chosen_variants = variants or tuple(VARIANTS)
    if not chosen_variants or len(set(chosen_variants)) != len(chosen_variants) or set(chosen_variants) - set(VARIANTS):
        raise SystemExit('Invalid variants')
    if all_repeats is not None and all_repeats not in (1, 3):
        raise SystemExit('Repeats must be 1 or 3')
    corpus = cases() + (schedule_guard_cases() if allow_intent_change else [])
    jobs = []
    for case in corpus:
        for variant in chosen_variants:
            context = render_context(case.history, variant)
            params = {'context': context, 'user_message': case.message, 'message': case.message}
            template = prompts[case.task] if variant == 'old' else candidate_prompts[case.task]
            prompt = template.format_map(params)
            repeats = all_repeats if all_repeats is not None else (3 if case.group == 'synthetic_multi_turn' else 1)
            for repeat in range(repeats):
                jobs.append({'job_id': f'{case.id}:{variant}:{repeat}', 'case': asdict(case),
                             'variant': variant, 'repeat': repeat, 'prompt': prompt,
                             'prompt_sha256': digest(prompt.encode()), 'context_chars': len(context)})
    random.Random(SEED).shuffle(jobs)
    value = {'created_at': datetime.now(UTC).isoformat(), 'schema': 1,
             'model': 'qwen3.5-flash', 'provider': 'dashscope', 'temperature': 0,
             'max_tokens': 256, 'concurrency': 3, 'seed': SEED, 'calls_max': len(jobs),
             'snapshot_sha256': digest(snapshot.read_bytes()),
             'candidate_snapshot_sha256': digest(candidate_snapshot.read_bytes()) if candidate_snapshot else None,
             'variants': {key: VARIANTS[key] for key in chosen_variants}, 'all_repeats': all_repeats,
             'intent_prompt_changed': candidate_prompts['intent.unified'] != prompts['intent.unified'],
             'selection_policy': {'prefer': 'tight if no reproducible label regression vs old and wide',
                                  'reproducible_regression': 'when all arms have3 draws: old0 failures, candidate>=2 of3 failures; single draws are descriptive',
                                  'query_grounding': 'synthetic relevance: expected noun substring in enhanced_query; report separately from level accuracy',
                                  'cost': 'report actual provider tokens and observed cache; missing cache yields cost interval, never silently zero',
                                  'budget': 'tight1500 supports 5 short turns; constant hard cap even at configured20; no 6000 growth',
                                  'limitations': 'synthetic screening, not production-weighted; no claim of overall batch release acceptance; latency nonstreamed full response'},
             'jobs': jobs}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest': str(target), 'sha256': digest(target.read_bytes()), 'calls': len(jobs),
                      'cases': len(corpus), 'synthetic': len(synthetic_cases())}, ensure_ascii=False))


def score(task_key: str, raw: str, expected: str, query_terms) -> dict:
    got = TASKS[task_key].parse(raw)
    grounded = None
    query = None
    if query_terms:
        try:
            query = str(json.loads(raw)['enhanced_query'])
            grounded = all(term in query for term in query_terms)
        except (ValueError, TypeError, KeyError):
            grounded = False
    return {'got': got, 'correct': got == expected, 'query': query, 'query_grounded': grounded}


async def run(manifest_path: Path, output: Path, *, max_calls: int = 477):
    from openai import AsyncOpenAI
    manifest = json.loads(manifest_path.read_text())
    if output.exists():
        raise SystemExit('Refusing to overwrite evidence; no automatic resume/retry')
    api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not api_key:
        raise SystemExit('DASHSCOPE_API_KEY is required in process environment')
    jobs = manifest['jobs']
    if len(jobs) > max_calls or len(jobs) != manifest['calls_max']:
        raise SystemExit('Call-budget mismatch')
    stop = asyncio.Event()
    lock = asyncio.Lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Claim before issuing any paid request; concurrent starts cannot both pass
    # the earlier existence check and silently duplicate the complete batch.
    with output.open('x'):
        pass
    client = AsyncOpenAI(api_key=api_key, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', timeout=60, max_retries=0)
    iterator = iter(jobs)
    completed = 0
    async def worker():
        nonlocal completed
        while not stop.is_set():
            job = next(iterator, None)
            if job is None:
                return
            row = {key: job[key] for key in ('job_id', 'variant', 'repeat', 'prompt_sha256', 'context_chars')}
            row.update({'case_id': job['case']['id'], 'task': job['case']['task'], 'group': job['case']['group'],
                        'expected': job['case']['expected'], 'manifest_sha256': digest(manifest_path.read_bytes()),
                        'started_at': datetime.now(UTC).isoformat(), 'model': manifest['model']})
            start = time.perf_counter()
            try:
                result = await client.chat.completions.create(
                    model=manifest['model'], messages=[{'role': 'user', 'content': job['prompt']}],
                    temperature=manifest['temperature'], max_tokens=manifest['max_tokens'],
                    extra_body={'enable_thinking': False})
                raw = result.choices[0].message.content or ''
                usage = result.usage.model_dump() if result.usage else {}
                details = usage.get('prompt_tokens_details') or {}
                row.update({'raw': raw, 'provider_usage': usage, 'input_tokens': usage.get('prompt_tokens'),
                            'output_tokens': usage.get('completion_tokens'), 'cached_tokens': details.get('cached_tokens'),
                            'finish_reason': result.choices[0].finish_reason,
                            **score(job['case']['task'], raw, job['case']['expected'], job['case']['query_terms'])})
            except Exception as exc:
                # Error class/status are enough for diagnosis; do not print request headers/body.
                row.update({'error_type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)})
                stop.set()
            row['latency_seconds'] = time.perf_counter() - start
            async with lock:
                with output.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                completed += 1
                if completed % 24 == 0 or stop.is_set():
                    print(json.dumps({'completed': completed, 'max': len(jobs), 'stopped_on_error': stop.is_set()}), flush=True)
    try:
        await asyncio.gather(*(worker() for _ in range(manifest['concurrency'])))
    finally:
        await client.close()
    print(json.dumps({'completed': completed, 'planned': len(jobs), 'status': 'failed' if stop.is_set() else 'complete'}), flush=True)


def percentile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * q) - 1)] if values else None


def summarize(manifest_path: Path, source: Path, target: Path):
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    seen = {row['job_id'] for row in rows}
    planned = {job['job_id'] for job in manifest['jobs']}
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task'], row['group'], row['variant'])].append(row)
    result = []
    for (task, group, variant), entries in sorted(groups.items()):
        success = [x for x in entries if 'error_type' not in x]
        query = [x['query_grounded'] for x in success if x.get('query_grounded') is not None]
        known = [x for x in success if x.get('input_tokens') is not None and x.get('output_tokens') is not None]
        # Existing utility runner's listed CNY per million: 0.2/0.04/2.0.
        # This is an explicitly labelled estimate, not a claimed invoice total.
        def cost(row, lower):
            cached = row.get('cached_tokens')
            if cached is None:
                cached = row['input_tokens'] if lower else 0
            cached = min(cached, row['input_tokens'])
            return ((row['input_tokens'] - cached) * .2 + cached * .04 + row['output_tokens'] * 2) / 1e6
        result.append({'task': task, 'group': group, 'variant': variant, 'n': len(entries),
                       'errors': len(entries)-len(success), 'correct': sum(x['correct'] for x in success),
                       'accuracy_over_all_calls': sum(x['correct'] for x in success)/len(entries),
                       'query_grounded': sum(query), 'query_denominator': len(query),
                       'latency_p50_seconds': percentile([x['latency_seconds'] for x in success], .5),
                       'latency_p90_seconds': percentile([x['latency_seconds'] for x in success], .9),
                       'mean_input_tokens': sum(x['input_tokens'] for x in known)/len(known) if known else None,
                       'max_input_tokens': max((x['input_tokens'] for x in known), default=None),
                       'cache_missing_calls': sum(x.get('cached_tokens') is None for x in known),
                       'estimated_cny': {'lower': sum(cost(x,True) for x in known), 'upper': sum(cost(x,False) for x in known)},
                       'cost_pricing_basis': 'existing evals.utility_model.run_eval.PRICING qwen3.5-flash; not bill reconciliation'})
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row['case_id']].setdefault(row['variant'], []).append(row)
    regressions = []
    for case_id, variants in by_case.items():
        old = variants.get('old', [])
        if len(old) != 3 or not all(x.get('correct') for x in old):
            continue
        for variant in ('wide','tight'):
            entries = variants.get(variant, [])
            if len(entries) == 3 and sum(not x.get('correct') for x in entries) >= 2:
                regressions.append({'case_id': case_id, 'variant': variant})
    value = {'status': 'complete' if seen == planned and len(rows) == len(planned) and not any('error_type' in x for x in rows) else 'incomplete_or_failed',
             'manifest_sha256': digest(manifest_path.read_bytes()), 'calls': len(rows), 'planned': len(planned),
             'by_task_group_variant': result, 'reproducible_label_regressions': regressions,
             'limitations': manifest['selection_policy']}
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'summary': str(target), 'status': value['status'], 'calls': len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--snapshot', type=Path, required=True)
    prepare_parser.add_argument('--manifest', type=Path, required=True)
    prepare_parser.add_argument('--candidate-snapshot', type=Path)
    prepare_parser.add_argument('--all-repeats', type=int, choices=(1, 3))
    prepare_parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS))
    prepare_parser.add_argument('--allow-intent-change', action='store_true',
                                help='explicitly authorized intent.unified wording experiment')
    live_parser = sub.add_parser('run')
    live_parser.add_argument('--manifest', type=Path, required=True)
    live_parser.add_argument('--output', type=Path, required=True)
    live_parser.add_argument('--max-calls', type=int, default=477)
    summary_parser = sub.add_parser('summarize')
    summary_parser.add_argument('--manifest', type=Path, required=True)
    summary_parser.add_argument('--input', type=Path, required=True)
    summary_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.snapshot, args.manifest, candidate_snapshot=args.candidate_snapshot,
                                          all_repeats=args.all_repeats, variants=tuple(args.variants) if args.variants else None,
                                          allow_intent_change=args.allow_intent_change)
    elif args.command == 'run': asyncio.run(run(args.manifest,args.output,max_calls=args.max_calls))
    else: summarize(args.manifest,args.input,args.output)


if __name__ == '__main__':
    main()
