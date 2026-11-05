"""CHAT01.10.2 复读门禁的 GitHub 控制器 (quality-repetition-gate.yml 调用)。公开日志只有状态和数字。

    python -m evals.reply_repetition.gate --mode gate --case-set <set> --baseline-root . --candidate-root candidate
    python -m evals.reply_repetition.gate --mode calibrate --case-set <set> --baseline-root .

服务端 (OIDC) 给出用例、生产提示词快照、价格; 评测 key 只在本进程, 回放与打分子进程只拿到本地计费
代理的地址和占位 key (meter.py)。gate: 基线 (本次 main) 与候选各回放 runs_per_arm 次 (由仓库里的
calibration.json 定, 它须是同一用例集标定的), 打分后按 compare --gate 判定。calibrate: 只回放 main,
把 A/A 标定结果写进报告 (之后人工提交 calibration.json)。用例与回复文本只在内存和临时目录, 不上传。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from evals.quality.nightly import Control, public_failure

SHARDS = 4
PARALLEL_RUNS = 2
SAMPLES = 5
CALIBRATION_RUNS = 8


class GateControl(Control):
    PREFIX = "/repetition-gate"

    def retryable(self, path, operation):
        # 结算按块 ID 幂等; 预留块与结束不能盲目重放
        return path.startswith(self.PREFIX + "/jobs/") and operation in {"renew", "settle_block", "uncertain"}

    async def action(self, operation, data=None):
        return await self.request(self.PREFIX + "/jobs/" + self.job["id"],
                                  dict(lease_token=self.job["lease_token"], operation=operation, data=data or {}))

    async def reserve_block(self, model, amount):
        return (await self.action("reserve_block", dict(model=model, amount=str(amount))))["id"]

    async def settle_block(self, charge_id, usage):
        await self.action("settle_block", dict(charge_id=charge_id, usage=usage))


def prompt_rows(manifest: dict) -> list[dict]:
    """Web 管的提示词按生产原文; 其余 content=null, 各组用自己源码的默认文本 (部署会同步成那样)。"""
    protected = set(manifest["protected_prompts"])
    return [{**p, "content": p["content"] if p["key"] in protected else None} for p in manifest["prompts"]]


def arms(mode: str, runs: int) -> list[tuple[str, str]]:
    names = [("base", f"base-{i}") for i in range(1, runs + 1)]
    if mode == "gate":
        names += [("candidate", f"cand-{i}") for i in range(1, runs + 1)]
    # 两组交替, 时间上的波动 (供应商负载) 两边均摊
    return sorted(names, key=lambda a: (int(a[1].rsplit("-", 1)[1]), a[0]))


async def replay(root: Path, arm: str, files: dict, env: dict, cap: float) -> dict:
    procs = []
    for i in range(SHARDS):
        procs.append(await asyncio.create_subprocess_exec(
            sys.executable, "-m", "evals.reply_repetition.run_replay", "--arm", arm,
            "--cases", str(files["cases"]), "--prompt-snapshot", str(files["prompts"]),
            "--production-context", str(files["contexts"]), "--output-dir", str(files["runs"]),
            "--samples", str(SAMPLES), "--shard", f"{i}/{SHARDS}", "--max-cost-cny", str(cap), "--live",
            cwd=str(root), env={**env, "PYTHONPATH": str(root)},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL))
    done, failed, codes = 0, 0, []
    for proc in procs:
        out, _ = await proc.communicate()
        codes.append(proc.returncode)
        try:  # 最后一行是本分片的汇总 (只有编号和错误类型)
            summary = json.loads(out.decode().strip().splitlines()[-1])
            done += summary["done"]
            failed += len(summary["failed"])
        except (ValueError, IndexError, KeyError):
            failed += 1
    return dict(arm=arm, done=done, failed=failed, exit_codes=codes)


async def score(root: Path, files: dict, names: list[str], env: dict) -> int:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "evals.reply_repetition.score", "--cases", str(files["cases"]),
        "--runs", str(files["runs"]), "--arms", ",".join(names), "--output", str(files["scores"]),
        cwd=str(root), env={**env, "PYTHONPATH": str(root)},
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    return await proc.wait()


def verdict(mode: str, files: dict, names: list[str], calibration: dict | None, runs: int,
            meta: dict | None = None) -> dict:
    from evals.reply_repetition import compare as cmp

    cases = json.loads(files["cases"].read_text())
    scores = json.loads(files["scores"].read_text())
    if mode == "calibrate":
        metrics = cmp.calibrate(cases, scores, [names], runs)
        # 整份可直接提交为 calibration.json (只有数字与来源说明)
        document = cmp.calibration_document(metrics, {"runs_per_arm": runs, **(meta or {})})
        return dict(calibration=document, margins={m: v["margin"] for m, v in metrics.items()})
    candidate = "+".join(n for n in names if n.startswith("cand-"))
    baseline = "+".join(n for n in names if n.startswith("base-"))
    report = cmp.compare(cases, scores, candidate, baseline, calibration)
    result = cmp.gate(report, calibration)
    diffs = {k: report["paired_diff"][k] for k in ("positive_repeated", "negative_repeated", "all_responded")}
    return dict(gate=result, paired_diff=diffs, coverage={g: report["coverage"][g] for g in ("all", "positive")},
                missing=report["coverage"]["missing"], arms=report["arms"])


def read_keys() -> dict:
    """工作流把评测 key 写进文件 (不放进本进程环境变量, 子进程读不到); 读完即删。本地可用 EVAL_* 环境变量。"""
    folder = os.environ.get("EVAL_KEYS_DIR")
    keys = {}
    for provider, name in (("ark", "EVAL_ARK_API_KEY"), ("dashscope", "EVAL_DASHSCOPE_API_KEY")):
        path = Path(folder) / provider if folder else None
        if path is not None and path.exists():
            keys[provider] = path.read_text().strip()
            path.unlink()
        else:
            keys[provider] = os.environ.pop(name, "").strip()
    return keys


def child_env(port: int) -> dict:
    """回放与打分子进程: 只有代理地址和占位 key, 没有评测 key、OIDC 凭据、数据库或 Redis。"""
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT"}}
    proxy = f"http://127.0.0.1:{port}"
    return {**env, "PYTHON_DOTENV_DISABLED": "1", "PYTHONUNBUFFERED": "1", "TRACING_BACKEND": "none",
            "DATABASE_URL": "postgresql://test:test@127.0.0.1:1/quality_no_database",
            "REDIS_URL": "redis://127.0.0.1:1/15", "JWT_SECRET": "isolated-repetition-gate-renderer-no-auth",
            "ARK_API_KEY": "metered-by-local-proxy", "DASHSCOPE_API_KEY": "metered-by-local-proxy",
            "ARK_BASE_URL": proxy + "/ark", "DASHSCOPE_BASE_URL": proxy + "/dashscope"}


def check_candidate(root: Path) -> None:
    text = (root / "evals/reply_repetition/run_replay.py").read_text()
    if "GATE_PROTOCOL = 1" not in text:
        raise RuntimeError("candidate_predates_repetition_gate_protocol")


def check_calibration(calibration: dict, case_set: str, route: list[str]) -> int:
    from evals.reply_repetition.run_replay import MODEL

    if calibration.get("case_set") != case_set:
        raise RuntimeError("calibration_not_for_this_case_set")
    if calibration.get("reply_model") != MODEL or route[1] != MODEL:
        raise RuntimeError("calibration_not_for_this_reply_model")
    if int(calibration.get("samples_per_run") or 0) != SAMPLES:
        raise RuntimeError("calibration_sample_count_differs")
    return int(calibration["runs_per_arm"])


async def run(args, control: GateControl, manifest: dict, keys: dict) -> tuple[str, dict]:
    from app.config import settings
    from app.services.quality import transport
    from evals.reply_repetition.meter import Meter, serve
    from evals.reply_repetition.run_replay import MODEL

    roots = {"base": args.baseline_root.resolve()}
    calibration = None
    if args.mode == "gate":
        roots["candidate"] = args.candidate_root.resolve()
        check_candidate(roots["candidate"])
        calibration = json.loads((roots["base"] / "evals/reply_repetition/calibration.json").read_text())
        runs = check_calibration(calibration, args.case_set, manifest["route"])
    else:
        if manifest["route"][1] != MODEL:
            raise RuntimeError("route_differs_from_replay_model")
        runs = CALIBRATION_RUNS
    for model in manifest["prices"]:
        transport.pinned_price(model, manifest["prices"])  # 任何一个没定价: 开跑前就停, 不花钱
    meter = Meter(control, manifest["prices"],
                  {"ark": settings.ark_base_url, "dashscope": settings.dashscope_base_url}, keys, manifest["route"])
    port, stop = await serve(meter)
    report = dict(mode=args.mode, case_set=args.case_set, runs_per_arm=runs, samples=SAMPLES,
                  cases=len(manifest["cases"]), replays=[], scoring={})
    try:
        with tempfile.TemporaryDirectory(prefix="repetition-gate-") as folder:
            tmp = Path(folder)
            files = {"cases": tmp / "cases.json", "contexts": tmp / "contexts.json",
                     "prompts": tmp / "prompts.json", "runs": tmp / "runs"}
            for name, value in (("cases", manifest["cases"]), ("contexts", manifest["contexts"]),
                                ("prompts", prompt_rows(manifest))):
                files[name].write_text(json.dumps(value, ensure_ascii=False))
                files[name].chmod(0o600)
            env = child_env(port)
            replaying, scoring = asyncio.Semaphore(PARALLEL_RUNS), asyncio.Semaphore(PARALLEL_RUNS)

            async def one(group: str, arm: str) -> Path | None:
                async with replaying:
                    if meter.exhausted:
                        return None
                    # 单例费用上限交给计费代理; 这里只防失控 (远高于一次回放的实际花费)
                    report["replays"].append(await replay(roots[group], arm, files, env, cap=20))
                async with scoring:
                    out = tmp / f"scores-{arm}.json"
                    code = await score(roots["base"], {**files, "scores": out}, [arm], env)
                    report["scoring"][arm] = code
                    return out if code == 0 else None

            names = arms(args.mode, runs)
            outputs = await asyncio.gather(*[one(g, a) for g, a in names])
            if meter.exhausted:
                return "budget_exhausted", report
            if any(o is None for o in outputs):
                report["reason"] = "scoring_failed"
                return "failed", report
            scores = [row for out in outputs for row in json.loads(out.read_text())]
            files["scores"] = tmp / "scores.json"
            files["scores"].write_text(json.dumps(scores, ensure_ascii=False))
            meta = dict(case_set=args.case_set, cases=len(manifest["cases"]), samples_per_run=SAMPLES,
                        reply_model=MODEL, source_commit=os.environ.get("GITHUB_SHA", "")[:8],
                        repetition_judge="qwen3.5-plus", responded_judge="qwen3.5-flash")
            report.update(verdict(args.mode, files, [a for _, a in names], calibration, runs, meta))
            return "completed", report
    finally:
        await stop()
        await meter.close()
        report["spend"] = meter.summary()


def public(status: str, report: dict) -> dict:
    """只放状态与数字进公开日志。"""
    out = {"status": status, "mode": report.get("mode"), "reason": report.get("reason"),
           "cost_cny": (report.get("spend") or {}).get("cost_cny"),
           "replays": [{k: r[k] for k in ("arm", "done", "failed")} for r in report.get("replays", [])]}
    if "gate" in report:
        out.update(gate_passed=report["gate"]["passed"], checks=report["gate"]["checks"],
                   paired_diff={k: {"diff": v["diff"], "ci95": v["ci95"]} for k, v in report["paired_diff"].items()})
    if "margins" in report:
        out.update(margins=report["margins"], calibration=report["calibration"])
    return out


async def main(args) -> int:
    os.environ.update(PYTHON_DOTENV_DISABLED="1",
                      DATABASE_URL="postgresql://test:test@127.0.0.1:1/quality_no_database",
                      REDIS_URL="redis://127.0.0.1:1/15")
    keys = read_keys()
    candidate = None
    if args.mode == "gate":
        candidate = subprocess.check_output(["git", "-C", str(args.candidate_root), "rev-parse", "HEAD"],
                                            text=True).strip()
    control = GateControl()
    manifest = await control.request(GateControl.PREFIX + "/begin", dict(
        mode=args.mode, case_set=args.case_set, candidate_sha=candidate,
        keys_configured=bool(keys["ark"] and keys["dashscope"])))
    if manifest["status"] != "running" or "job" not in manifest:  # 重跑同一次 attempt 不会重放计费
        print(json.dumps({"status": manifest["status"], "reason": manifest.get("reason")}))
        return 0 if manifest["status"] in {"skipped", "completed"} else 1
    control.job = manifest["job"]

    async def heartbeat():
        while True:
            await asyncio.sleep(30)
            await control.renew()

    heart = asyncio.create_task(heartbeat())
    work = asyncio.create_task(run(args, control, manifest, keys))
    status, report = "failed", {"mode": args.mode}
    try:
        done, _ = await asyncio.wait([work, heart], return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            status, report = work.result()
        else:
            heart.result()
    except Exception as exc:
        report = {**report, "reason": "controller_error", **public_failure(exc)}
    finally:
        for task in (work, heart):
            task.cancel()
        await asyncio.gather(work, heart, return_exceptions=True)
    await control.finish(status, report)
    print(json.dumps(public(status, report), ensure_ascii=False))
    passed = report.get("gate", {}).get("passed", args.mode == "calibrate")
    return 0 if status == "completed" and passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "calibrate"), required=True)
    parser.add_argument("--case-set", required=True)
    parser.add_argument("--baseline-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-root", type=Path)
    parsed = parser.parse_args()
    if parsed.mode == "gate" and parsed.candidate_root is None:
        parser.error("--candidate-root is required for --mode gate")
    try:
        code = asyncio.run(main(parsed))
    except BaseException as exc:
        print(json.dumps(public_failure(exc)))
        code = 1
    sys.exit(code)
