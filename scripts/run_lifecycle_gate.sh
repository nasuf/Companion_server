#!/usr/bin/env bash
# 闸门: 新的生命周期规则在三种访问假设下都不得让"有用记忆留存率"低于现行规则。
#
# 三个场景缺一不可。只跑第一个会得出过分乐观的结论 —— 那个场景假设有用的记忆被
# 用到更频繁, 而这本身就有利于任何基于使用信号的策略。后两个场景抽掉这个假设。
set -euo pipefail
cd "$(dirname "$0")/.."

SNAPSHOT="${1:-/tmp/lifecycle_snapshot.json}"
JUDGED="${2:-/tmp/agreement.json}"
PAIRS="${3:-/tmp/pairs_full.json}"

run() {
  echo
  echo "════ 场景: $3 ════"
  .venv/bin/python -m evals.memory_lifecycle.run_eval \
    --snapshot "$SNAPSHOT" --judged "$JUDGED" --pairs "$PAIRS" \
    --access-useful "$1" --access-other "$2" | sed -n '/闸门/,$p'
}

run 0.5  0.05 "使用与有用性相关 (乐观)"
run 0.05 0.05 "使用与有用性无关 (对照)"
run 0    0    "完全不访问 (最坏)"
