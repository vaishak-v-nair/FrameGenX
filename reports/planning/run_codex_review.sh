#!/usr/bin/env bash
# Review-only launcher. Prompts prohibit application execution and edits.
set -u
phase="${1:-}"
case "$phase" in ceo|design|dx|eng) ;; *) echo 'Expected ceo, design, dx or eng' >&2; exit 2 ;; esac
export PATH="/c/Users/vaish/.bun/bin:/c/Users/vaish/AppData/Local/Programs/OpenAI/Codex/bin:$PATH"
source /c/Users/vaish/.claude/skills/gstack/bin/gstack-codex-probe
repo=/e/BrosKi/framegenx
out="$repo/reports/planning"
prompt="$out/${phase}_cli_prompt.txt"
[ ! -f "$out/${phase}_cli_document.txt" ] || prompt="$out/${phase}_cli_document.txt"
_gstack_codex_timeout_wrapper 600 codex exec -C "$repo" -s read-only -c 'web_search="disabled"' -o "$out/${phase}_codex.md" - < "$prompt" > "$out/${phase}_codex.log" 2>&1
review_exit=$?
printf '%s_CODEX_EXIT: %s\n' "$phase" "$review_exit"
tail -10 "$out/${phase}_codex.log"
exit "$review_exit"
