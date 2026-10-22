# Verify planning artifacts only. Does not import or execute the application.
$ErrorActionPreference = 'Stop'
$planPath = 'docs/plans/STACK_EVOLUTION_PLAN.md'
$plan = Get-Content -LiteralPath $planPath -Raw
$audit = @([regex]::Matches($plan, '(?m)^\| A(\d+) \|[^\r\n]+'))
$ids = @($audit | ForEach-Object { [int]$_.Groups[1].Value })
$taste = @($audit | Where-Object Value -Match '\| Taste T\d')
$challenges = @($audit | Where-Object Value -Match '\| User Challenge UC')
$mechanical = @($audit | Where-Object Value -Match '\| Mechanical')
$eng = $plan.Split('## 10. Engineering review')[1].Split('## 11. Cross-phase')[0]
$design = $plan.Split('## 8. Design review')[1].Split('## 9. Developer')[0]
$dx = $plan.Split('## 9. Developer experience review')[1].Split('## 10. Engineering')[0]
$testPart = $eng.Split('### Pass 3:')[1].Split('### Pass 4:')[0]
$failurePart = $eng.Split('### Engineering failure modes registry')[1].Split('### Engineering NOT')[0]
$tasks = @(Get-Content -LiteralPath 'reports/planning/implementation_tasks.json' -Raw | ConvertFrom-Json)
$qaPath = 'C:\Users\vaish\.gstack\projects\vaishak-v-nair-FrameGenX\vaish-main-eng-review-test-plan-20261015-214500.md'
$todo = Get-Content -LiteralPath 'TODOS.md' -Raw
$http = $plan.Split('$base =')[1].Split('```')[0]
$httpCalls = @($http.Split("`n") | Where-Object { $_.Contains('Invoke-RestMethod') })
$originalTools = @('React','TypeScript','Streamlit','FastAPI','PostgreSQL','Redis','LangChain','OpenAI','Gemini','Claude','LangGraph','CrewAI','Google ADK','A2A','LlamaIndex','ChromaDB','RAGAS','Graph RAG','MCP','n8n','ElevenLabs','LangSmith','Phoenix','Opik','PyTorch','PEFT','LoRA/QLoRA','MLflow','Docker','CI/CD')
$missingTools = @($originalTools | Where-Object { -not $plan.Contains($_) })
$check = [ordered]@{}
$check.terminal_review_heading = (([regex]::Matches($plan, '(?m)^## .+'))[-1].Value.Trim() -eq '## GSTACK REVIEW REPORT')
$check.terminal_unresolved_status = ($plan.TrimEnd().Split("`n")[-1].Trim() -eq 'NO UNRESOLVED DECISIONS')
$check.audit_contiguous_74 = ($ids.Count -eq 74 -and ($ids -join ',') -eq ((1..74) -join ','))
$check.audit_classification = ($taste.Count -eq 9 -and $challenges.Count -eq 2 -and $mechanical.Count -eq 63)
$check.ceo_eleven_sections = (@(1..11 | Where-Object { -not $plan.Contains("### 4.$_ ") }).Count -eq 0)
$check.design_seven_passes = ([regex]::Matches($design, '(?m)^### Pass \d:').Count -eq 7)
$check.dx_eight_passes = ([regex]::Matches($dx, '(?m)^### Pass \d:').Count -eq 8)
$check.engineering_four_passes = ([regex]::Matches($eng, '(?m)^### Pass \d:').Count -eq 4)
$check.test_map_32_groups = ([regex]::Matches($testPart, '(?m)^G\d{2} ').Count -eq 32)
$check.test_requirements_32_rows = ([regex]::Matches($testPart, '(?m)^\| G\d{2} \|').Count -eq 32)
$check.failure_registry_32_rows = ([regex]::Matches($failurePart, '(?m)^\| G\d{2} \|').Count -eq 32)
$check.qa_artifact_exists = Test-Path -LiteralPath $qaPath
$missingRecords = @(@('ceo','design','dx','eng') | Where-Object {
    -not ((Test-Path -LiteralPath "reports/planning/$($_)_codex.md") -and
          (Test-Path -LiteralPath "reports/planning/$($_)_subagent.md"))
})
$check.all_review_records_exist = ($missingRecords.Count -eq 0)
$check.aggregate_30_four_phases = ($tasks.Count -eq 30 -and @($tasks.phase | Sort-Object -Unique).Count -eq 4)
$check.aggregate_current_run_only = (@($tasks | Where-Object run_id -NE '20261015-framegenx-stack').Count -eq 0)
$check.conditional_todos_six = ([regex]::Matches($todo, '(?m)^### F[1-6] ').Count -eq 6)
$check.submission_lookup_correct = ($plan.Contains('/api/v1/submissions?request_key=') -and -not $plan.Contains('GET /api/v1/jobs?request_key='))
$check.http_all_calls_authenticated = ($httpCalls.Count -eq 6 -and @($httpCalls | Where-Object { -not $_.Contains('-Headers') }).Count -eq 0)
$check.explicit_v1_contract_required = ($plan.Contains('a missing/empty contract returns 422') -and $plan.Contains('never extracts or changes intent after confirmation'))
$check.approval_has_receipt = ($plan.Contains('Status: APPROVED ROADMAP WITH RUNTIME GATES') -and
    (Test-Path -LiteralPath 'reports/planning/APPROVAL_2026-10-16.md'))
$execution = @(Get-Content -LiteralPath 'reports/planning/EXECUTION_TASKS.json' -Raw | ConvertFrom-Json)
$check.execution_tasks_separate_states = ($execution.Count -eq 43 -and @($execution | Where-Object {
    -not ($_.planning.status -and $_.execution.status -and $_.verification.status -and $_.acceptance.status)
}).Count -eq 0)
$check.original_proposal_retained = ($plan.Contains('## Original proposal') -and $missingTools.Count -eq 0)
$failed = @($check.GetEnumerator() | Where-Object { -not $_.Value } | Select-Object -ExpandProperty Key)
$record = [ordered]@{
    generated_at = (Get-Date -Format o)
    scope = 'Planning package verification only; no application test or quality qualification'
    status = $(if ($failed.Count) { 'FAILED' } else { 'PASSED_PLANNING_ARTIFACT_CHECKS' })
    plan = $planPath
    plan_sha256 = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash
    decisions = @{ total=$audit.Count; mechanical=$mechanical.Count; taste=$taste.Count; challenges=$challenges.Count }
    tasks = @{ total=$tasks.Count; phases=@($tasks | Group-Object phase | ForEach-Object { @{phase=$_.Name;count=$_.Count} }) }
    checks = $check
    failed = $failed
    qa_artifact = $qaPath
    verification_attempts = @(
        'Initial inline validator had CRLF and PowerShell pipeline-coercion errors; application was not executed.',
        'This fail-fast script corrects those validator errors and reruns all planning-artifact checks.'
    )
    limitations = @(
        'This command checks planning artifacts only; implementation evidence is recorded separately',
        'This command does not test the application, clean install, CI, browser, accessibility or GPU workload',
        'No calibrated hallucination reduction verified',
        'Review instances are same-family Codex; CLI document-only',
        'User A approved the staged roadmap; approval does not clear runtime gates'
    )
}
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath 'reports/planning/approved_package_verification.json' -Encoding utf8
[pscustomobject]@{ status=$record.status; checks=$check.Count; failed=$failed; decisions=$record.decisions; tasks=$tasks.Count } | ConvertTo-Json -Depth 5
if ($failed.Count) { throw 'Planning artifact checks failed.' }
