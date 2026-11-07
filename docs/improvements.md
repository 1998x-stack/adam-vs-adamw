# Portfolio engineering improvement history

### 2026-11-07 00:15 UTC — ChatGPT Portfolio Iteration
- Run ID: portfolio-20261107T001533Z-a1bf3d
- Repo / Base SHA: 1998x-stack/adam-vs-adamw / fe10b7d7e210ce3b1f6a6ea0a67e9fffbe2e1a5e
- Historical context: Default branch docs/improvements.md returned HTTP 404 (first initialization); two baseline commits and prior merged PR #1 inspected; no matching open PR found. Preserve existing optimizer semantics, experimental splits, and CLI defaults.
- Product problem & expected user value: Invalid --seeds input should produce actionable argparse diagnostics rather than an uncaught ValueError or downstream PyTorch seed failure.
- Engineering diagnosis & evidence: adamw_lab/run.py parse_args converted comma-separated seeds with int() without handling ValueError and accepted negative seeds. Reproduction inputs: abc, 42,, -1.
- Options & decision: CLI seed validation scored 17 (benefit 4 + correctness 4 + evidence 5 + testability 5 - cost 1 - risk 0); finite LR/decay validation and output-directory collision safeguards deferred because they affect broader behavior and need separate tests.
- Changes: adamw_lab/run.py catches invalid seed tokens and rejects negative seeds using argparse error code 2; tests/test_cli.py adds ten valid/invalid cases; existing valid seed syntax and optimizer API unchanged.
- Validation: A (isolated reconstructed CLI parser, not full checkout): PYTHONPATH=. python -m pytest tests/test_cli.py -q, 10 passed in 0.14s on Python sandbox. D: committed repository test suite and CI not executed locally because GitHub clone DNS failed. E: no end-to-end training run.
- Security & regression review: No secrets, workflows, deployment, permissions, dataset or optimizer updates; invalid input fails before output creation. Full repository regression unverified.
- Risk & rollback: Low CLI compatibility risk; restore prior run.py and remove new test/history entry through a normal revert PR if needed. Do not merge before repository CI and required checks are verified.
- Follow-up: Run full pytest and a quadratic CPU smoke test from a real checkout, then verify branch protection and required checks.
- Commit: pending
- PR: pending
