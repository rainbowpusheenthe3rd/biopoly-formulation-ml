# Contributing

## Workflow
- `main` is protected; work on feature branches and merge on completion.
- Install hooks once: `uv run pre-commit install && uv run pre-commit install --hook-type commit-msg`
- Lint/test locally: `uv run ruff check . && BIOPOLY_TRACKING_BACKEND=noop uv run pytest`

## Commit convention
`type(scope): subject` — enforced by [`scripts/check_commit_msg.py`](scripts/check_commit_msg.py).

Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`, `ci`.
Example scopes: `modelling`, `server`, `cicd`, `data`, `inverse`.

```
feat(modelling): add quantile forward model
fix(server): return 422 on unknown target key
chore(cicd): pin uv to 0.11.8
```
