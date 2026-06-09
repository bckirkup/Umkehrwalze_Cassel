# Contributing

## Setup

```bash
git clone https://github.com/bckirkup/Umkehrwalze_Cassel.git
cd Umkehrwalze_Cassel
pip install -e ".[dev]"
```

## Development workflow

1. Create a feature branch from `main`.
2. Make changes.
3. Run linting and tests:
   ```bash
   ruff check src tests
   pytest
   ```
4. Commit and push. CI runs on Python 3.10 and 3.11 across Ubuntu and Windows.
5. Open a PR against `main`.

## Code conventions

- **Style**: enforced by `ruff` (pycodestyle E, pyflakes F, isort I). Line length limit is 100 characters.
- **Type hints**: use throughout; prefer `X | None` over `Optional[X]`.
- **Imports**: stdlib → third-party → local, separated by blank lines. `from __future__ import annotations` at the top of every module.
- **GPU code**: guard behind `torch.cuda.is_available()` so CI (CPU-only) never breaks.
- **Tests**: put under `tests/`, prefix with `test_`. Use `pytest` with `--no-cov` during development; CI enforces a 70% coverage threshold.

## Project layout

```
src/revprint/    # main package (installed as editable)
tests/           # pytest test suite
docs/            # design documents and specs
inputs/          # sample data (not committed to git for large corpora)
outputs/         # pipeline outputs (gitignored)
```

## Adding a new processing stage

1. Create `src/revprint/your_stage.py` with a single-page entry-point function that takes `(input_path, output_path, **params) -> dict[str, object]` (metadata).
2. Wire it into `batch_pipeline.py` in the appropriate phase order.
3. Add a test in `tests/test_your_stage.py` using synthetic images (no real data dependency).
4. Update the batch pipeline stages table in `README.md`.

## Environment variables

Configuration is managed via `pydantic-settings`.  See `src/revprint/config.py` for all available `RPK_*` variables and their defaults.  Copy `.env.example` to `.env` for local overrides.

## Commit messages

Use imperative mood.  First line ≤ 72 characters summarising the change.  Body is optional but welcome for non-trivial changes.

## Licence

By contributing you agree your work is released under the Apache 2.0 licence (see `LICENSE`).
