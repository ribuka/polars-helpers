# instructions

- A repository for polars-helpers: small helpers for polars DataFrames and LazyFrames, used as `import polars_helpers as plh`.
- Always respond in Japanese.
- This repository is in development (0.x); backward compatibility is NOT required.
  - NEVER add or keep compatibility layers, deprecated aliases, or re-exports for old names.
  - When the public API (`__all__` of `polars_helpers`) changes, update `README.md` and the tests in the same change.

## Package rules

- The package imports only the standard library and polars.
- Every helper accepts both `pl.DataFrame` and `pl.LazyFrame` where it can, and never collects a whole `LazyFrame` just to read its schema or shape.
- Function names do not repeat "polars" (e.g. `get_shape`, not a `get_shape_...polars` variant); the package name already says they are for polars.
- `polars_helpers.utils` holds helpers that do not depend on polars (e.g. `natural_keys`); they are not re-exported from the top level.
- The lower bound of `polars` in `pyproject.toml` is the oldest version the CI `test-lowest` job tests. Raise it when a helper starts to need a newer polars API.

## Python

- Always use `uv run -m`.
- NEVER use `python -m`.
- Add NumPy-style docstrings and type hints to every function and class.
- Keep one primary responsibility per module and keep `__init__.py` a thin, explicit re-export of the public API.
- NEVER introduce circular imports.

### Verification commands

- Run lint: `uv run -m ruff check .`
- Run tests: `uv run -m pytest --tb=short -q`
- CI runs both on every pull request to main, and the tests again with the lowest allowed polars.

## Github

### Commit

- Write the subject line as `<type>: <summary>`, with `<summary>` in Japanese and code identifiers kept as-is.
- Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `style`, `build`, `ci`, `chore`, `revert`.
- NEVER include session URLs in commit messages or pull request descriptions.

### Applications that use this repository

- Applications depend on this repository through a git source pinned by their `uv.lock`.
- A change that an application needs is merged here first; the application then runs `uv lock --upgrade-package polars-helpers` in its own pull request.

## Temporary files

- Create all temporary scripts and investigation files under `tmp/` at the repository root, and delete them when the task is complete.
