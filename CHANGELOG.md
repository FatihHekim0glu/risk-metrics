# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `drawdown_table`, `sterling_ratio`, and `max_drawdown_duration` no longer raise `AttributeError` on a return series with a plain integer index. Duration now falls back to a bar count when the index is not a `DatetimeIndex`, so the documented `sterling_ratio` example runs.

### Added

- Coverage configuration with a `fail_under = 85` gate and omits for the network data layer and the Streamlit app.

### Changed

- Expanded the test suite to cover validation, error, and edge paths across every module, raising line coverage of the core metrics to roughly 90%.

## [0.1.0] - 2026-05-20

### Added

- Initial project scaffolding: `pyproject.toml` with hatchling backend, src layout, and optional `data`, `dashboard`, and `dev` extras.
- MIT license and project README with quick-start example.
- Continuous integration workflow running ruff, mypy, and pytest across Python 3.10, 3.11, and 3.12.
- Package skeleton at `src/riskmetrics/` with PEP 561 typing marker, project-wide constants (`PERIODS_PER_YEAR = 252`), shared type aliases, and input-validation helpers (`ensure_series`, `align_inner`, `validate_min_obs`).

[Unreleased]: https://github.com/FatihHekim0glu/risk-metrics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/risk-metrics/releases/tag/v0.1.0
