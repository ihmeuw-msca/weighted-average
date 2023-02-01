# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2023-02-01

### Added

- Added optional argument `stdev` to `Smoother.__call__`

### Removed

- Removed option to smooth multiple columns at once (i.e., `Smoother.__call__`
  arguments `observed` and `smoothed` are now str and no longer str or
  list[str])

## [1.0.0] - 2023-01-06

### Added

- Added smoother, dimension, kernel, and distance modules
- Added tests modules
- Added documentation including user guide and API

## Changed

- Moved source code from weave/ to src/weave/
- Moved tests from weave/tests/ to tests/

[Unreleased]: https://github.com/ihmeuw-msca/weighted-average
[1.1.0]: https://github.com/ihmeuw-msca/weighted-average/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ihmeuw-msca/weighted-average/releases/tag/v1.0.0

