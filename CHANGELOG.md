# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2024-04-15

### Added

- Added standard deviation to smoother output.

## [1.2.0] - 2024-04-12

### Added

- Added `inverse` kernel that implements inverse-distance weights.
- Added `down_weight` argument to smoother to change relative importance of neighbors
  for in-sample points.

## [1.1.2] - 2024-02-06

### Fixed

- Added keyword to `df.any()` commands in `Smoother.check_data` for new pandas versions.

## [1.1.1] - 2023-03-14

### Removed

- Removed requirement that `key[0] <= key[1]` in `distance_dict` and updated documentation.

### Fixed

- Fixed problem in `Smoother.check_data` where dimension names were being overwritten by `distance_dict` keys.

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

### Changed

- Moved setup info from setup.py to setup.cfg
- Moved source code from weave/ to src/weave/
- Moved tests from weave/tests/ to tests/

[1.3.0]: https://github.com/ihmeuw-msca/weighted-average/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/ihmeuw-msca/weighted-average/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/ihmeuw-msca/weighted-average/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/ihmeuw-msca/weighted-average/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/ihmeuw-msca/weighted-average/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ihmeuw-msca/weighted-average/releases/tag/v1.0.0
