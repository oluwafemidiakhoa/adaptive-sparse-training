# Changelog

All notable changes to Adaptive Sparse Training will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-11-08

### Fixed
- **Critical Bug**: Fixed `IndexError: Dimension specified as 0 but tensor has no dimensions` when using `warmup_epochs > 0`
  - Added validation in `SundewAlgorithm.compute_significance()` to detect scalar loss tensors
  - Added validation in `SundewAlgorithm.select_samples()` to detect scalar loss tensors
  - Added warning in `AdaptiveSparseTrainer.__init__()` when criterion doesn't use `reduction='none'`
  - Improved error messages to guide users toward correct loss function configuration

### Changed
- Enhanced error handling to provide clear guidance when loss tensors have incorrect dimensions

## [1.0.0] - 2025-10-28

### Added
- Initial release of Adaptive Sparse Training (AST)
- Sundew algorithm with PI-controlled adaptive sample selection
- Multi-factor significance scoring (loss + entropy)
- Energy tracking and monitoring
- Mixed precision training support (AMP)
- Comprehensive documentation and examples
- Validated on CIFAR-10 (92.40% accuracy, 61% energy savings)
- Validated on ImageNet-100 (92.12% accuracy, 61% energy savings)
