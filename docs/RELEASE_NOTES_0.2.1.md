# UniCell 0.2.1 release notes

## scFoundation batching repair

`HMCNDataset` now converts each scFoundation expression row to a tensor before
the training and inference loops call `torch.stack`. Previously, these rows
were pandas `Series`, causing `TypeError: expected Tensor as element 0 in
argument 0, but got Series` when processing a batch.

This fix concerns the input representation for the scFoundation branch. It
does not change the expression, GeneFormer, or scGPT input paths, model
architecture, losses, optimizer settings, or checkpoint format. The packaged
`unicell-train-backbone` command continues to support `expr`, `GeneFormer`, and
`scGPT`; scFoundation remains available through the Python API and tutorial.

## Release contents and documentation

- The package, build scripts, installer, and current usage examples use version
  0.2.1. English and Chinese README files and usage guides are retained.
- The scFoundation tutorial and trainer API reference describe the repaired
  batching interface and the remaining validation limits.
- The offline build guide lists the required Python build tools, seven
  checkpoint files, and cluster-specific ARM wheels.
- The expression checkpoint archive and its manifest are unchanged from
  v0.2.0. The checkpoint archive is downloaded directly from the existing
  v0.2.0 asset instead of being uploaded again. Its original provenance remains
  in the manifest; the new code
  release does not imply retraining or new model weights. The GitHub release
  provides checksums for its actual assets.

## Third-party licensing status

All SCAD code and the existing licensing notices are retained unchanged. The
SCAD code redistribution grant remains undocumented in the checked upstream
sources. Retaining this disclosure does not supply permission or establish
that redistribution is authorized; readers should consult the upstream
project and rights holders for clarification.

The original UniCell contributions retain their MIT license. The complete
distribution also includes code and data under other terms, including GPL v3,
Apache-2.0, MIT/BSD, and CC BY 4.0. It is not an MIT-only distribution. Existing
license copies, attribution, and the unresolved SCAD/DSBN documentation notes
remain in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Validation limits

Fixing the batching error does not establish that the entire scFoundation
tutorial, every supported platform, or the paper's experiments have been
reproduced. Full training and paper-result reproduction have not been rerun
for this release. The GitHub release records the specific checks performed
and the corresponding source commit; these checks should be read within
their stated scope.
