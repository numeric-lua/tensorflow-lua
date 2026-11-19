# API stability and versioning

tensorflow-lua follows semantic versioning for the Lua-facing API.

## Public surface

The public surface is what the README and `docs/api.md` document:

- `tf.Tensor`, `tf.Graph`, `tf.Session`, `tf.CheckpointReader`;
- graph-building helpers (`tf.placeholder`, `tf.constant`, `tf.variable`,
  arithmetic, reductions, `tf.gradients`, gradients/optimizers);
- `tf.layers` functions documented in `docs/api.md`;
- `tf.Sequential` methods documented in `docs/api.md`;
- `tf.Functional` methods and constructor options documented in `docs/api.md`;
- `tf.losses`, `tf.metrics`, `tf.callbacks`, `tf.data`, `tf.schedules`;
- SavedModel loading, checkpoint save/restore, and GraphDef import/export.

Names not in those documents are internal. The `_` prefix marks private fields.

## Stability rules

- **Major** versions may remove or change documented APIs.
- **Minor** versions add APIs and may add fields; they do not remove documented
  functions or silently change required arguments.
- **Patch** versions fix behavior without changing the documented surface.
- Experimental features are explicitly marked in `docs/api.md`; they may change
  in minor releases.

## Known platform limits

- Linux x86-64 with libtensorflow 2.16.2 or 2.18.0 is the tested matrix.
- CUDA/GPU placement is not implemented.
- Official SavedModel export is not implemented because the TensorFlow C API
  does not expose the MetaGraphDef writer. GraphDef export plus native
  checkpoints is the supported deployment path.
- The native C library has no gradient for `Conv3D`; the layer composes
  portable Conv2D kernels and is trainable but slower than a fused kernel.
- Deployment uses the portable bundle format (`graph.pb` + native V2 checkpoint
  + manifest); official SavedModel export is unavailable.
- LSTM/GRU layers unroll a static time dimension at graph-construction time.
- AUC uses an O(N^2) pairwise graph and is intended for moderate evaluation
  batches.

## Deprecation

A documented API is deprecated for at least one minor release before removal.
Deprecated paths remain functional and are called out in `CHANGELOG.md`.
