# tensorflow-lua agent notes

- Scope: native LuaJIT bindings to the TensorFlow C API. This is not a second
  tensor engine; all execution is TensorFlow's.
- Keep the first milestone focused on GraphDef/SavedModel inference: tensors,
  graph import, sessions, feeds/fetches, dtypes, shapes, and status errors.
- Do not vendor libtensorflow. Fetch it with `scripts/fetch_libtensorflow.sh`.
- No releases, tags, or package publishes without the coordinator.
- Every behavior change needs a test; tests must run without Python. Small
  pre-generated SavedModel/GraphDef fixtures belong under `tests/fixtures/`.
- C API calls must check `TF_Status`; every created API object needs a
  deterministic delete path or `__gc`.
