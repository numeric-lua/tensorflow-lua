# Performance notes

`scripts/bench.sh` is a small, dependency-free benchmark harness. It is a
smoke benchmark, not a CI gate: this machine is thermally throttled and the
numbers move by 10–20% between runs.

## 0.6.0 baseline

Environment:

- LuaJIT 2.1
- libtensorflow 2.18.0 CPU
- Linux x86-64
- small synthetic tensors and a 4→8→1 Sequential model

Latest local run:

```text
tensor create/totable        2000 iterations   113.745 ms   17583.2 ops/s
session add                  1000 iterations    84.967 ms   11769.3 ops/s
model predict x1000          1000 iterations   113.240 ms    8830.8 ops/s
model train epoch x20          20 iterations    56.583 ms     353.5 ops/s
```

The native tensor constructor now fills numeric and boolean tensors directly
instead of materialising an intermediate flat Lua table; the change is small
but measurable at this allocation size. Larger gains will require reducing
feed/fetch setup and repeated batch conversion rather than micro-optimising the
FFI calls.

Run:

```sh
TENSORFLOW_ROOT="$PWD/third_party/libtensorflow" ./scripts/bench.sh
```
