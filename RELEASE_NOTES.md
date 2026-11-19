# tensorflow-lua 0.20.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- unrolled LSTM for fixed-length `[batch, time, input]` sequences;
- unrolled GRU with input/recurrent gate projections;
- trainable parameters and graph-level gradients;
- final hidden/cell states and full output sequences;
- 174 dependency-free tests.

These layers unroll the time dimension at graph construction time, so the
sequence length is static. They are composed from portable TensorFlow ops and
do not depend on a native recurrent kernel.

The 0.19 feature set remains: 3-D convolution/pooling/upsampling,
attention/transformer layers, tensor-backed datasets, functional-model
checkpointing, gradient accumulation, AUC/macro metrics, and native schedules.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.20.0-1.rockspec
```
