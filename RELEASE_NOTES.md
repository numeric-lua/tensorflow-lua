# tensorflow-lua 0.19.0

Native LuaJIT bindings for the TensorFlow C API.

## Highlights

- trainable UpSampling3D for NDHWC volumes, implemented with
  reshape/tile/reshape;
- Sequential shape tracking and summaries for 3-D upsampling;
- 173 dependency-free tests.

The 0.18 feature set remains: tensor-backed datasets, attention/transformer
layers, trainable composed Conv3D, 3-D pooling, functional-model checkpointing,
gradient accumulation, AUC/macro metrics, and native LR schedules.

## Install

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
luarocks --lua-version=5.1 make tensorflow-lua-0.19.0-1.rockspec
```
