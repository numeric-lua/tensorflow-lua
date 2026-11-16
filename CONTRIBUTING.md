# Contributing

## Requirements

- LuaJIT 2.1
- libtensorflow 2.18 CPU (see `scripts/fetch_libtensorflow.sh`)
- `luacheck` for style checks

## Checks

```sh
./scripts/fetch_libtensorflow.sh
./scripts/test.sh
./scripts/lint.sh
```

Tests are dependency-free and must not require Python or network access once the
library is present. Add a regression or error-path test for every behavior
change.

## Constraints

- Keep the binding on the public TensorFlow C API. Do not vendor
  libtensorflow or add a second numeric engine.
- Check every `TF_Status`; use `ffi.gc` for handles and explicit close/destroy
  paths where available.
- Do not create releases, tags, or package publishes without coordinator
  approval.
- Keep LuaJIT 2.1 portability: do not use Lua 5.2+ `//`, bitwise operator
  syntax, or `goto`; use LuaJIT's `bit` library.
