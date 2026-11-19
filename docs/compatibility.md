# libtensorflow compatibility

tensorflow-lua binds to the TensorFlow C API. A version is supported only when
Google publishes an official `libtensorflow` C archive for Linux x86-64 and the
full suite has been exercised against that archive.

## Compatibility matrix

| libtensorflow | Linux x86-64 archive | Verification | Status |
| --- | --- | --- | --- |
| 2.16.2 | published, checksum pinned | full suite; CI Ubuntu 22.04/24.04 | supported |
| 2.18.0 | published, checksum pinned | full suite; CI Ubuntu 22.04/24.04 | supported |
| 2.18.1 | published, checksum pinned | full suite; CI Ubuntu 22.04/24.04 | supported |
| 2.19.x | not published | not run | unsupported |
| 2.20.x | not published | not run | unsupported |

The CI declaration is in `.github/workflows/test.yml`. The fetch script pins
the SHA-256 of every supported Linux x86-64 CPU archive.

## Why 2.19 and 2.20 are not supported

Google's [TensorFlow C installation guide](https://www.tensorflow.org/install/lang_c)
states: "TensorFlow 2.18 was the last release of Linux x86 libtensorflow
packages." The official GCS prefixes for
`versions/2.19.0/` and `versions/2.20.0/` contain Python wheels but no
`libtensorflow-cpu-linux-x86_64.tar.gz` archive. The same is true for the 2.19
and 2.20 patch releases and release candidates checked during this task.

Because no official 2.19/2.20 Linux x86-64 archive was available, the suite
could not be run against those versions. This project therefore makes no
support claim for them:

```console
$ TENSORFLOW_VERSION=2.19.0 ./scripts/fetch_libtensorflow.sh
libtensorflow 2.19.0 is not available for linux-x86_64.
...
Supported Linux x86-64 versions: 2.16.2, 2.18.0, 2.18.1.
```

The fetch script rejects 2.19/2.20 before downloading and rejects other
unpinned versions unless `TENSORFLOW_SHA256` explicitly verifies the build.
`./tests/test_fetch_libtensorflow.sh` covers these diagnostics.

## Operation availability

The following areas are exercised by the full suite on every supported archive.
No differences were observed among 2.16.2, 2.18.0, and 2.18.1.

| Area | Operations used | Notes |
| --- | --- | --- |
| SeparableConv2D | `DepthwiseConv2dNative`, `Conv2D` | The fused `SeparableConv2D` op is not registered by the C library on supported hosts, so the layer composes depthwise and pointwise convolutions. Both component ops have gradients. |
| 1-D/2-D pooling | `MaxPool`, `AvgPool` | Used directly for 2-D pooling and via expand/squeeze for 1-D pooling. |
| 3-D pooling | `MaxPool3D`, `AvgPool3D` | Also exercised through global 3-D pooling layers. |
| Checkpoints | `SaveV2`, `RestoreV2`, `TF_NewCheckpointReader` | Native V2 save/restore and reading official TensorFlow checkpoints are covered. |

For 2.19 and 2.20 these operation checks were not run because no official
archive exists. If Google ever resumes Linux x86-64 C archives, add the archive
checksum, add the version to the CI matrix, run the full suite, and update this
table before declaring support.
