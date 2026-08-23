# Go conformance Mind

This maintained canary uses the official Extism Go PDK and always returns
canonical `Wait` with retained private memory. It uses TinyGo's bare
`wasm-unknown` target rather than the PDK's more typical WASI target, keeping
the resulting module inside Blob's deterministic capability profile. The
conformance build currently pins TinyGo 0.41.1:

```sh
tinygo build -target=wasm-unknown -buildmode=c-shared -no-debug \
  -o build/go_wait_mind.wasm .
```

The standard Go `GOOS=wasip1` build is intentionally not accepted because it
imports WASI. Both supported executors differentially test this bare artifact.
