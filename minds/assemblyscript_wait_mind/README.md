# AssemblyScript conformance Mind

This maintained artifact uses the official Extism AssemblyScript PDK and the
canonical Blob Mind ABI. It intentionally always returns `Wait` with retained
private memory. Its purpose is to catch language-PDK, artifact-admission, and
executor compatibility regressions without adding policy complexity.

Build it with:

```sh
npm install --prefix minds/assemblyscript_wait_mind
npm run --prefix minds/assemblyscript_wait_mind build
```

The output is `build/assemblyscript_wait_mind.wasm`. Both the stock Extism and
restricted compatible executors must admit it and produce identical canonical
commitments and state hashes. `package-lock.json` pins the AssemblyScript
compiler and official PDK dependency used by conformance.
