package main

import "github.com/extism/go-pdk"

var waitRetainDecision = []byte{
	0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0,
	1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 1, 0,
	17, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 1, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,
}

//go:wasmexport reference_mind_function
func referenceMindFunction() int32 {
	// Exercise the official PDK input surface. The policy intentionally ignores
	// observation contents and always waits.
	if len(pdk.Input()) == 0 {
		return 1
	}
	pdk.Output(waitRetainDecision)
	return 0
}

func main() {}
