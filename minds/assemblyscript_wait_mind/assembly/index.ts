import { Host } from "@extism/as-pdk";

// Canonical Cap'n Proto encoding of ReferenceMindDecision {
//   action: Wait,
//   signal: None,
//   memory_update: Retain,
// }.
//
// Keeping this deliberately small makes the artifact a useful cross-language
// ABI and deterministic-PDK canary. A behavior-rich AssemblyScript Mind can
// layer generated schema bindings on the same byte contract later.
const WAIT_RETAIN_DECISION: u8[] = [
  0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0,
  1, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 2, 0, 1, 0,
  17, 0, 0, 0, 2, 0, 0, 0, 12, 0, 0, 0, 1, 0, 1, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
];

export function myAbort(
  _message: string | null,
  _fileName: string | null,
  _lineNumber: u32,
  _columnNumber: u32,
): void {}

export function reference_mind_function(): i32 {
  // Exercise the official PDK input surface. The policy intentionally ignores
  // observation contents and always waits.
  const input = Host.input();
  if (input.length == 0) return 1;

  const output = new Uint8Array(WAIT_RETAIN_DECISION.length);
  for (let index = 0; index < WAIT_RETAIN_DECISION.length; index++) {
    output[index] = WAIT_RETAIN_DECISION[index];
  }
  Host.output(output);
  return 0;
}
