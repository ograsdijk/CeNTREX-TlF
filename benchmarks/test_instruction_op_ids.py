"""Cross-language test: verify InstructionOp integer values match between Python and Rust."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from centrex_tlf.lindblad.ir import InstructionOp

EXPECTED = {
    "CONST": 1,
    "SLOT": 2,
    "TEMP": 3,
    "TIME": 4,
    "ADD": 5,
    "SUB": 6,
    "MUL": 7,
    "DIV": 8,
    "POW": 9,
    "NEG": 10,
    "CONJ": 11,
    "BUILTIN_FUNC": 12,
    "HELPER_FUNC": 13,
    "GT": 14,
    "GE": 15,
    "LT": 16,
    "LE": 17,
    "EQ": 18,
    "NE": 19,
}

errors = []
for name, expected_value in EXPECTED.items():
    actual = getattr(InstructionOp, name, None)
    if actual is None:
        errors.append(f"  MISSING: InstructionOp.{name}")
    elif int(actual) != expected_value:
        errors.append(f"  MISMATCH: InstructionOp.{name} = {int(actual)}, expected {expected_value}")

for member in InstructionOp:
    if member.name not in EXPECTED:
        errors.append(f"  EXTRA: InstructionOp.{member.name} = {int(member)} (not in expected set)")

if errors:
    print("FAILED: InstructionOp value mismatches:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print(f"PASSED: all {len(EXPECTED)} InstructionOp values match")
