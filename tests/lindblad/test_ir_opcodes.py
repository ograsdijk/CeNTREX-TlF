import pytest

from centrex_tlf.lindblad.ir import InstructionOp, BuiltinFunctionId

EXPECTED_OPCODES = {
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

EXPECTED_BUILTINS = {
    "SIN": 1,
    "COS": 2,
    "TAN": 3,
    "EXP": 4,
    "ABS": 5,
    "REAL": 6,
    "IMAG": 7,
}


@pytest.mark.parametrize("name,value", EXPECTED_OPCODES.items())
def test_instruction_op_values_match_rust(name, value):
    assert InstructionOp[name] == value


@pytest.mark.parametrize("name,value", EXPECTED_BUILTINS.items())
def test_builtin_function_id_values_match_rust(name, value):
    assert BuiltinFunctionId[name] == value


def test_instruction_op_has_no_extra_members():
    assert set(InstructionOp.__members__.keys()) == set(EXPECTED_OPCODES.keys())


def test_builtin_function_id_has_no_extra_members():
    assert set(BuiltinFunctionId.__members__.keys()) == set(EXPECTED_BUILTINS.keys())
