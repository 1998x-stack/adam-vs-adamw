"""Regression tests for CLI seed validation."""

import pytest

from adamw_lab.run import parse_args


@pytest.mark.parametrize(
    "seeds,expected",
    [("42", [42]), ("0, 42", [0, 42]), ("42,43,44", [42, 43, 44])],
)
def test_valid_seeds(seeds, expected):
    assert parse_args(["--experiment", "quadratic", "--seeds", seeds]).seeds == expected


@pytest.mark.parametrize("seeds", ["", " ", "abc", "42,bad", "42,", "-1", "1,-2"])
def test_invalid_seeds_use_argparse_error(seeds, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--experiment", "quadratic", "--seeds", seeds])
    assert exc.value.code == 2
    assert "--seeds must be comma-separated nonnegative integers" in capsys.readouterr().err
