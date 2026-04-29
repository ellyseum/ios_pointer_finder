"""Shared pytest fixtures for the ios_pointer_finder test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def golden_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "golden"


@pytest.fixture(scope="session")
def native_size():
    """Default model native size (W, H)."""
    from train import NATIVE_H, NATIVE_W

    return (NATIVE_W, NATIVE_H)


@pytest.fixture(scope="session")
def train_size():
    """Default model train size (W, H)."""
    from train import TRAIN_H, TRAIN_W

    return (TRAIN_W, TRAIN_H)
