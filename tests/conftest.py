from __future__ import annotations

import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Create temp dirs without tempfile.mkdtemp, which can create bad ACLs on Windows."""
    base = ROOT / ".pytest_tmp"
    base.mkdir(exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    path = base / f"{safe_name}-{uuid.uuid4().hex}"
    path.mkdir()

    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
