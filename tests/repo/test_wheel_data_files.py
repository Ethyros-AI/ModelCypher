from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


def test_wheel_includes_domain_taxonomy_yaml(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["poetry", "build", "-f", "wheel", "-o", str(tmp_path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "modelcypher/data/domain_taxonomy.yaml" in archive.namelist()
