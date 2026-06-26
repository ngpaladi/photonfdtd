"""Guard that every declared version number stays in sync.

The package version is declared in two machine-readable places:

* ``pyproject.toml``           -> ``[project].version`` (what gets published)
* ``src/photonfdtd/__init__`` -> ``__version__`` (what ``import`` reports)

These must always agree. (The ``v0.1``/``v0.2`` strings scattered through the
README and module docstrings are descriptive prose about when features
landed, not version declarations, and are intentionally not checked here.)

The release-vs-PyPI correspondence is enforced separately, in the publish
workflow (``.github/workflows/publish.yml``), which can only be checked when a
release is actually cut.
"""
import re
import sys
from pathlib import Path

import photonfdtd

# Repo root resolved relative to THIS file, not the working directory, so the
# test passes regardless of where pytest is invoked from.
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    # Prefer a real TOML parse when the stdlib provides one (Python 3.11+).
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)["project"]["version"]
    # Fallback for 3.10: grab the first `version = "..."` in the [project] table.
    project = text.split("[project]", 1)[1]
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project, re.MULTILINE)
    assert match, "could not find [project].version in pyproject.toml"
    return match.group(1)


def test_pyproject_matches_dunder_version():
    pyproject = _pyproject_version()
    assert pyproject == photonfdtd.__version__, (
        f"version skew: pyproject.toml says {pyproject!r} but "
        f"photonfdtd.__version__ is {photonfdtd.__version__!r}. "
        "Bump both together."
    )
