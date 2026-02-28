from __future__ import annotations

import importlib
from importlib import metadata


CORE_DEPS = [
    "numpy",
    "pandas",
    "scipy",
    "astropy",
    "astroquery",
    "matplotlib",
    "plotly",
    "poliastro",
]


def main() -> int:
    failures: list[tuple[str, Exception]] = []
    for name in CORE_DEPS:
        try:
            importlib.import_module(name)
            version = metadata.version(name)
            print(f"{name}: {version}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"{name}: FAILED ({exc})")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
