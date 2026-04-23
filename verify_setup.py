#!/usr/bin/env python3
"""Quick checks that the panocount_gui environment can run the Analyzer."""
from __future__ import annotations

import sys


def main() -> int:
    print("PanoCount Analyzer — environment verification")
    print("-" * 50)

    if sys.platform == "win32":
        import os

        print("\nDLL path checks:")
        conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
        paths_to_check = [
            ("Library\\bin", os.path.join(conda_prefix, "Library", "bin")),
            (
                "torch\\lib",
                os.path.join(conda_prefix, "Lib", "site-packages", "torch", "lib"),
            ),
            (
                "mingw-w64\\bin",
                os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
            ),
        ]
        for name, path in paths_to_check:
            exists = os.path.isdir(path)
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {name}: {path}")

        nvfuser = os.path.join(
            conda_prefix,
            "Lib",
            "site-packages",
            "torch",
            "lib",
            "nvfuser_codegen.dll",
        )
        print("\nnvfuser_codegen.dll:")
        if os.path.exists(nvfuser):
            print("  [WARN] nvfuser_codegen.dll present")
            print("         This may cause WinError 182")
            print(r"         Run: .\delete_nvfuser.bat")
        else:
            print("  [OK]   nvfuser_codegen.dll absent")
            print("         (No nvfuser DLL conflict risk)")

    print("\nImports:")
    try:
        import torch

        print(f"  [OK] torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"  [FAIL] torch: {e}")
        return 1

    try:
        from PySide6.QtWidgets import QApplication

        print("  [OK] PySide6")
    except Exception as e:
        print(f"  [FAIL] PySide6: {e}")
        return 1

    try:
        import cv2

        print(f"  [OK] cv2 {cv2.__version__}")
    except Exception as e:
        print(f"  [FAIL] cv2: {e}")
        return 1

    try:
        import numpy

        print(f"  [OK] numpy {numpy.__version__}")
    except Exception as e:
        print(f"  [FAIL] numpy: {e}")
        return 1

    try:
        import matplotlib

        print(f"  [OK] matplotlib {matplotlib.__version__}")
    except Exception as e:
        print(f"  [FAIL] matplotlib: {e}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
