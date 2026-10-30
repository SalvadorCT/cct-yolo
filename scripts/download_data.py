"""Atajo: python scripts/download_data.py [--what metadata|images|all]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cct.download import main  # noqa: E402

if __name__ == "__main__":
    main()
