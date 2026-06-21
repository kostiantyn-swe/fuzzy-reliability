"""Одноразово перенести існуючі сценарії з saved/*.json у saved/anonymous/*.json."""
from pathlib import Path
import shutil

SCENARIOS_DIR = Path(__file__).parent.parent / "saved"
ANON_DIR = SCENARIOS_DIR / "anonymous"
ANON_DIR.mkdir(exist_ok=True)

moved = 0
for fp in SCENARIOS_DIR.glob("*.json"):
    if fp.is_file():
        shutil.move(str(fp), str(ANON_DIR / fp.name))
        moved += 1

print(f"Переміщено {moved} файлів у saved/anonymous/")
