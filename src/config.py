from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def path(p):
    return ROOT / p
