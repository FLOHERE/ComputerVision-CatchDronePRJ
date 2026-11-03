from pathlib import Path

import pytest
import yaml


@pytest.fixture
def cfg():
    return yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8"))
