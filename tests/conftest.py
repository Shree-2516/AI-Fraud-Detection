import pytest
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
