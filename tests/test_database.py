import pytest
from electrical_load_dashboard.database import Database

@pytest.fixture
def db():
    """Fixture for the database."""
    db = Database(':memory:')  # Use in-memory database for testing
    yield db
    db.close()

def test_insert_average_power(db):
    db.insert_average_power('Test Device', 100.0)
    cursor = db.connection.cursor()
    cursor.execute("SELECT * FROM average_power")
    result = cursor.fetchone()
    assert result[1] == 'Test Device'
    assert result[2] == 100.0

def test_insert_user_input(db):
    db.insert_user_input('Test Device', 5.0, 3)
    cursor = db.connection.cursor()
    cursor.execute("SELECT * FROM user_inputs")
    result = cursor.fetchone()
    assert result[1] == 'Test Device'
    assert result[2] == 5.0
    assert result[3] == 3

