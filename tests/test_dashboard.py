import pytest
from electrical_load_dashboard.app import ElectricalLoadDashboard

@pytest.fixture
def app():
    """Fixture for the Dash app."""
    dashboard = ElectricalLoadDashboard()
    yield dashboard

