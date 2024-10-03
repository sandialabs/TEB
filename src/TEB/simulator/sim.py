import pandas as pd
import time

class Simulation:
    """Simulation class to run the load calculations."""

    def __init__(self, db):
        self.db = db

    def run_simulation(self):
        """Run the simulation and return a DataFrame of results."""
        results = []
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT * FROM user_inputs")
        user_inputs = cursor.fetchall()

        for device in user_inputs:
            device_name, usage_hours, frequency = device[1], device[2], device[3]
            total_usage = usage_hours * frequency
            results.append({'device_name': device_name, 'total_usage': total_usage})
            time.sleep(0.1)  # Simulate time delay for progress

        return pd.DataFrame(results)

