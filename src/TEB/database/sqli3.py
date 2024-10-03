import sqlite3

class Database:
    """Database class to manage electrical load data."""

    def __init__(self, db_name='loads.db'):
        self.connection = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        """Create tables for average power consumption and user inputs."""
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS average_power (
                    id INTEGER PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    average_power_consumption REAL NOT NULL
                )
            ''')
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS user_inputs (
                    id INTEGER PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    usage_hours_per_day REAL NOT NULL,
                    frequency_per_week INTEGER NOT NULL
                )
            ''')

    def insert_average_power(self, device_name, average_power):
        """Insert average power consumption for a device."""
        with self.connection:
            self.connection.execute('''
                INSERT INTO average_power (device_name, average_power_consumption)
                VALUES (?, ?)
            ''', (device_name, average_power))

    def insert_user_input(self, device_name, usage_hours, frequency):
        """Insert user input for device usage."""
        with self.connection:
            self.connection.execute('''
                INSERT INTO user_inputs (device_name, usage_hours_per_day, frequency_per_week)
                VALUES (?, ?, ?)
            ''', (device_name, usage_hours, frequency))

    def close(self):
        """Close the database connection."""
        self.connection.close()

