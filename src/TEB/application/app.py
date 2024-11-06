import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
from TEB.database.sqli3 import Database
from TEB.simulator.sim import TieredAnalysis

class ElectricalLoadDashboard:
    """Main application class for the dashboard."""

    def __init__(self):
        self.app = dash.Dash(__name__)
        self.db = Database()
        self.simulation = TieredAnalysis(self.db)
        self.app.layout = self.create_layout()
        self.register_callbacks()

    def create_layout(self):
        """Create the layout for the dashboard."""
        return html.Div([
            html.H1("Electrical Load Dashboard"),
            dcc.Input(id='device-name', type='text', placeholder='Device Name'),
            dcc.Input(id='usage-hours', type='number', placeholder='Usage Hours/Day'),
            dcc.Input(id='frequency', type='number', placeholder='Frequency/Week'),
            html.Button('Submit', id='submit-button'),
            html.Button('Run TieredAnalysis', id='run-simulation'),
            dcc.Loading(id="loading", children=[html.Div(id='output')]),
            dcc.Graph(id='usage-graph'),
            dcc.Progress(id='progress-bar', value=0, max=100)
        ])

    def register_callbacks(self):
        """Register callbacks for user interactions."""
        @self.app.callback(
            Output('output', 'children'),
            Input('submit-button', 'n_clicks'),
            State('device-name', 'value'),
            State('usage-hours', 'value'),
            State('frequency', 'value'),
        )
        def submit_data(n_clicks, device_name, usage_hours, frequency):
            if n_clicks is None:
                return ""
            self.db.insert_user_input(device_name, usage_hours, frequency)
            return f"Inserted {device_name} with {usage_hours} hours/day and {frequency} times/week."

        @self.app.callback(
            Output('usage-graph', 'figure'),
            Output('progress-bar', 'value'),
            Input('run-simulation', 'n_clicks'),
        )
        def run_simulation(n_clicks):
            if n_clicks is None:
                return {}, 0
            df = self.simulation.run_simulation()
            fig = px.bar(df, x='device_name', y='total_usage', title='Total Usage per Device')
            return fig, 100  # Simulate progress completion

    def run(self):
        """Run the Dash application."""
        self.app.run_server(debug=True)

if __name__ == '__main__':
    dashboard = ElectricalLoadDashboard()
    dashboard.run()

