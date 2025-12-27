import threading
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
import time

import logging


class LivePointCloud:
    def __init__(self, port=8050):
        self.points = np.empty((0, 3))
        self.colors = np.empty((0, 3))  # RGB per point
        self.xlim = (-8000, 8000)
        self.ylim = (-8000, 8000)
        self.zlim = (-8000, 8000)
        self.port = port

        # Dash app
        self.app = Dash(__name__)
        self.app.layout = html.Div([
            dcc.Graph(id='pointcloud', figure=self._create_figure()),
            dcc.Interval(id='update-interval', interval=1000, n_intervals=0)  # updates every 100ms
        ])

        # Dash callback to update the figure
        @self.app.callback(
            Output('pointcloud', 'figure'),
            Input('update-interval', 'n_intervals')
        )
        def update_figure(n):
            return self._create_figure()

        # Start Dash server in background thread
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

        # Give server a moment to start
        time.sleep(1)

    def _run_server(self):
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(debug=False, port=self.port, use_reloader=False)

    def _create_figure(self):
        # FLIPPED Y AND Z DIMENSIONS

        if len(self.points) > 0:
            color_strings = [
                f"rgb({int(r)},{int(g)},{int(b)})"
                for r, g, b in self.colors
            ]
        else:
            color_strings = []

        fig = go.Figure(
            data=[go.Scatter3d(
                x=-self.points[:, 0] if len(self.points) > 0 else [],
                y=self.points[:, 2] if len(self.points) > 0 else [],
                z=self.points[:, 1] if len(self.points) > 0 else [],
                mode='markers',
                marker=dict(
                    size=3,
                    color=color_strings
                )
            )],
            layout=go.Layout(
                width=960,
                height=720,
                scene=dict(
                    xaxis=dict(range=self.xlim, title='X'),
                    yaxis=dict(range=self.ylim, title='Z'),
                    zaxis=dict(range=self.zlim, title='Y'),
                    aspectmode='cube'
                ),
                dragmode='lasso',
                uirevision='lock'
            )
        )
        return fig


    def add_points(self, new_points):
        """
        Accepts:
        - Nx3  -> XYZ (default blue)
        - Nx6  -> XYZRGB
        """

        new_points = np.asarray(new_points)

        if new_points.shape[1] == 3:
            xyz = new_points
            rgb = np.tile(np.array([[0, 0, 255]]), (len(xyz), 1))  # default blue

        elif new_points.shape[1] == 6:
            xyz = new_points[:, :3]
            rgb = new_points[:, 3:6]

        else:
            raise ValueError("new_points must be Nx3 or Nx6 (XYZ or XYZRGB)")

        self.points = (
            np.vstack([self.points, xyz]) if self.points.size else xyz
        )
        self.colors = (
            np.vstack([self.colors, rgb]) if self.colors.size else rgb
        )



if __name__ == "__main__":
    import numpy as np
    import time

    viewer = LivePointCloud()  # opens browser tab

    # Stream points gradually
    for _ in range(50):
        new_points = np.random.uniform(-8000, 8000, size=(100, 3))
        viewer.add_points(new_points)
        time.sleep(0.1)  # simulate real-time updates
