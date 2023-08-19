"""
 https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

df = px.data.iris()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Cell Teste", className="display-4"),
        html.Hr(),
        html.P(
            "Analize Dashboard", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("General", href="/", active="exact"),
                dbc.NavLink("Model", href="/page-1", active="exact"),
                dbc.NavLink("Model Len", href="/page-2", active="exact"),
                dbc.NavLink("Scores", href="/page-3", active="exact"),
                dbc.NavLink("Reports", href="/page-4", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P(
                         html.Div(
                                [
                                    html.H4("filtered by petal width"),
                                    
                                    dcc.Graph(id="graph"),
                                    html.P("Petal Width:"),
                                    dcc.RangeSlider(
                                        id="range-slider",
                                        min=0,
                                        max=2.5,
                                        step=0.1,
                                        marks={0: "0", 2.5: "1.0"},
                                        value=[0.5, 2],
                                    ),
                                ]
                            )           
                    )
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    elif pathname == "/page-3":
        return html.P("Oh cool, this is page 3!")
    elif pathname == "/page-4":
        return html.P("Oh cool, this is page 4!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(
    Output("graph", "figure"),
    Input("range-slider", "value"),
)
def update_chart(slider_range):
    low, high = slider_range
    mask = (df.petal_width > low) & (df.petal_width < high)

    fig = px.scatter_3d(
        df[mask],
        x="sepal_length",
        y="sepal_width",
        z="petal_width",
        color="species",
        hover_data=["petal_width"],
    )
    return fig



if __name__ == "__main__":
    #app.run_server(port=8888)
    app.run_server(debug=True)