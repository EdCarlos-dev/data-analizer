#https://dash.plotly.com/interactive-graphing

# construção do dashbord
import dash
from dash import Dash, html, dcc, callback, Output, Input, dash_table
from dash.dash_table import FormatTemplate, DataTable
from dash.dash_table.Format import Group, Scheme, Symbol, Format
import dash_bootstrap_components as dbc

# importar o  dash_design_kit as ddk

# construção dos gráficos
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Tratamento dos dados
import pandas as pd
import numpy as np
from natsort import index_natsorted   

data = pd.read_csv('final_df_compare_melt.csv')
model_ordered = data.sort_values(by=['model_len' ,'metrics', 'Model'],key = lambda x:np.argsort(index_natsorted(data['Model'])))
model_len_order = model_ordered.sort_values(by=['model_len' ,'metrics', 'Model'],key = lambda x:np.argsort(index_natsorted(model_ordered['model_len'])))
df = model_len_order.sort_values(by=['model_len' ,'metrics', 'Model'],key = lambda x:np.argsort(index_natsorted(model_len_order['metrics'])))
final_df_compare_describe = df.describe()

# df_order = df.sort_values(by=['model_len' ,'metrics'],key = lambda x:np.argsort(index_natsorted(df['model_len'])))
# mean_data_all = df_order.groupby(['model_len','metrics'] ).mean().reset_index(level=['model_len','metrics'])
# mean_data = mean_data_all.sort_values(by=['model_len' ,'metrics'],key = lambda x:np.argsort(index_natsorted(mean_data_all['model_len'])))

data_unet = df.loc[df['Model']=='unet'].dropna() 
df['model_len']=df['model_len'].apply(lambda x: x.replace('imagens', '')) 
metrics_list = []

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
                            html.H1("Cell Segment", className="display-5"),
                            html.Hr(),
                            html.P(
                                "Segmentation Dashboard", className="lead"
                            ),
                            dbc.Nav(
                                [
                                    dbc.NavLink("General", href="/", active="exact"),
                                    # dbc.NavLink("Unet", href="/unet", active="exact"),
                                    dbc.NavLink("Model", href="/model", active="exact"),
                                    dbc.NavLink("Model Len", href="/model-len", active="exact"),
                                    dbc.NavLink("Performance", href="/performance", active="exact"),
                                    dbc.NavLink("Reports", href="/reports", active="exact"),
                                ],
                                vertical=True,
                                pills=True,
                            ),
                        ],
                        style=SIDEBAR_STYLE,
                    )


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


page_general =      html.Div([
                                #https://dash.plotly.com/dash-core-components/upload
                                
                                html.H1(children='General Graphs', style={'textAlign':'center'}),
                                dcc.Graph(id='graph-general_pixel'),
                                dcc.Graph(id='graph-general_226'),
                                dcc.Graph(id='graph-general_226_optic'),
                                dcc.Graph(id='graph-general_226_pixel'),

                                """ 1 - pixel a pixel \n
                                    2 - unet 226 imagens \n
                                    3 - unet 226 somente optico \n
                                    4 - unet 226 imagens + pixel a pixel
                                """
                            ])


# page_unet =         html.Div([
#                                 html.H1(children='Unet Analizer',  style={'textAlign':'center'}),
#                                 dcc.Dropdown(data_unet.metrics.unique(), 'F1', id='dropdown-unet'),
#                                 html.Button('Report Generate', id='button-unet-report', n_clicks=0,style={}),
#                                 dcc.Graph(id='graph-unet'),
#                                 DataTable(id='table_unet', page_size=10),

#                             ])

metrics_general = df.metrics.unique()
metrics_options = [{'label': i, 'value': i} for i in metrics_general]                    

page_model =        html.Div([
                                html.H1(children='Model ',  style={'textAlign':'center'}),


                                html.Div('Select Model'),
                                #dcc.Dropdown(df.Model.unique(), 'unet',  id='dropdown-models'),
                                
                                #html.Div('Select Metric'),
                                #dcc.Dropdown(id='metrics-dropdown2', value=df.metrics.unique(), options=df.metrics.unique(), multi=True,),
                                # dcc.Checklist(id='metrics-dropdown',
                                #                         options=df.metrics.unique(),
                                #                         value=df.metrics.unique(),
                                
                                    html.Div([
                                                dcc.Dropdown(df.Model.unique(), 'unet',  id='dropdown-models')
                                                ], style={'display': 'inline-block','width': '49%'}),
                                    html.Div([
                                                   dcc.Checklist(id='metrics-dropdown',
                                                        options=metrics_list,
                                                        value=metrics_list,
                                                    ),

                                            #    dbc.DropdownMenu(
                                             #   children=[
                                              #      dcc.Checklist(id='metrics-dropdown1',
                                               #         options=metrics_list,
                                                #        value=metrics_list,
                                                 #   ),
                                               # ],
                                             #   label="",
                                            #)
                                                ], style={'display': 'inline-block', 'width': '49%'}),

                                html.Button('Report Generate', id='button-model-report', n_clicks=0,style={}),               
                                # html.Div(
                                #             children=dbc.DropdownMenu(
                                #                 children=[
                                #                     dcc.Checklist(id='metrics-dropdown',
                                #                         options=df.metrics.unique(),
                                #                         value=df.metrics.unique(),
                                #                     ),
                                #                 ],
                                #                 label="Y",
                                #             ),
                                #         ),
                                html.Div(
                                    [
                                    dcc.Graph(id='graph-model'),
                                    dcc.Graph(id='graph-model_b'),
                                    dcc.Graph(id='graph-model_lines'),
                                    DataTable(id='table_model', page_size=10),
                                    DataTable(id='table_model_b', page_size=10),
                                    ]
                                ),
                                

                            ])


page_model_len =    html.Div([
                                html.H1(children='Model_len Images',  style={'textAlign':'center'}),
                                html.Button('Report Generate', id='button-model-len-report', n_clicks=0,style={}),
                                html.Div('Select Model Len'),
                                dcc.Dropdown(df.model_len.unique(), '50 images', id='dropdown-model_len'),
                                #html.Div('Select Metric'),
                                # dcc.Dropdown(id='metrics-unet-dropdown', value=df.metrics.unique(), options=df.metrics.unique(), multi=True,),
                                
                                html.Div(
                                            children=dbc.DropdownMenu(
                                                children=[
                                                    dcc.Checklist(id='metrics-unet-dropdown',
                                                        options=metrics_list,
                                                        value=metrics_list,
                                                    ),
                                                ],
                                                label = "Metric Selector",
                                            ),
                                        ),


                                dcc.Graph(id='graph-unet-model-len'),

                                dcc.Dropdown(df.Model.unique(), 'logisticRegression', id='dropdown-model_len-model'),
                                

                                html.Div(
                                            children=dbc.DropdownMenu(
                                                children=[
                                                    dcc.Checklist(id='metrics-othermodels-dropdown',
                                                        options=df.metrics.unique(),
                                                        value=df.metrics.unique(),
                                                    ),
                                                ],
                                                label = "Metric Selector",
                                            ),
                                        ),



                                    html.Div([
                                                dcc.Graph(id='graph-logisticregression')
                                                ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
                                    html.Div([
                                                dcc.Graph(id='graph-logisticregressionproba')
                                                ], style={'display': 'inline-block', 'width': '49%'}),


                                #dcc.Graph(id='graph-logisticregression'),
                                #dcc.Graph(id='graph-logisticregressionproba'),
                                DataTable(id='table_two', page_size=10),

                            ])

page_performace =   html.Div([
                                html.H1(children='Model Performace', style={'textAlign':'center'}),
                                html.P("Models Performace"),
                                
                            ])

page_reports =      html.Div([
                                html.H1(children='Reports Grnerate', style={'textAlign':'center'}),
                                html.P("Reports"),
                            ])
# aqui concatenando  as páginas ao menu
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_general
    # elif pathname == "/unet":
    #     return page_unet
    elif pathname == "/model":
        return page_model
    elif pathname == "/model-len":
        return page_model_len
    elif pathname == "/performance":
        return page_performace
    elif pathname == "/reports":
        return page_reports
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

#############################
# mage model len 
#############################
# atualizando o segundo menu a partir do prrimeiro
@app.callback(
    Output('metrics-unet-dropdown', 'options'),
    [Input('dropdown-model_len', 'value')]
)
def update_metrics_page_model_len(selected_categoria):
    data_model = df.loc[df['model_len']==f'{selected_categoria}'].dropna()
    data_unet = data_model[data_model.Model=='unet'].dropna()
    filtered_items = data_unet.metrics.unique()
    #df[df['Model'] == selected_categoria]['Item'].unique()
    item_options = [{'label': item, 'value': item} for item in filtered_items]
    return item_options

# model len page firt graph
@app.callback(
    Output(component_id = 'graph-unet-model-len', component_property = 'figure'),
    Input('dropdown-model_len', 'value'),
    Input('metrics-unet-dropdown','value')
)
def update_graph(value, metric):
    data_to_graph = df[df.model_len==value]
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)].dropna()
    data_to_graph = data_to_graph[data_to_graph.Model=='unet'].dropna()
    graph = px.box(data_to_graph, x='Model', y='scores', color='metrics')
    graph.update_traces(marker = dict(opacity = 0))

    return graph

# a partir do model len obtido, verificar outro modelo chamado que não seja unet


@app.callback(
    Output(component_id = 'graph-logisticregression', component_property = 'figure'),
    Input('dropdown-model_len', 'value'),
    Input('dropdown-model_len-model', 'value'),
    Input('metrics-othermodels-dropdown','value')
)
def update_graph_model_metric_only(len,model,metric):

    data_to_graph = df[df.model_len==len]
    data_to_graph = data_to_graph[data_to_graph.Model==model].dropna()
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)].dropna()
    graph = px.box(data_to_graph, x='Model', y='scores', color='metrics')
    graph.update_traces(marker = dict(opacity = 0))

    return graph

@app.callback(
    Output(component_id = 'graph-logisticregressionproba', component_property = 'figure'),
    Input('dropdown-model_len', 'value'),
    Input('dropdown-model_len-model', 'value'),
    Input('metrics-othermodels-dropdown','value')
)
def update_graph_model_metric_unetcompare(len, model, metric):

    data_to_graph = df[df.model_len==len]
    data_to_graph = data_to_graph[data_to_graph.Model==model].dropna()
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)].dropna()
    graph = px.box(data_to_graph, x='Model', y='scores', color='metrics')
    graph.update_traces(marker = dict(opacity = 0))

    return graph


@app.callback(
    Output(component_id = 'table_two', component_property = 'data'),
    Input('dropdown-model_len', 'value')
)
def update_table(value):
   
    data_to_table = df[df.model_len==value]
    data=data_to_table.to_dict('records')
    return data


##################################
# mage model
##################################

# atualizando o segundo menu
@app.callback(
    Output('metrics-dropdown', 'options'),
    [Input('dropdown-models', 'value')]
)
def update_metrics_page_model(selected_categoria):
    data_model = df.loc[df['Model']==f'{selected_categoria}'].dropna()
    filtered_items = data_model.metrics.unique()
    #df[df['Model'] == selected_categoria]['Item'].unique()
    item_options = [{'label': item, 'value': item} for item in filtered_items]
    return item_options

# page model graph A

@app.callback(
    Output(component_id = 'graph-model', component_property = 'figure'),
    Input('dropdown-models', 'value'),
    Input('metrics-dropdown','value')
)
def update_graph_model(value, metric):
    # if(value=='unet'):
    
    data_to_graph = df[df.Model==value].dropna()
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)].dropna()
    graph = px.box(data_to_graph.loc[data_to_graph['model_len']!='226 somente optico' ]  ,x="model_len", y='scores' ,color='metrics', title=f'Model {value} sample comparison 50 to 226 imagens')
    graph.update_traces(marker = dict(opacity = 0))

    #update_layout(  barmode='group',yaxis_range = [0,1])

    return graph


# page model graph B
@app.callback(
    Output(component_id = 'graph-model_b', component_property = 'figure'),
    Input('dropdown-models', 'value'),
   Input('metrics-dropdown','value')
)
def update_graph_model_b(value, metric):
    # if(value=='unet'):

    data_to_graph = df[df.Model==value].dropna()
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)].dropna()
    graph_b = px.box(data_to_graph.loc[(data_to_graph['model_len']=='226 somente optico') | (data_to_graph['model_len']=='226 ') ]  ,x="model_len", y='scores' ,color='metrics', title= f'Model {value} sample comparison 226 imagens complete and 226 optical only')
    graph_b.update_traces(marker = dict(opacity = 0))

    return graph_b

# page model graph lines
@app.callback(
    Output(component_id = 'graph-model_lines', component_property = 'figure'),
    Input('dropdown-models', 'value')
)
def update_graph_model_lines(value):
    # if(value=='unet'):

    data_to_graph = mean_data_all[mean_data_all.Model==value]
    graph_lines = px.line(data_to_graph.loc[data_to_graph['model_len']!='226 somente optico' ]   ,x='model_len', y='scores',color='metrics', title='Unet sample comparison ')
    graph_lines.update_traces(marker = dict(opacity = 0))

    return graph_lines

# page model table A
@app.callback(
    Output(component_id = 'table_model', component_property = 'data'),
    Input('dropdown-models', 'value')
)
def update_table_model(value):
   
    data_to_table = df[df.Model==value]
    data=data_to_table.to_dict('records')
    return data

# page model table B
@app.callback(
    Output(component_id = 'table_model_b', component_property = 'data'),
    Input('dropdown-models', 'value')
)
def update_table_model_b(value):
   
    data_to_table = df[df.Model==value]
    data=data_to_table.to_dict('records')
    return data


##################################
# page Unet  first graph 
##################################

@app.callback(
    Output(component_id = 'graph-unet', component_property = 'figure'),
    Input('dropdown-unet', 'value')
)
def update_graph_unet(value):
    data = df[df.metrics==value]
    data_to_graph = data[data.Model=='unet'].dropna()
    graph_unet = px.box(data_to_graph.loc[data_to_graph['model_len']!='226 somente optico' ]  ,x="model_len", y='scores' ,color='metrics', title=f'Model Unet {value} sample comparison 50 to 226 imagens')
    graph_unet.update_traces(marker = dict(opacity = 0))

    return graph_unet

@app.callback(
    Output(component_id = 'table_unet', component_property = 'data'),
    Input('dropdown-unet', 'value')
)
def update_table_unet(value):
    data = df[df.metrics==value]
    data_to_table = data[data.Model=='unet']
    data=data_to_table.to_dict('records')
    return data

'''

@app.callback(
    Output(),
    Input()
)
def report_generate()

'''    

if __name__ == "__main__":
    #app.run_server(port=8888)
    app.run(debug=True)