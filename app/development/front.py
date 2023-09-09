#https://dash.plotly.com/interactive-graphing
# https://hellodash.pythonanywhere.com/
# https://community.plotly.com/t/dash-bootstrap-theme-light-dark-switcher-with-toggle/56205/2
# https://plotly.com/python/templates/
# https://plotly.com/python/figure-labels/
# https://dash.plotly.com/dash-core-components/upload

# construção do dashbord
import dash
from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
from dash.dash_table import FormatTemplate, DataTable
from dash.dash_table.Format import Group, Scheme, Symbol, Format
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO

import dash_ag_grid as dag
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

from datetime import datetime

from fpdf import FPDF

# pdf generator
# data atual
today = datetime.now()
# formatando a data
today = today.strftime('%Y-%m-%d')

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('dev/apps/dashboard/lgcmlogo.png', 10, 8, 33)
        self.image('dev/apps/dashboard/fzusp.png', 160, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 12)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Data Analysis Report'+ '- Generated at-' +f'{today}', 0, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Report Page ' + str(self.page_no()) + '/{nb}' + '-Report Generated at-' +f'{today}', 0, 0, 'C')


def report(pdf, fig_name):
    
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Times', '', 12)
    pdf.cell(200, 10, txt = f"Experiment",ln = 1, align = 'C')
    pdf.cell(200, 10, txt = f"Data", ln = 2, align = 'C')
    pdf.image(fig_name, x=10, y=50, w=pdf.w/1.2, h=pdf.h/2.5 )
    # pdf.image(f'{path}{table_name}.png', x=10, y=170, w=pdf.w/1.2, h=pdf.h/2.5 )



# data general page
general_melt_df = pd.read_csv('general_metrics_melt.csv')
metric_order_categorical=['Precision'  ,    'Recall'  ,  'F1'  ,  'Jaccard', 'unet_Precision'  ,    'unet_Recall'  ,  'unet_F1'  ,  'unet_Jaccard']
general_melt_df['metrics'] = general_melt_df['metrics'].astype('category')
general_melt_df['metrics'] = general_melt_df['metrics'].cat.reorder_categories(metric_order_categorical, ordered = True)
general_melt_df.sort_values(by='metrics', inplace=True)
general_melt_df['metrics'] = general_melt_df['metrics'].astype('string')


# data performace
performace_df = pd.read_csv('general_performaces.csv')
performace_melt = pd.melt(performace_df, id_vars=['epoch','len_data'], value_vars=['dice_coef','iou','loss','lr','precision','recall','val_dice_coef','val_iou','val_loss','val_precision','val_recall'] , var_name= 'metrics' , value_name='scores')
#performace_melt.to_csv('dev/apps/dashboard/general_performace_melt.csv')
performace_melt_df =  pd.read_csv('general_performace_melt.csv')

# data model  - model len
data = pd.read_csv('final_df_compare_melt.csv')

metric_order_categorical=['Precision'  ,    'Recall'  ,  'F1'  ,  'Jaccard', 'unet_Precision'  ,    'unet_Recall'  ,  'unet_F1'  ,  'unet_Jaccard']
data['metrics'] = data['metrics'].astype('category')
data['metrics'] = data['metrics'].cat.reorder_categories(metric_order_categorical, ordered = True)
data.sort_values(by='metrics', inplace=True)
data['metrics'] = data['metrics'].astype('string')

model_ordered = data.sort_values(by=['model_len' ,'metrics', 'Model'],key = lambda x:np.argsort(index_natsorted(data['Model'])))
df = model_ordered.sort_values(by=['model_len' ,'metrics', 'Model'],key = lambda x:np.argsort(index_natsorted(model_ordered['model_len'])))

# df = model_len_order.sort_values(by=['model_len' ,'metrics', 'Model'],key = lambda x:np.argsort(index_natsorted(model_len_order['metrics'])))


'''
# final_df_compare_describe = df.describe()
# df_order = df.sort_values(by=['model_len' ,'metrics'],key = lambda x:np.argsort(index_natsorted(df['model_len'])))
# mean_data_all = df_order.groupby(['model_len','metrics'] ).mean().reset_index(level=['model_len','metrics'])
# mean_data = mean_data_all.sort_values(by=['model_len' ,'metrics'],key = lambda x:np.argsort(index_natsorted(mean_data_all['model_len'])))
# data_unet = df.loc[df['Model']=='unet'].dropna() 
'''

df['model_len']=df['model_len'].apply(lambda x: x.replace('imagens', '')) 

metrics_list = df.metrics.unique()

metrics_list_page_model = ['Precision'  ,    'Recall'  ,  'F1'  ,  'Jaccard']

models_list = df.Model.unique()
without_unet = df.loc[df['Model']!='unet'].dropna()
models_list_without_unet = without_unet.Model.unique()

performace_metrics_list = performace_melt_df.metrics.unique()
''
metric_model_order = ['Precision'  ,    'Recall'  ,  'F1'  ,  'Jaccard']

# theme
def dark_mode_graph(graph):
    dark_graph = graph.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='rgba(0, 0, 0, 0)',
                                #paper_bgcolor='rgba(0, 0, 0, 0)',
                            )
    return dark_graph

def white_mode_graph(graph):

    white_graph = graph.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)'
                    )
    return white_graph

# graph line constructor functions
def graph_line_constructor(data_to_graph, x_value, y_value , color_target , title_name,  light_mode = None):
    
    graph = px.line(data_frame = data_to_graph , 
                    x = x_value , 
                    y= y_value , 
                    color = color_target ,
                    title = title_name )
    graph.update_layout(
                            title_font=dict(size=20),       # Tamanho da fonte do título
                            # xaxis_title_font=dict(size=14),  # Tamanho da fonte do eixo X
                            # yaxis_title_font=dict(size=14),  # Tamanho da fonte do eixo Y
                            # legend_font=dict(size=14)       # Tamanho da fonte da legenda
                            font=dict(
                                        # family="Courier New, monospace",
                                        size=14,
                                        # color="RebeccaPurple"
                                    )
                        ) 
    
    graph.update_yaxes(range = [0,1])

    if light_mode == 'dark':
        dark = dark_mode_graph(graph)
        return dark
    elif light_mode == 'white':
        white = white_mode_graph(graph)
        return white
    else:
        return graph

# boxplot graph constructor
def graph_boxplot_constructor(data_to_graph, x_value, y_value , color_target , title_name,  light_mode = None):

    graph = px.box(data_frame = data_to_graph , 
                    x = x_value , 
                    y= y_value , 
                    color = color_target ,
                    title = title_name )
    
    graph.update_layout(
                            title_font=dict(size=20),       # Tamanho da fonte do título
                            # xaxis_title_font=dict(size=14),  # Tamanho da fonte do eixo X
                            # yaxis_title_font=dict(size=14),  # Tamanho da fonte do eixo Y
                            # legend_font=dict(size=14)       # Tamanho da fonte da legenda
                            font=dict(
                                        # family="Courier New, monospace",
                                        size=14,
                                        # color="RebeccaPurple"
                                    )
                        )   
    graph.update_yaxes(range = [0,1])
    graph.update_traces(marker = dict(opacity = 0))
    if light_mode == 'dark':
        dark = dark_mode_graph(graph)
        return dark
    
    elif light_mode == 'white':
        white = white_mode_graph(graph)
        return white
    else:
        return graph

# graph performace page constructor
def performace_graph(len, metrics_list, light_mode = 'white'):

    data_to_graph = performace_melt_df[performace_melt_df['metrics'].isin(metrics_list)]
    data_to_graph = data_to_graph.loc[data_to_graph['len_data']==len]
    graph_performaces = graph_line_constructor(data_to_graph,  
                                                x_value= 'epoch' , 
                                                y_value= 'scores' , 
                                                color_target= 'metrics', 
                                                title_name= f'{metrics_list[0]} analyze - {len} images', 
                                                light_mode = light_mode)

    return  graph_performaces


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# dbc.themes.BOOTSTRAP

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "8rem",
    "margin-right": "2rem",
    "bottom": 0,
    "width": "15rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

MENU_STYLE = {
    # "position": "fixed",
    "margin-left":"15rem",
    "background-color": "#ffffff", 
    
    }

CONTENT_STYLE = {
    "top": "8rem",
    "bottom": 0,
    "width": "10rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "margin-left":"15rem",
    "width": "100rem",
    
    "padding": "2rem 1rem",
    # "background-color": "#f8f9fa"
}

menu = html.Div(
                        [   

                            
                            html.H1("", className="display-5"),
                            html.Hr(),
                            html.P(
                                "", className="lead"
                            ),
                            
                            dbc.Nav(
                                [   
                                    html.Img(src="lgcmlogo.png"),
                                    html.Img(src=dash.get_asset_url('lgcmlogo.png')), 
                                    dbc.NavLink("General", href="/", active="exact"),
                                    dbc.NavLink("Model", href="/model", active="exact"),
                                    dbc.NavLink("Model Len", href="/model-len", active="exact"),
                                    dbc.NavLink("Performance", href="/performance", active="exact"),
                                        
                                ],
                                
                                pills=True,
                            ),
                            
                        ],
                        style = MENU_STYLE,
                        id='menu-dash'
                    )

sidebar_format = html.Div(id = 'pagesidebar', style=SIDEBAR_STYLE)

content = html.Div(id="page-content", style=CONTENT_STYLE)




app.layout = html.Div([dcc.Location(id="url"), menu, sidebar_format, content])


page_general =      html.Div([

                                html.Div([
                                     html.H1(children='General Analisys',  style={'textAlign':'left'}),
                                ]),
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label="Graph", tab_id="Graph_tab"),
                                        dbc.Tab(label="Table", tab_id="Table_tab"),
                                    ],
                                    id="tabs",
                                    active_tab="Graph_tab"
                                    
                                ),
                                    html.Div([
                                                dcc.Graph(id='graph-page_general')
                                                ], style={'width': '33%', 'display': 'inline-block'}),
                                    html.Div([
                                                dcc.Graph(id='graph-page_general2')
                                                ], style={ 'width': '33%', 'display': 'inline-block'}),                                
                                    html.Div([
                                                dcc.Graph(id='graph-page_general3')
                                                ], style={ 'width': '33%', 'display': 'inline-block'}),
                                     
                                #  dbc.Row([
                                #         dbc.Col(dcc.Graph(id='graph-page_general'), width=4),
                                #         dbc.Col(dcc.Graph(id='graph-page_general2'), width=4),
                                #         dbc.Col(dcc.Graph(id='graph-page_general3'), width=4), ])                                                                
                                
                            ])

sidebar_general_page = html.Div( 
                    [
                        
                        html.H1("", className="display-5"),
                        html.Hr(),
                        html.P(
                                "Graph Options", className="lead"
                            ),
                                         
                        html.P(
                                    "Click to Report", className="lead"
                                ), 
                        dbc.Button('Report Generate', id='general_report_pdf', n_clicks = 0 , style={
                        'background-color': '#0d6efd',
                        'color': 'white'}),
                        html.Div([html.P(id="paragraph_id", children=["Button not clicked"])]),
                    ])



page_model =        html.Div([
                                html.H1(children='Model Analisys',  style={'textAlign':'left'}),

                                dbc.Tabs(
                                    [
                                        dbc.Tab(label="Graph", tab_id="Graph_tab"),
                                        dbc.Tab(label="Table", tab_id="Table_tab"),
                                    ],
                                    id="tabs",
                                    active_tab="Graph_tab"
                                    
                                ),                                
                                html.Div(
                                    [
                                    
                                    html.Div([dcc.Graph(id='graph-model'),], style={'margin-left':'10%','display': 'inline-block','width': '80%'} ),
                                    html.Div([dcc.Graph(id='graph-model_b'),], style={'margin-left':'30%','display': 'inline-block','width': '45%'} ),

                                   
                                    # dcc.Graph(id='graph-model_lines'),
                                    # DataTable(id='table_model', page_size=10),
                                    # DataTable(id='table_model_b', page_size=10),
                                    ]
                                ),
                                
                            ])

sidebar_page_model = html.Div( 
                    [
                        html.H1("", className="display-5"),
                        html.Hr(),
                        html.P(
                                "Graph Options", className="lead"
                            ),

                        html.P(
                                    "Unet - Analisys", className="lead"
                                ),
                        html.Div('Select Metric'),
                        html.Div([

                                        # dcc.Dropdown(df.Model.unique(), 'unet',  id='dropdown-models'),
                                        html.Div(''),
                                        dcc.Checklist(
                                                        id='metrics-dropdown',
                                                        options=metrics_list_page_model,
                                                        value=metrics_list_page_model,
                                                    ),                           
                                            
                                        ]),
                        html.P(
                                    "Click to Report", className="lead"
                                ), 
                        dbc.Button('Report Generate', id='sidebar_page_model_report_button', n_clicks=0,style={
                        'background-color': '#0d6efd',
                        'color': 'white'})

                    ])


page_model_len =    html.Div([
                                html.H1(children='Model Len Images',  style={'textAlign':'left'}),
                                dbc.Tabs(
                                    [
                                        dbc.Tab(label="Graph", tab_id="Graph_tab"),
                                        dbc.Tab(label="Table", tab_id="Table_tab"),
                                    ],
                                    id="tabs",
                                    active_tab="Graph_tab"
                                    
                                ),

                                html.Div([dcc.Graph(id='graph-unet-model-len'),],style={'margin-left':'5%','width': '45%', 'display': 'inline-block'}),
                                html.Div([dcc.Graph(id='graph-subsurt-model-len'),],style={'width': '45%', 'display': 'inline-block'} ),
                               
                                html.Div([
                                            
                                            dcc.Graph(id='graph-logisticregression')
                                            ], style={'margin-left':'5%','width': '45%', 'display': 'inline-block'}),
                                html.Div([
                                            dcc.Graph(id='graph-logisticregressionproba')
                                            ], style={'width': '45%', 'display': 'inline-block'}),

                                # DataTable(id='table_two', page_size=10),
                            ])


sidebar_page_model_len = html.Div( 
                    [
                        html.H1("", className="display-5"),
                        html.Hr(),
                        html.P(
                                    "Model - Unet", className="lead"
                                ),
                        html.Div('Select Model Len'), 
                        html.Div([

                                        dcc.Dropdown(df.model_len.unique(), df.model_len.unique()[0], id='dropdown-model_len'),
                                        dcc.Checklist(id='metrics-unet-dropdown',
                                                        options=metrics_list,
                                                        value=metrics_list,
                                                    ),                                           
                                        ]),

                        html.Div('Models Compare'),
                        html.Div([

                                        dcc.Dropdown(options=models_list_without_unet ,value=models_list_without_unet[0], id='dropdown-model_len-model'),
                                        dcc.Checklist(
                                                        id='metrics-othermodels-dropdown',
                                                        options=metrics_list,
                                                        value=metrics_list,

                                                    ),                                                                                  
                                    ]),                        
                                
                        html.P(
                                    "Click to Report", className="lead"
                                ), 
                        dbc.Button('Report Generate', id='page-model-len-button-report', n_clicks=0,style={
                        'background-color': '#0d6efd',
                        'color': 'white'})

                    ])



page_performace =   html.Div([
                                html.H1(children='Model Performace', style={'textAlign':'left'}),
                                 dbc.Tabs(
                                    [
                                        dbc.Tab(label="Graph", tab_id="Graph_tab"),
                                        dbc.Tab(label="Table", tab_id="Table_tab"),
                                    ],
                                    id="tabs",
                                    active_tab="Graph_tab"                                   
                                ),

                                html.Div([
                                            dcc.Graph(id='graph-page_performace1')
                                            ], style={'display': 'inline-block','width': '33%'}),
                                html.Div([
                                            dcc.Graph(id='graph-page_performace2')
                                            ], style={'display': 'inline-block', 'width': '33%'}),
                                html.Div([
                                            dcc.Graph(id='graph-page_performace3')
                                            ], style={'display': 'inline-block','width': '33%'}),
                                html.Div([
                                            dcc.Graph(id='graph-page_performace4')
                                            ], style={'display': 'inline-block', 'width': '33%'}),
                            
                                html.Div([
                                            dcc.Graph(id='graph-page_performace5')
                                            ], style={'display': 'inline-block','width': '33%'}),
                                html.Div([
                                            dcc.Graph(id='graph-page_performace6')
                                            ], style={'display': 'inline-block', 'width': '33%'}),
                                    
                            ])

sidebar_page_performace = html.Div( 
                    [
                        html.H1("", className="display-5"),
                        html.Hr(),
                        
                        html.P(
                                    "Performace Model", className="lead"
                                ), 
                        html.Div([

                                        html.Div([
                                                dcc.Dropdown(performace_melt_df.len_data.unique(), performace_melt_df.len_data.unique()[0], id='dropdown-model_len-performace'), 
                                                    ], style={})
                                            
                                        ]),
                        html.P(
                                    "Click to Report", className="lead"
                                ), 
                        dbc.Button('Report Generate', id='button-report-performace', n_clicks=0,style={
                        'background-color': '#0d6efd',
                        'color': 'white'})

                    ])


page_reports =      html.Div([
                                html.H1(children='Reports Grnerate', style={'textAlign':'left'}),
                                html.P("Reports"),

                            ])

graphs_general =  dbc.Row([
                    dbc.Col(dcc.Graph(id='graph-page_general'), width=6),
                    dbc.Col(dcc.Graph(id='graph-page_general2'), width=6),
                    dbc.Col(dcc.Graph(id='graph-page_general3'), width=6), ])  


# aqui concatenando  os sidebars as páginas e ao menu
@app.callback(  Output("page-content", "children"),
                Output("pagesidebar", "children"),
                [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_general, sidebar_general_page
    elif pathname == "/model":
        return page_model, sidebar_page_model
    elif pathname == "/model-len":
        return page_model_len, sidebar_page_model_len
    elif pathname == "/performance":
        return page_performace , sidebar_page_performace
    
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )




#############################
# page general
#############################
@app.callback(
    Output('graph-page_general', 'figure'),
    Output('graph-page_general2', 'figure'),
    Output('graph-page_general3', 'figure'),
    Output('general_report_pdf', 'n_clicks'),

    Input("url", "pathname"),
    Input('general_report_pdf', 'n_clicks')

)
def general_graph_pixel(pathname,n_clicks):

    metric_model = ['Precision'  ,    'Recall'  ,  'F1'  ,  'Jaccard']
    model_segmentation = ['pixel','only_optico','afm_optico']
    graphic_list = []

    if n_clicks == 0:

        for model in model_segmentation:

            data_to_graph = general_melt_df[general_melt_df.type==model] 
            data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric_model)]
            graph = graph_boxplot_constructor(data_to_graph, x_value="metrics", y_value='scores'  , color_target='metrics' , title_name=f'{model}_segmentation', light_mode='white')
            graphic_list.append(graph)
        
        graphic_list.append(n_clicks)
        
        return graphic_list  

    else:

        pdf = PDF()

        for model in model_segmentation:

            data_to_graph = general_melt_df[general_melt_df.type==model] 
            data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric_model)]
            graph = graph_boxplot_constructor(data_to_graph, x_value="metrics", y_value='scores'  , color_target='metrics' , title_name=f'{model}_segmentation', light_mode='white')
            
            graph.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_{model}_{metric_model}.png')       
            report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_{model}_{metric_model}.png')  
            graphic_list.append(graph)
        
        pdf.output(f'dev/apps/dashboard/dash_reports/general_report_pixel_{today}.pdf', 'F')
        n_clicks = 0
        graphic_list.append(n_clicks)

        return  graphic_list



##################################
# page model
##################################

# atualizando o segundo menu
# @app.callback(
#     Output('metrics-dropdown', 'options'),
#     Input('dropdown-models', 'value')
# )
# def update_metrics_page_model(selected_categoria = 'unet'):
#     data_model = df.loc[df['Model']==f'{selected_categoria}'].dropna()
#     filtered_items = data_model.metrics.unique()
#     #df[df['Model'] == selected_categoria]['Item'].unique()
#     item_options = [{'label': item, 'value': item} for item in filtered_items]
#     return item_options

# page model graph A and B

@app.callback(
    Output(component_id = 'graph-model', component_property = 'figure'),
    Output(component_id = 'graph-model_b', component_property = 'figure'),
    Output('sidebar_page_model_report_button', 'n_clicks'),


    Input('metrics-dropdown','value'),
    Input('sidebar_page_model_report_button', 'n_clicks')
)
def update_graph_model(metric, n_clicks):

    data_to_graph = df[df.Model=='unet'].dropna()
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)].dropna()

    if n_clicks == 0:

        graph = graph_boxplot_constructor(data_to_graph = data_to_graph.loc[data_to_graph['model_len']!='226 somente optico' ]  , x_value="model_len", y_value='scores'   , color_target='metrics' , title_name=f'Model Unet sample comparison 50 to 226 imagens', light_mode='white')        
        graph_b = graph_boxplot_constructor(data_to_graph = data_to_graph.loc[(data_to_graph['model_len']=='226 somente optico') | (data_to_graph['model_len']=='226 ') ]  , x_value="model_len", y_value='scores'   , color_target='metrics' , title_name= f'Unet comparison 226 imagens complete and 226 optical only', light_mode='white')
        
        return graph, graph_b, n_clicks

    else:
        pdf = PDF()
        graph = graph_boxplot_constructor(data_to_graph = data_to_graph.loc[data_to_graph['model_len']!='226 somente optico' ]  , x_value="model_len", y_value='scores'   , color_target='metrics' , title_name=f'Model Unet sample comparison 50 to 226 imagens', light_mode='white')
        graph_b = graph_boxplot_constructor(data_to_graph = data_to_graph.loc[(data_to_graph['model_len']=='226 somente optico') | (data_to_graph['model_len']=='226 ') ]  , x_value="model_len", y_value='scores'   , color_target='metrics' , title_name= f'Unet comparison 226 imagens complete and 226 optical only', light_mode='white')


        graph.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_Model Unet sample comparison 50 to 226 imagens_{metric}.png') 
        graph_b.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_Unet comparison 226 imagens complete and 226 optical only_{metric}.png') 

        report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_Model Unet sample comparison 50 to 226 imagens_{metric}.png')  
        report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_Unet comparison 226 imagens complete and 226 optical only_{metric}.png')  
        
        pdf.output(f'dev/apps/dashboard/dash_reports/model_report_unet_{today}.pdf', 'F')

        n_clicks = 0

        return graph, graph_b, n_clicks



# # page model graph lines
# @app.callback(
#     Output(component_id = 'graph-model_lines', component_property = 'figure'),
#     Input('dropdown-models', 'value')
# )
# def update_graph_model_lines(value):
#     # if(value=='unet'):

#     data_to_graph = df[df.Model==value]
#     graph_lines = px.line(data_to_graph.loc[data_to_graph['model_len']!='226 somente optico' ]   ,x='model_len', y='scores',color='metrics', title='Unet sample comparison ')
#     graph_lines.update_traces(marker = dict(opacity = 0))

#     return graph_lines

# # page model table A
# @app.callback(
#     Output(component_id = 'table_model', component_property = 'data'),
#     Input('dropdown-models', 'value')
# )
# def update_table_model(value):
   
#     data_to_table = df[df.Model==value]
#     data=data_to_table.to_dict('records')
#     return data

# # page model table B
# @app.callback(
#     Output(component_id = 'table_model_b', component_property = 'data'),
#     Input('dropdown-models', 'value')
# )
# def update_table_model_b(value):
   
#     data_to_table = df[df.Model==value]
#     data=data_to_table.to_dict('records')
#     return data


#############################
# page model len 
#############################
# primeiro dropdown atualizando o primeiro checklist
# receber model_len
# trazer os dados para o modelo unet
# reunir as métricas
@app.callback(
    Output('metrics-unet-dropdown', 'options'),
    Input('dropdown-model_len', 'value')
)
def update_metrics_page_model_len(selected_categoria):
    data_model = df.loc[df['model_len']==f'{selected_categoria}'].dropna()
    data_unet = data_model[data_model.Model=='unet'].dropna()
    filtered_items = data_unet.metrics.unique()
    item_options = [{'label': item, 'value': item} for item in filtered_items]
    return item_options

# primeiro dropdown atualizando o segundo
# atualizar o segundo dropdown de modelos de acordo com o model len 
# carregando os outros modelos usados na amostra diferentes de unet
@app.callback(
    Output('dropdown-model_len-model', 'options'),
    Input('dropdown-model_len', 'value')
)
def update_models_by_model_len(selected_categoria):
    data_model_len = df.loc[df['model_len']==f'{selected_categoria}'].dropna()
    data_model = data_model_len.loc[data_model_len['Model']!='unet'].dropna()
    filtered_items = data_model.Model.unique()

    return filtered_items


# segundo dropdown atualizando o segundo checklist 
# com as métricas do modelo selecionado no dropdown
@app.callback(
    Output('metrics-othermodels-dropdown', 'options'),
    Input('dropdown-model_len-model', 'value')
)
def update_metrics_by_no_unet_model(selected_categoria):
    data_model = df.loc[df['Model']==f'{selected_categoria}'].dropna()
    filtered_items = data_model.metrics.unique()

    return filtered_items

# separar as métricas : 'F1' , 'Jaccard', 'Precision' , 'Recall', 'unet_F1' , 'unet_Jaccard', 'unet_Precision' , 'unet_Recall' 
# model len page aqui o primeiro gráfico receberá o tamanho da amostra e as metricas selecionadas para o modelo unet
# a partir do model len obtido, e do modelo diferente de unet trazer apenas as quatro primeiras métricas 

@app.callback(
    Output(component_id = 'graph-unet-model-len', component_property = 'figure'),
    Output(component_id = 'graph-logisticregression', component_property = 'figure'),
    Output(component_id = 'graph-logisticregressionproba', component_property = 'figure'),
    Output(component_id ='graph-subsurt-model-len',component_property = 'figure'),
    Output('page-model-len-button-report', 'n_clicks'),

    Input('dropdown-model_len', 'value'),
    Input('dropdown-model_len-model', 'value'),
    Input('metrics-othermodels-dropdown','value'),
    Input('metrics-unet-dropdown','value'),
    Input('page-model-len-button-report', 'n_clicks')
)
def update_graph_model_metric_only(len_images,model,metric, metric_unet, n_clicks):
    metrics_model_only = ['F1' , 'Jaccard', 'Precision' , 'Recall']
    metrics_model_unetcompare = ['unet_F1', 'unet_Jaccard', 'unet_Precision' , 'unet_Recall'] 
    data_to_sungraph =  df[df.model_len==len_images]
    data_to_graph = df[df.model_len==len_images]

    data_to_graph_unet = df[df.model_len==len_images]
    data_to_graph_unet = data_to_graph_unet[data_to_graph_unet.Model=='unet'].dropna()


    data_to_graph = data_to_graph[data_to_graph.Model==model]
    data_to_graph = data_to_graph.loc[data_to_graph['metrics'].isin(metric)]
    

    data_to_graph_unet = data_to_graph_unet.loc[data_to_graph_unet['metrics'].isin(metric_unet)].dropna()

    data_to_graph_a = data_to_graph.loc[data_to_graph['metrics'].isin(metrics_model_only)]
    a_count = len(data_to_graph_a['Process Date'].unique())
    cells_count = int(len(data_to_sungraph['Model'])/8)

    data_to_graph_b = data_to_graph.loc[data_to_graph['metrics'].isin(metrics_model_unetcompare)]
    # b_count = data_to_graph_b ['Model'].value_counts()[model]

    if n_clicks == 0:

        graph_model_unet = graph_boxplot_constructor(data_to_graph_unet,  x_value='Model' , y_value='scores' , color_target='metrics', title_name=f'Unet metrics Sample {len_images} images', light_mode = 'white')
        graph_model_metric_only = graph_boxplot_constructor(data_to_graph = data_to_graph_a,  x_value = 'Model', y_value = 'scores' , color_target = 'metrics', title_name = f"Model Called {model} to {a_count} cells <br> Sample Training {len_images} Images", light_mode = 'white')
        graph_model_metric_compare_unet = graph_boxplot_constructor(data_to_graph = data_to_graph_b,  x_value ='Model' , y_value='scores', color_target ='metrics', title_name = f'Unet metrics to the same {a_count} cells Model Called {model} <br> Sample Training {len_images} images', light_mode = 'white')
        graph_sunburst_model_len = px.pie(
                                                        data_to_sungraph,
                                                        names='Model',
                                                        #values='model_len',
                                                        color='Model',
                                                        hole=.7,
                                                        title=f'Sample test {cells_count} <br> number cells '
                                                    )
        graph_sunburst_model_len.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)'
                    )





        return graph_model_unet, graph_model_metric_only, graph_model_metric_compare_unet,graph_sunburst_model_len, n_clicks
    
    else:
        pdf = PDF() 
        graph_model_unet = graph_boxplot_constructor(data_to_graph_unet,  x_value='Model' , y_value='scores' , color_target='metrics', title_name=f'Unet in {len_images} images', light_mode = 'white')
        graph_model_metric_only = graph_boxplot_constructor(data_to_graph = data_to_graph_a,  x_value = 'Model', y_value = 'scores' , color_target = 'metrics', title_name = f'Model len {len_images} Model {model} Metrics', light_mode = 'white')
        graph_model_metric_compare_unet = graph_boxplot_constructor(data_to_graph = data_to_graph_b,  x_value ='Model' , y_value='scores', color_target ='metrics', title_name = f'Model len {len_images} Model {model} Metrics X unet', light_mode = 'white')
        
        graph_model_unet.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_Model_Unet_{len_images}_images_{metric_unet}.png') 
        graph_model_metric_only.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_Model_Model_{model}_{len_images}_images_{metric}.png') 
        graph_model_metric_compare_unet.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_Model_Model_{model}_{len_images}_images_{metric}_X_unet.png') 
        
        report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_Model_Unet_{len_images}_images_{metric_unet}.png')  
        report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_Model_Model_{model}_{len_images}_images_{metric}.png') 
        report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_Model_Model_{model}_{len_images}_images_{metric}_X_unet.png') 
         
        pdf.output(f'dev/apps/dashboard/dash_reports/model_len_report_{today}.pdf', 'F')
        n_clicks = 0

        return graph_model_unet, graph_model_metric_only, graph_model_metric_compare_unet,graph_sunburst_model_len, n_clicks



@app.callback(
    Output(component_id = 'table_two', component_property = 'data'),
    Input('dropdown-model_len', 'value')
)
def update_table(value):
   
    data_to_table = df[df.model_len==value]
    data=data_to_table.to_dict('records')
    return data

##########################
# Page Performace
##########################


@app.callback(
    Output('graph-page_performace1', 'figure'),
    Output('graph-page_performace2', 'figure'),
    Output('graph-page_performace3', 'figure'),
    Output('graph-page_performace4', 'figure'),
    Output('graph-page_performace5', 'figure'),
    Output('graph-page_performace6', 'figure'),
    Output('button-report-performace', 'n_clicks'),

    Input('dropdown-model_len-performace', 'value'),
    Input('button-report-performace', 'n_clicks')
)
def performace_graph_recall(len, n_clicks):

    
    performace_dictionary = {

                            'dice_list':        ['dice_coef','val_dice_coef'],
                            'iou_list':         ['iou','val_iou'],
                            'loss_list':        ['loss','val_loss'],
                            'precision_list':   ['precision','val_precision'],
                            'recall_list':      ['recall','val_recall'],
                            'lr_list':          ['lr']
                        }
    list_variables=[]

    if n_clicks == 0:

        for key_performace in performace_dictionary:

            graph_performace_to_page = performace_graph(len,  performace_dictionary[key_performace])
            list_variables.append(graph_performace_to_page)

        list_variables.append(n_clicks)

        return list_variables        

    else:
        pdf = PDF()    
        for key_performace in performace_dictionary:
            graph_performace_to_page = performace_graph(len,  performace_dictionary[key_performace])
            list_variables.append(graph_performace_to_page)
            
            graph_performace_to_page.write_image(f'dev/apps/dashboard/dash_reports/graph_reports_{len}_images_{key_performace}_teste.png')       
            report(pdf,f'dev/apps/dashboard/dash_reports/graph_reports_{len}_images_{key_performace}_teste.png')
                
        pdf.output(f'dev/apps/dashboard/dash_reports/performace_report_{today}.pdf', 'F')

        n_clicks = 0
        list_variables.append(n_clicks)

        return list_variables


if __name__ == "__main__":
    #app.run_server(port=8888)
    app.run(debug=True)