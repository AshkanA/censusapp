import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
from fuzzywuzzy import process

# Load the Excel file
file_path = './data/CensusData.xlsx'  # Updated data source
data = pd.read_excel(file_path)

# Data preprocessing
data.columns = data.columns.str.strip()
data['City'] = data['City'].astype(str)
data['City'] = data['City'].replace({'11170': 'Huizen', '11253': 'Sint Eustatius'})
data = data[~data['City'].isin(['11276', '11445'])]

city_codes = data['City'].unique()
city_name_code_pairs = sorted([(city_code, city_code) for city_code in city_codes])

total_population_vars = [
    'JuridischAanwezige_Bevolking',
    'FeitelijkAanwezige_Bevolking',
    'TijdelijkAanwezige_Bevolking',
    'TijdelijkAfwezige_Bevolking',
    'Married Women', 'Married Men', 'Single Women', 'Single Men', 'Widow Women', 'Widow Men'
]

sub_filters = {
    'JuridischAanwezige_Bevolking': {
        'JuridischAanwezige_Mannen_BinnenKom': 'Males Inside',
        'JuridischAanwezige_Mannen_BuitenKom': 'Males Outside',
        'JuridischAanwezige_Vrouwen_BinnenKom': 'Females Inside',
        'JuridischAanwezige_Vrouwen_BuitenKom': 'Females Outside'
    },
    'FeitelijkAanwezige_Bevolking': {
        'FeitelijkAanwezige_Mannen_BinnenKom': 'Males Inside',
        'FeitelijkAanwezige_Mannen_BuitenKom': 'Males Outside',
        'FeitelijkAanwezige_Vrouwen_BinnenKom': 'Females Inside',
        'FeitelijkAanwezige_Vrouwen_BuitenKom': 'Females Outside'
    },
    'TijdelijkAanwezige_Bevolking': {
        'TijdelijkAanwezige_Mannen_BinnenKom': 'Males Inside',
        'TijdelijkAanwezige_Mannen_BuitenKom': 'Males Outside',
        'TijdelijkAanwezige_Vrouwen_BinnenKom': 'Females Inside',
        'TijdelijkAanwezige_Vrouwen_BuitenKom': 'Females Outside'
    },
    'TijdelijkAfwezige_Bevolking': {
        'TijdelijkAfwezige_Mannen_BinnenKom': 'Males Inside',
        'TijdelijkAfwezige_Mannen_BuitenKom': 'Males Outside',
        'TijdelijkAfwezige_Vrouwen_BinnenKom': 'Females Inside',
        'TijdelijkAfwezige_Vrouwen_BuitenKom': 'Females Outside'
    }
}

translated_headings = {
    'JuridischAanwezige_Bevolking': 'Legally Present Population',
    'FeitelijkAanwezige_Bevolking': 'Actually Present Population',
    'TijdelijkAanwezige_Bevolking': 'Temporarily Present Population',
    'TijdelijkAfwezige_Bevolking': 'Temporarily Absent Population',
    'Married Women': 'Married Women',
    'Married Men': 'Married Men',
    'Single Women': 'Single Women',
    'Single Men': 'Single Men',
    'Widow Women': 'Widow Women',
    'Widow Men': 'Widow Men'
}

app = Dash(__name__)
app.config.suppress_callback_exceptions = True

colors = {
    'background': '#FFFFFF',  # White background
    'text': '#000000',  # Black text
    'accent': '#008000',  # Green accent color
    'title': '#008000'  # Green title color
}

search_style = {
    'width': '350px',
    'height': '40px',
    'font-family': 'Arial, sans-serif',
    'font-size': '16px',
    'border-radius': '20px',
    'border': '2px solid #000000',
    'margin-right': '20px'
}

dropdown_style = {
    'width': '350px',
    'height': '42px',
    'font-family': 'Arial, sans-serif',
    'font-size': '16px',
    'border-radius': '20px',
    'border': '1px solid #000000',
    'box-shadow': 'none',
    'padding': '0 0px',
    'background-color': 'white'
}

app.layout = html.Div(
    style={'backgroundColor': colors['background'], 'textAlign': 'center', 'padding': '20px'},
    children=[
        html.Div(
            id='title-container',
            children=[
                html.H1('Population Trends Explorer', style={'color': colors['title'], 'font-weight': 'bold', 'font-size': '42px', 'font-family': 'Arial, sans-serif'})
            ]
        ),
        html.Div(
            id='search-container',
            style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'},
            children=[
                dcc.Input(
                    id='search-bar',
                    type='text',
                    placeholder='Search for a city...',
                    style=search_style
                ),
                dcc.Dropdown(
                    id='city-dropdown',
                    options=[],  # Empty initially
                    value=None,
                    style=dropdown_style,
                    className='custom-dropdown'
                ),
                html.Span('Compare', style={'font-family': 'Arial, sans-serif', 'font-size': '16px', 'margin': '0 20px', 'color': colors['text']}),  # Visual indication to compare cities
                dcc.Dropdown(
                    id='city-dropdown-compare',
                    options=[],  # Empty initially
                    value=None,
                    style=dropdown_style,
                    className='custom-dropdown'
                ),
                html.Button('Download CSV', id='btn-csv', style={'margin-left': '20px', 'height': '42px', 'color': colors['text']}),
                dcc.Download(id='download-dataframe-csv')
            ]
        ),
        dcc.Store(id='filtered-options-store', data=[]),
        html.Div(
            id='sub-filters-container',
            style={'display': 'flex', 'justifyContent': 'center', 'margin-top': '20px'},
            children=[
                html.Div(
                    style={'margin': '0 20px'},
                    children=[
                        html.H4('Legally Present Population', style={'color': colors['title'], 'font-family': 'Arial, sans-serif', 'font-size': '16px'}),
                        dcc.Checklist(
                            id='juridisch-subfilters',
                            options=[{'label': label, 'value': var} for var, label in sub_filters['JuridischAanwezige_Bevolking'].items()],
                            value=[],  # Default to none selected
                            style={'color': colors['text'], 'font-family': 'Arial, sans-serif', 'font-size': '14px'},
                            inline=True
                        )
                    ]
                ),
                html.Div(
                    style={'margin': '0 20px'},
                    children=[
                        html.H4('Actually Present Population', style={'color': colors['title'], 'font-family': 'Arial, sans-serif', 'font-size': '16px'}),
                        dcc.Checklist(
                            id='feitelijk-subfilters',
                            options=[{'label': label, 'value': var} for var, label in sub_filters['FeitelijkAanwezige_Bevolking'].items()],
                            value=[],  # Default to none selected
                            style={'color': colors['text'], 'font-family': 'Arial, sans-serif', 'font-size': '14px'},
                            inline=True
                        )
                    ]
                ),
                html.Div(
                    style={'margin': '0 20px'},
                    children=[
                        html.H4('Temporarily Present Population', style={'color': colors['title'], 'font-family': 'Arial, sans-serif', 'font-size': '16px'}),
                        dcc.Checklist(
                            id='tijdelijk-aanwezige-subfilters',
                            options=[{'label': label, 'value': var} for var, label in sub_filters['TijdelijkAanwezige_Bevolking'].items()],
                            value=[],  # Default to none selected
                            style={'color': colors['text'], 'font-family': 'Arial, sans-serif', 'font-size': '14px'},
                            inline=True
                        )
                    ]
                ),
                html.Div(
                    style={'margin': '0 20px'},
                    children=[
                        html.H4('Temporarily Absent Population', style={'color': colors['title'], 'font-family': 'Arial, sans-serif', 'font-size': '16px'}),
                        dcc.Checklist(
                            id='tijdelijk-afwezige-subfilters',
                            options=[{'label': label, 'value': var} for var, label in sub_filters['TijdelijkAfwezige_Bevolking'].items()],
                            value=[],
                            style={'color': colors['text'], 'font-family': 'Arial, sans-serif', 'font-size': '14px'},
                            inline=True
                        )
                    ]
                )
            ]
        ),
        html.Div(
            id='graph-container',
            style={'height': '600px'}
        ),
        html.Div(
            id='time-slider-container',
            style={'margin-top': '20px'},
            children=[
                dcc.RangeSlider(
                    id='time-slider',
                    min=data['Year'].min(),
                    max=data['Year'].max(),
                    value=[data['Year'].min(), data['Year'].max()],
                    marks={str(year): str(year) for year in data['Year'].unique()},
                    step=None
                )
            ]
        ),
        html.Div(
            id='visualization-controls',
            style={'display': 'flex', 'justifyContent': 'center', 'margin-top': '20px'},
            children=[
                dcc.Dropdown(
                    id='visualization-dropdown',
                    options=[
                        {'label': 'Line Chart', 'value': 'line'},
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Pie Chart', 'value': 'pie'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Table', 'value': 'table'}
                    ],
                    value='line',
                    style={'width': '200px', 'font-family': 'Arial, sans-serif', 'font-size': '16px', 'color': colors['text']}
                ),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[],  # Empty initially
                    value=None,
                    style={'width': '200px', 'margin-left': '20px', 'font-family': 'Arial, sans-serif', 'font-size': '16px', 'color': colors['text']}
                )
            ]
        ),
        html.Div(
            id='legend-checkboxes',
            style={'display': 'flex', 'justifyContent': 'center', 'margin-top': '20px'},
            children=[
                dcc.Checklist(
                    id='legend-checklist',
                    options=[{'label': translated_headings[var], 'value': var} for var in total_population_vars],
                    value=total_population_vars,  # Default to all selected
                    style={'color': colors['text'], 'font-family': 'Arial, sans-serif', 'font-size': '14px'},
                    inline=True
                )
            ]
        ),
        dcc.Download(id="download-csv"),
        html.Link(
            rel='stylesheet',
            href='/assets/custom.css'
        )
    ]
)

def filter_data(city_code, variables, year_range):
    """Filter data based on city code, variables, and year range."""
    try:
        return data[(data['City'] == city_code) & 
                    (data['Variable'].isin(variables)) & 
                    (data['Year'] >= year_range[0]) & 
                    (data['Year'] <= year_range[1])].sort_values('Year')
    except Exception as e:
        print(f"Error filtering data: {e}")
        return pd.DataFrame()

def generate_graph(filtered_data, visualization_type, city_names, selected_subfilters, selected_year):
    """Generate graph based on the filtered data and visualization type."""
    fig = go.Figure()
    for city_name, city_code in city_names.items():
        city_filtered_data = filtered_data[filtered_data['City'] == city_code]
        city_filtered_data = city_filtered_data.sort_values('Year')
        if visualization_type == 'line':
            for variable in total_population_vars + selected_subfilters:
                city_data = city_filtered_data[city_filtered_data['Variable'] == variable]
                legend_name = f'{translated_headings.get(variable, variable)} ({city_name})'
                hover_text = city_data['Year'].astype(str) + '<br>' + legend_name + '<br>' + city_data['Total'].astype(str)
                fig.add_trace(go.Scatter(
                    x=city_data['Year'],
                    y=city_data['Total'],
                    mode='lines+markers',
                    marker=dict(size=6),
                    line=dict(width=3),
                    name=legend_name,
                    text=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor="white", font_size=16)
                ))
        elif visualization_type == 'bar':
            for variable in total_population_vars + selected_subfilters:
                city_data = city_filtered_data[city_filtered_data['Variable'] == variable]
                legend_name = f'{translated_headings.get(variable, variable)} ({city_name})'
                hover_text = city_data['Year'].astype(str) + '<br>' + legend_name + '<br>' + city_data['Total'].astype(str)
                fig.add_trace(go.Bar(
                    x=city_data['Year'],
                    y=city_data['Total'],
                    name=legend_name,
                    text=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor="white", font_size=16)
                ))
        elif visualization_type == 'pie':
            year_data = city_filtered_data[city_filtered_data['Year'] == selected_year]
            aggregated_data = year_data.groupby('Variable').agg({'Total': 'sum'}).reindex(total_population_vars)
            hover_text = [f'{translated_headings[var]}: {aggregated_data.loc[var, "Total"]}' for var in total_population_vars]
            fig.add_trace(go.Pie(
                labels=[translated_headings[var] for var in total_population_vars],
                values=aggregated_data['Total'],
                text=hover_text,
                hoverinfo='text',
                hoverlabel=dict(bgcolor="white", font_size=16)
            ))
        elif visualization_type == 'histogram':
            for variable in total_population_vars + selected_subfilters:
                city_data = city_filtered_data[city_filtered_data['Variable'] == variable]
                legend_name = f'{translated_headings.get(variable, variable)} ({city_name})'
                hover_text = city_data['Year'].astype(str) + '<br>' + legend_name + '<br>' + city_data['Total'].astype(str)
                fig.add_trace(go.Histogram(
                    x=city_data['Year'],
                    y=city_data['Total'],
                    name=legend_name,
                    text=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor="white", font_size=16)
                ))
        elif visualization_type == 'table':
            table_data = city_filtered_data[city_filtered_data['Variable'].isin(total_population_vars + selected_subfilters)]
            table_data = table_data[['Year', 'Variable', 'Total']].sort_values(['Year', 'Variable'])
            table_data['Variable'] = table_data['Variable'].map(translated_headings)
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(table_data.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[table_data.Year, table_data.Variable, table_data.Total],
                           fill_color='lavender',
                           align='left',
                           height=30)
            )])
        else:
            for variable in total_population_vars + selected_subfilters:
                city_data = city_filtered_data[city_filtered_data['Variable'] == variable]
                legend_name = f'{translated_headings.get(variable, variable)} ({city_name})'
                hover_text = city_data['Year'].astype(str) + '<br>' + legend_name + '<br>' + city_data['Total'].astype(str)
                fig.add_trace(go.Scatter(
                    x=city_data['Year'],
                    y=city_data['Total'],
                    mode='lines+markers',
                    marker=dict(size=6),
                    line=dict(width=3),
                    name=legend_name,
                    text=hover_text,
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor="white", font_size=16)
                ))

    fig.update_layout(
        title=f'Population Trends for {", ".join(city_names.keys())}',
        xaxis_title='Year',
        yaxis_title='Population',
        font=dict(family='Arial, sans-serif', size=14, color=colors['text']),
        margin=dict(t=50, b=50, l=50, r=50),
        height=600,
        autosize=True,
        showlegend=True,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        xaxis=dict(type='category', showline=True, linewidth=1, linecolor='black', categoryorder='category ascending'),  # Ensure chronological order
        yaxis=dict(showline=True, linewidth=1, linecolor='black')
    )
    return fig

@app.callback(
    Output('filtered-options-store', 'data'),
    [Input('search-bar', 'value')]
)
def update_store(search_value):
    """Update filtered options store based on search bar value."""
    if not search_value:
        return [{'label': city_name, 'value': city_code} for city_name, city_code in city_name_code_pairs]

    search_value = search_value.lower()
    matches = process.extract(search_value, [city_name for city_name, _ in city_name_code_pairs], limit=10)
    filtered_city_name_code_pairs = [
        {'label': match[0], 'value': next(city_code for city_name, city_code in city_name_code_pairs if city_name == match[0])}
        for match in matches
    ]
    return filtered_city_name_code_pairs

@app.callback(
    Output('city-dropdown', 'options'),
    Output('city-dropdown', 'value'),
    Output('city-dropdown-compare', 'options'),
    Output('city-dropdown-compare', 'value'),
    Output('year-dropdown', 'options'),
    Output('year-dropdown', 'value'),
    [Input('filtered-options-store', 'data')]
)
def update_dropdown(filtered_options):
    """Update city dropdown options based on filtered options."""
    if not filtered_options:
        return [], None, [], None, [], None

    selected_city_code = filtered_options[0]['value']
    filtered_data = filter_data(selected_city_code, total_population_vars, [data['Year'].min(), data['Year'].max()])
    years = sorted(filtered_data['Year'].unique())
    year_options = [{'label': str(year), 'value': year} for year in years]

    return filtered_options, selected_city_code, filtered_options, None, year_options, years[0]

@app.callback(
    Output('year-dropdown', 'style'),
    Output('time-slider-container', 'style'),
    [Input('visualization-dropdown', 'value')]
)
def display_controls(visualization_type):
    """Conditionally display year dropdown and time slider based on visualization type."""
    if visualization_type == 'pie':
        return {'width': '200px', 'margin-left': '20px', 'font-family': 'Arial, sans-serif', 'font-size': '16px', 'color': colors['text']}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'margin-top': '20px'}

@app.callback(
    Output('graph-container', 'children'),
    [Input('city-dropdown', 'value'),
     Input('city-dropdown-compare', 'value'),
     Input('juridisch-subfilters', 'value'),
     Input('feitelijk-subfilters', 'value'),
     Input('tijdelijk-aanwezige-subfilters', 'value'),
     Input('tijdelijk-afwezige-subfilters', 'value'),
     Input('visualization-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('legend-checklist', 'value'),
     Input('time-slider', 'value')]
)
def update_graph(selected_city_code, selected_city_compare_code, selected_juridisch, selected_feitelijk, selected_tijdelijk_aanwezige, selected_tijdelijk_afwezige, visualization_type, selected_year, selected_legend, year_range):
    """Update graph based on selected filters, legend checkboxes, and time slider."""
    if not selected_city_code or (visualization_type == 'pie' and not selected_year):
        return []

    selected_subfilters = selected_juridisch + selected_feitelijk + selected_tijdelijk_aanwezige + selected_tijdelijk_afwezige
    filtered_data = filter_data(selected_city_code, selected_legend + selected_subfilters, year_range)

    city_names = {selected_city_code: selected_city_code}
    if selected_city_compare_code:
        compare_filtered_data = filter_data(selected_city_compare_code, selected_legend + selected_subfilters, year_range)
        filtered_data = pd.concat([filtered_data, compare_filtered_data])
        filtered_data = filtered_data.sort_values('Year')
        city_names[selected_city_compare_code] = selected_city_compare_code

    fig = generate_graph(filtered_data, visualization_type, city_names, selected_subfilters, selected_year)

    return dcc.Graph(figure=fig)

@app.callback(
    Output("download-csv", "data"),
    Input("btn-csv", "n_clicks"),
    [State('city-dropdown', 'value'),
     State('city-dropdown-compare', 'value'),
     State('juridisch-subfilters', 'value'),
     State('feitelijk-subfilters', 'value'),
     State('tijdelijk-aanwezige-subfilters', 'value'),
     State('tijdelijk-afwezige-subfilters', 'value'),
     State('legend-checklist', 'value'),
     State('time-slider', 'value')]
)
def download_csv(n_clicks, selected_city_code, selected_city_compare_code, selected_juridisch, selected_feitelijk, selected_tijdelijk_aanwezige, selected_tijdelijk_afwezige, selected_legend, year_range):
    """Download CSV data based on selected filters."""
    if not selected_city_code:
        return None
    
    selected_subfilters = selected_juridisch + selected_feitelijk + selected_tijdelijk_aanwezige + selected_tijdelijk_afwezige
    filtered_data = filter_data(selected_city_code, selected_legend + selected_subfilters, year_range)

    if selected_city_compare_code:
        compare_filtered_data = filter_data(selected_city_compare_code, selected_legend + selected_subfilters, year_range)
        filtered_data = pd.concat([filtered_data, compare_filtered_data])
        filtered_data = filtered_data.sort_values('Year')
    
    return dcc.send_data_frame(filtered_data.to_csv, f"{selected_city_code}_filtered_data.csv", index=False)

if __name__ == '__main__':
    app.run_server(debug=True)
