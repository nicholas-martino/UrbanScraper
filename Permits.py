import pandas as pd
import plotly.express as px
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
from plotly.graph_objects import Layout
import plotly.io as pio

app = dash.Dash(__name__, external_stylesheets =[dbc.themes.BOOTSTRAP])
pio.templates.default = "plotly_white"

# Plot permit fees
fees = pd.read_csv('/Users/nicholasmartino/Google Drive/PhD/PermitFees.csv')
plot_df = pd.DataFrame()

df = fees[(fees['Item'] == '3(d)<40000') | (fees['Item'] == '2(b)max')| (fees['Item'] == '1max')]
df = df.replace({
    '1max': 'Rezoning, not CD-1',
    '3(d)<40000':'Rezoning CD-1 (<40,000 m2)',
    '2(b)max':'MF Development'
})

# Read data
rez = pd.read_csv('/Volumes/Samsung_T5/Databases/Permits/Rezoning.csv')
rez = rez.sort_values('Approved')
rez['Duration (Days)'] = [dur.days for dur in pd.to_timedelta(rez['Duration'])]
rez = rez[rez['Approved'] > rez['Applied']]
rental_df = pd.DataFrame({
    'Year': [year for year in rez['Year'].unique()],
    'Total': [len(rez.loc[rez['Year'] == year]) for year in rez['Year'].unique()],
    'Rental': [rez.loc[rez['Year'] == year].sum()['Rental'] for year in rez['Year'].unique()]
})
rental_df['Not rental'] = rental_df['Total'] - rental_df['Rental']
rental_df['RentalMS'] = rental_df['Rental']/rental_df['Total']

# Setup figures
timeline = px.timeline(rez, x_start='Applied', x_end='Approved', y='ID', color='Duration (Days)', height=700, title="Rezoning Timeline")
timeline = timeline.update_layout(yaxis={'visible': False, 'showticklabels': True})
cost = px.line(df, x='Year', y='Fee', color='Item', height=350, line_shape="spline", title="Maximum Permit Fees")
rental = px.line(rental_df, x="Year", y=["Rental", "Not rental","Total"], height=350, line_shape="spline", title='Applications')

def style(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family="Roboto Light",
        margin={
            'l': 0,
            'r': 0,
            'b': 0,
            't': 30
        },
        title_x=0.5,
        legend=dict(
            title='',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            orientation='h'
        )
    )
    return fig

timeline, cost, rental = [style(fig) for fig in [timeline, cost, rental]]

content = html.Div(
    style={'margin-left':'1%'},
    id="page-content",
    children=[
        dbc.Row(
            className='pretty_container',
            children=[
                dbc.Col(
                    width={'size': 2},
                    children=[
                        html.Div(
                            className='pretty_container',
                            style={'fontFamily': 'Roboto Light', 'fontSize':14, 'margin-right': 50},
                            children=[
                                """
                                The high availability of credit due to low interest rates; rapid population growth and 
                                foreign investments in real estate have boosted demand in the ownership market. 
                                As a consequence, rental housing have become less attractive for developers.
                                """,
                                html.Br(), html.Br(),
                                """
                                Even though the city cannot change most of those issues,
                                it can decrease permit approval costs specifically for rental market developers by 
                                generating pre-approved mixed use rental housing designs 
                                """,
                                html.Br(), html.Br(),
                                """
                                How money saved from automating early stages of approval process can contribute to the rental affordability crisis?
                                """
                            ]
                        )
                    ]
                ),

                dbc.Col(
                    width={'size': 6, 'offset': 0},
                    children=[
                        dcc.Graph(
                            id='timeline',
                            figure=timeline
                        ),
                    ]
                ),
                dbc.Col(
                    width={'size': 4, 'offset': 0},
                    children=[
                        dcc.Graph(
                            id='cost',
                            figure=cost
                        ),
                        dcc.Graph(
                            id='rental',
                            figure=rental
                        ),
                    ]
                )
            ]
        ),

    ]
)



app.layout = html.Div([
    content,
])

if __name__ == '__main__':
    app.run_server(host='localhost', port=8050, debug=True)
