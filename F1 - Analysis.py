#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import datetime

import numpy as np 
import pandas as pd
import scipy.stats as sp

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


# In[2]:


races = pd.read_csv('C:/Users/Home/Desktop/F1/races.csv')
status = pd.read_csv('C:/Users/Home/Desktop/F1/status.csv')
drivers = pd.read_csv('C:/Users/Home/Desktop/F1/drivers.csv')
results = pd.read_csv('C:/Users/Home/Desktop/F1/results.csv')
constructors = pd.read_csv('C:/Users/Home/Desktop/F1/constructors.csv')


# In[3]:


concat_driver_name = lambda x: f"{x.forename} {x.surname}" 

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)


# In[4]:


# Preparing F1 history victories dataset
results_copy = results.set_index('raceId').copy()
races_copy = races.set_index('raceId').copy()

results_copy = results_copy.query("position == '1'")
results_copy['position'] = 1 # casting position 1 to int 

results_cols = ['driverId', 'position']
races_cols = ['date']
drivers_cols = ['driver', 'driverId']

results_copy = results_copy[results_cols]
races_copy = races_copy[races_cols]
drivers_copy = drivers[drivers_cols]

f1_victories = results_copy.join(races_copy)
f1_victories = f1_victories.merge(drivers_copy, on='driverId', how='left')

# Victories cumulative sum
f1_victories = f1_victories.sort_values(by='date')

f1_victories['victories'] = f1_victories.groupby(['driverId']).cumsum()   

# Getting the top five f1 biggest winners drivers id
f1_biggest_winners = f1_victories.groupby('driverId').victories.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_ids = [driver for driver, race in f1_biggest_winners.index]

# Dataset ready
f1_victories_biggest_winners = f1_victories.query(f"driverId == {f1_biggest_winners_ids}")


# In[5]:


# Prepare dataset to plot

cols = ['date', 'driver', 'victories']
winner_drivers = f1_victories_biggest_winners.driver.unique()

colors = {
    'Alain Prost': '#d80005', 
    'Ayrton Senna': '#ffffff', 
    'Michael Schumacher': '#f71120',
    'Sebastian Vettel': '#10428e',
    'Lewis Hamilton': '#e6e6e6'
}

winners_history = pd.DataFrame()

# Including other drivers races date (like a cross join matrix, 
# but cosidering column "victories" in a shift operation) 
for driver in winner_drivers:
    # Current driver victories
    driver_history = f1_victories_biggest_winners.query(f"driver == '{driver}'")[cols]
    
    # Other drivers list
    other_drivers = winner_drivers[winner_drivers != driver]
    other_drivers = list(other_drivers)
    
    # Other drivers victories
    other_driver_history = f1_victories_biggest_winners.query(f"driver == {other_drivers}")[cols]
    
    # Renaming other drivers victories to current driver
    other_driver_history['driver'] = driver
    
    # This isn't current driver victory, so receive zero to "shift" operation
    other_driver_history['victories'] = 0    
    
    driver_history = pd.concat([driver_history, other_driver_history])

    driver_history['color'] = colors[driver]
    
    # Sorting by date to correct "shift" operation
    driver_history.sort_values(by='date', inplace=True)
    
    # Reset index to get the last row (index-1) when necessary
    driver_history.reset_index(inplace=True)
    
    # Iterating each row for remain current driver victory when 
    # race date isn't the current driver victory
    for index, row in driver_history.iterrows():
        if not row['victories'] and index-1 > 0:
            driver_history.loc[index, 'victories'] = driver_history.loc[index-1, 'victories']
        
    # Plot dataset ready
    winners_history = pd.concat([winners_history, driver_history])


# In[6]:


# Plots the F1 race wons animated chart 
fig = go.Figure()

fig = px.bar(
    winners_history, 
    x='victories', 
    y='driver',
    color='driver',
    color_discrete_sequence=winners_history.color.unique(),
    orientation='h',
    animation_frame="date",
    animation_group="driver",
)

# Bar border line color
fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))

# X axis range
fig.update_layout(xaxis=dict(range=[0, 100]))

# Setting title
fig.update_layout(title_text="Race wins in F1 history between the top 5 winners drivers")

# Animation: Buttons labels and animation duration speed
fig.update_layout(
    updatemenus = [
        {
            "buttons": [
                # Play
                {
                    "args": [
                        None, 
                        {
                            "frame": {
                                "duration": 100, 
                                 "redraw": False
                            }, 
                            "fromcurrent": True,
                            "transition": {
                                "duration": 100, 
                                "easing": "linear"
                            }
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                # Pause
                {
                    "args": [
                        [None], 
                        {
                            "frame": {
                                "duration": 0, 
                                "redraw": False
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": 0
                            }
                        }
                    ],
                    "label": "Pause",
                    "method": "animate"
                }
            ]
        }
    ]
)

fig.show()


# In[8]:



# Dict for map drivers by id
winner_drivers_ids = f1_victories_biggest_winners[['driverId', 'driver']].drop_duplicates()
winner_drivers_map = {}

for _, row in winner_drivers_ids.iterrows():
    winner_drivers_map[row['driverId']] = row['driver']


# In[9]:


# Pole positions dataset
f1_biggest_winners_poles = results.query(f"driverId == {f1_biggest_winners_ids} & grid == 1")[['driverId', 'grid']]

# Driver name mapping
f1_biggest_winners_poles['driver'] = f1_biggest_winners_poles.driverId.map(winner_drivers_map)
f1_biggest_winners_poles['color'] = f1_biggest_winners_poles.driver.map(colors)

# Sum cumulative poles
f1_biggest_winners_poles['total_poles'] = f1_biggest_winners_poles.groupby(['driverId']).cumsum()   

# Total pole positions by winner drivers
f1_biggest_winners_total_poles = f1_biggest_winners_poles.groupby('driver').total_poles.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_total_poles = pd.DataFrame(f1_biggest_winners_total_poles).reset_index()

f1_biggest_winners_total_poles['color'] = f1_biggest_winners_total_poles.driver.map(colors)


# In[10]:


# Plot pole positions
fig = px.bar(
    f1_biggest_winners_total_poles, 
    x='driver', 
    y='total_poles',
    color='driver',
    color_discrete_sequence=f1_biggest_winners_total_poles.color
)

# Bar border line color
fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))

# Setting title
fig.update_layout(title_text="Pole positions between the top 5 race winners drivers")

fig.show()


# In[11]:


# Hamilton data
hamilton = drivers.query("driverRef == 'hamilton'")


# In[12]:


# Driver races dataframe

def get_races_by_driver_id(driver_id):
    columns = ['grid', 'position', 'raceId', 'constructorId', 'statusId']

    driver_races = results.query(f'driverId == {driver_id}')
    driver_races = driver_races[columns]

    driver_races.set_index('raceId', inplace=True)

    driver_races = driver_races.join(races.set_index('raceId')['date'])

    driver_races['is_pole'] = driver_races.grid == 1
    driver_races['is_first_place'] = driver_races.position == '1'

    driver_races.sort_values(by='date', inplace=True)

    driver_races['poles'] = driver_races.is_pole.cumsum()
    driver_races['races_won'] = driver_races.is_first_place.cumsum()

    driver_races = driver_races.set_index('constructorId').join(constructors.set_index('constructorId')['name'])
    driver_races = driver_races.rename(columns={'name': 'constructor'})
    
    driver_races = pd.merge(status, driver_races, on=['statusId', 'statusId']).sort_values(by='date')
    driver_races = driver_races.rename(columns={'status': 'race_status'})
    
    return driver_races
    
hamilton_races = get_races_by_driver_id(hamilton.driverId[0])


# In[15]:


# Dataframes to plot
mclaren = hamilton_races.query('constructor == "McLaren"')
mercedes = hamilton_races.query('constructor == "Mercedes"')

# To join gap between constructors
mclaren = pd.concat([mclaren, mercedes.head(1)])


# In[17]:


# Pole positions
mclaren_poles  = go.Scatter(x=mc_laren.date, y=mc_laren.poles, fill='tozeroy', name="McLaren", marker=dict(color="#D89A8C"))
mercedes_poles = go.Scatter(x=mercedes.date, y=mercedes.poles, fill='tozeroy', name="Mercedes", marker=dict(color="#C2C2C2"))

# Races won
mclaren_wons  = go.Scatter(x=mc_laren.date, y=mc_laren.races_won, fill='tozeroy', name="McLaren", marker=dict(color="#cb7967"), showlegend=False)
mercedes_wons = go.Scatter(x=mercedes.date, y=mercedes.races_won, fill='tozeroy', name="Mercedes", marker=dict(color="#b3b3b3"), showlegend=False)

# Drawing figure
fig = make_subplots(
    rows=2, 
    cols=1, 
    subplot_titles=("Pole positions","Races win")
)

fig.add_trace(mclaren_poles, row=1, col=1)
fig.add_trace(mercedes_poles, row=1, col=1)

fig.add_trace(mclaren_wons, row=2, col=1)
fig.add_trace(mercedes_wons, row=2, col=1)

fig.update_layout(
    height=600,
    title_text="Careers numbers",
    title_font_size=20,
    hovermode='x',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="left",
        x=0.01
    ),
)

fig.update_yaxes(range=[0, 100])


# In[18]:


# Linear regression

def linear_regression(x, y):
    x = np.array([
        time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple())
        for date in x
    ])

    y = np.array(y)

    slope, intercept, _, _, _ = sp.linregress(x,y)
    y_regression = (slope * x) + intercept
    
    return y_regression, slope


# In[19]:


# Constructors linear regressions

mc_laren_poles_y_reg, mc_laren_poles_slope = linear_regression(mc_laren.date, mc_laren.poles)
mercedes_poles_y_reg, mercedes_poles_slope = linear_regression(mercedes.date, mercedes.poles)

mc_laren_wons_y_reg, mc_laren_wons_slope = linear_regression(mc_laren.date, mc_laren.races_won)
mercedes_wons_y_reg, mercedes_wons_slope = linear_regression(mercedes.date, mercedes.races_won)

# Linear regression traces lines

mc_laren_poles_linreg = go.Scatter(x=mc_laren.date, y=mc_laren_poles_y_reg, line=dict(color='#984634', dash='dash'), hoverinfo='skip')
mercedes_poles_linreg = go.Scatter(x=mercedes.date, y=mercedes_poles_y_reg, line=dict(color='#999999', dash='dash'), hoverinfo='skip')

mc_laren_wons_linreg = go.Scatter(x=mc_laren.date, y=mc_laren_wons_y_reg, line=dict(color='#984634', dash='dash'), hoverinfo='skip')
mercedes_wons_linreg = go.Scatter(x=mercedes.date, y=mercedes_wons_y_reg, line=dict(color='#999999', dash='dash'), hoverinfo='skip')


# In[20]:


fig = make_subplots(
    rows=2, 
    cols=2, 
    subplot_titles=(
        "Pole positions Mc Laren",
        "Pole positions Mercedes",
        "Races win Mc Laren",
        "Races win Mercedes",
    )
)

fig.add_trace(mc_laren_poles_linreg, row=1, col=1)
fig.add_trace(mclaren_poles, row=1, col=1)

fig.add_trace(mercedes_poles_linreg, row=1, col=2)
fig.add_trace(mercedes_poles, row=1, col=2)

fig.add_trace(mc_laren_wons_linreg, row=2, col=1)
fig.add_trace(mclaren_wons, row=2, col=1)

fig.add_trace(mercedes_wons_linreg, row=2, col=2)
fig.add_trace(mercedes_wons, row=2, col=2)

fig.update_layout(
    height=900,
    title_text="Linear regression slopes",
    title_font_size=20,
    hovermode='x',
    showlegend=False,
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="left",
        x=0.01
    ),
)
fig.add_annotation(
    x=list(mc_laren.date)[50], 
    y=mc_laren_poles_y_reg[50],
    xref='x1', 
    yref='y1',
    text=f"Linear Regression Slope: {mc_laren_poles_slope}",
    showarrow=True,
    bordercolor="#929191",
    borderwidth=2,
    borderpad=4,
    ay=-80,
    ax=-30,
    arrowcolor="#929191",
    arrowwidth=2
)

fig.add_annotation (
    x=list(mercedes.date)[50], 
    y=mercedes_poles_y_reg[50],
    xref='x2', 
    yref='y2',
    text=f"Linear Regression Slope: {mercedes_poles_slope}",
    showarrow=True,
    bordercolor="#ffffff",
    borderwidth=2,
    borderpad=4,
    ay=80,
    ax=30,
    arrowcolor="#ffffff",
    arrowwidth=2
)

fig.add_annotation (
    x=list(mc_laren.date)[50], 
    y=mc_laren_wons_y_reg[50],
    xref='x3', 
    yref='y3',
    text=f"Linear Regression Slope: {mc_laren_wons_slope}",
    showarrow=True,
    bordercolor="#929191",
    borderwidth=2,
    borderpad=4,
    ay=-80,
    ax=-30,
    arrowcolor="#929191",
    arrowwidth=2
)

fig.add_annotation (
    x=list(mercedes.date)[50], 
    y=mercedes_wons_y_reg[50],
    xref='x4', 
    yref='y4',
    text=f"Linear Regression Slope: {mercedes_wons_slope}",
    showarrow=True,
    bordercolor="#ffffff",
    borderwidth=2,
    borderpad=4,
    ay=80,
    ax=30,
    arrowcolor="#ffffff",
    arrowwidth=2
)


# In[21]:


hamilton_mercedes_races = hamilton_races.query("constructor == 'Mercedes'")

# Preparing mercedes drivers datasets
mercedes_id = constructors.query("name == 'Mercedes'")['constructorId']
mercedes_id = int(mercedes_id)

mercedes_races = results.query(f"constructorId == {mercedes_id}")
mercedes_races = mercedes_races.merge(races, on='raceId')

first_hamilton_race = hamilton_mercedes_races.date.min()

hamilton_mercedes_team_mates_id = mercedes_races.query(f"date >= '{first_hamilton_race}' & driverId != {hamilton.driverId[0]}")
hamilton_mercedes_team_mates_id = hamilton_mercedes_team_mates_id.driverId.unique()

hamilton_mercedes_team_mates = []
    
for driver_id in hamilton_mercedes_team_mates_id:
    team_mate_races = get_races_by_driver_id(driver_id).query(f"constructor == 'Mercedes' & date >= '{first_hamilton_race}'")
    
    team_mate_races['driver'] = drivers.query(f"driverId == {driver_id}").driver.unique()[0]
    
    hamilton_mercedes_team_mates.append(team_mate_races)


# In[23]:


def team_mate_comparisson(index, team_mate_color):
    team_mate_name = hamilton_mercedes_team_mates[index].driver.unique()[0]

    last_team_mate_race = hamilton_mercedes_team_mates[index].date.max()
    first_team_mate_race = hamilton_mercedes_team_mates[index].date.min()
    
    hamilton_color = '#b3b3b3'
    
    # Only races in team mate period 
    hamilton_mercedes_team_mate_races = hamilton_mercedes_races.query(f"'{first_team_mate_race}' <= date <= '{last_team_mate_race}'").copy()

    # Reseting races won sum
    hamilton_mercedes_team_mate_races['races_won'] = hamilton_mercedes_team_mate_races.is_first_place.cumsum()
    # Races won
    team_mate_races_wons = go.Scatter(
        name=team_mate_name, 
        fill='tozeroy', 
        marker=dict(color=team_mate_color),
        x=hamilton_mercedes_team_mates[index].date, 
        y=hamilton_mercedes_team_mates[index].races_won 
    )

    hamilton_races_wons = go.Scatter(
        name='Lewis Hamilton',
        fill='tozeroy',
        marker=dict(color=hamilton_color),
        x=hamilton_mercedes_team_mate_races.date, 
        y=hamilton_mercedes_team_mate_races.races_won      
    )
    
    # Drawing figure
    fig = make_subplots(
        rows=2, 
        cols=2, 
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        horizontal_spacing = 0.1,
        vertical_spacing = 0.2,
        subplot_titles= (
            "Races win", 
            "Standings distribution",
            "Standings distribution",
        )
    )

    fig.add_trace(hamilton_races_wons,row=1, col=1)
    fig.add_trace(team_mate_races_wons, row=1, col=1)
    
    # Standings
    hamilton_standings = go.Box(
        name='Lewis Hamilton', 
        showlegend=False, 
        marker=dict(color=hamilton_color),
        y=hamilton_mercedes_team_mate_races.position
    )

    team_mate_standings = go.Box(
        name=team_mate_name, 
        showlegend=False, 
        marker=dict(color=team_mate_color),
        y=hamilton_mercedes_team_mates[index].position, 
    )
    
    fig.add_trace(hamilton_standings, row=2, col=1)
    fig.add_trace(team_mate_standings,row=2, col=2)

    fig.update_layout(
        height=700,
        margin=dict(b=10),
        title_text=f"Lewis Hamilton and {team_mate_name}",
        title_font_size=20,
        hovermode='x',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01
        ),
    )

    fig.update_layout(    
        {    
            'yaxis': {'range': [0, 50]},
            'yaxis2':{'range': [20, 0], "nticks": 10},
            'yaxis3':{'range': [20, 0], "nticks": 10},
        }
    )
    
    # Drawing figure race status
    fig_race_status = make_subplots(
        rows=1, 
        cols=2, 
        specs=[
            [{}, {}]
        ],
        horizontal_spacing = 0.1,
        subplot_titles= (
            "Race status",
            "Race status",
        )
    )
    
     # Race status
    hamilton_race_status_data  = hamilton_mercedes_team_mate_races.race_status.value_counts().to_frame()
    team_mate_race_status_data = hamilton_mercedes_team_mates[index].race_status.value_counts().to_frame()

    hamilton_race_status_data['percent'] = hamilton_mercedes_team_mate_races.race_status.value_counts(normalize=True)
    team_mate_race_status_data['percent'] = hamilton_mercedes_team_mates[index].race_status.value_counts(normalize=True)
    
    hamilton_race_status = go.Bar(
        name='Lewis Hamilton', 
        orientation='h',
        showlegend=False, 
        marker=dict(color=hamilton_color),
        x=hamilton_race_status_data.race_status,
        y=hamilton_race_status_data.index,
        hoverinfo='x+y+text',
        hovertext=hamilton_race_status_data.percent
    )

    team_mate_race_status = go.Bar(
        name=team_mate_name, 
        orientation='h',
        showlegend=False, 
        marker=dict(color=team_mate_color),
        x=team_mate_race_status_data.race_status,
        y=team_mate_race_status_data.index, 
        hoverinfo='x+y+text',
        hovertext=team_mate_race_status_data.percent
    )
    
    fig_race_status.add_trace(hamilton_race_status, row=1, col=1)
    fig_race_status.add_trace(team_mate_race_status,row=1, col=2)

    fig_race_status.update_layout(
        height=300,
        margin=dict(t=20),
        showlegend=False,
    )
    
    fig_race_status.update_layout(    
        {    
            'yaxis1':{'autorange': 'reversed'},
            'yaxis2':{'autorange': 'reversed'}
        }
    )
    
    fig.show()
    fig_race_status.show()


# In[24]:


team_mate_comparisson(index=0, team_mate_color='#6ed6d1')


# In[25]:


team_mate_comparisson(index=1, team_mate_color='#0e0000')


# In[ ]:




