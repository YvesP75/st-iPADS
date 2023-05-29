import time
import streamlit as st
from params import *
from stable_baselines3 import SAC
import pandas as pd
import pydeck as pdk
from TwoDimEnv import TwoDimEnv

import gymnasium as gym

def run_episode(env, model, lat_tg, lon_tg, rho_init=RHO_INIT, theta_init=THETA_INIT, zed=Z_INIT):
    '''
    runs the episode given an observation init ... and an env and a model
    :param env: TwoDimEnv
    :param model:
    :param lat_tg:  lattitude de la target (0,0)
    :param lon_tg: longitude de la target (0,0)
    :param rho_init:
    :param theta_init:
    :param zed:
    :return:
    '''
    obs = env.reset(training=False, rho_init=rho_init, theta_init=theta_init, z_init=zed)
    step = 0
    lat, lon = rhotheta_to_latlon(MOVE_TO_METERS * rho_init, theta_init, lat_tg, lon_tg)
    path = [[lon, lat, MOVE_TO_METERS * zed]]
    traj = [lat, lon, MOVE_TO_METERS * zed]
    while step < zed:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)
        rho = obs[0] * env.space_limits
        theta = obs[1] * 2 * PI
        lat, lon = rhotheta_to_latlon(MOVE_TO_METERS*rho, theta, lat_tg, lon_tg)
        traj = np.vstack((traj, [lat, lon, MOVE_TO_METERS*(zed - step)]))
        path.append([lon, lat, MOVE_TO_METERS*(zed - step)])
    df_col = pd.DataFrame(traj, columns=['lat', 'lon', 'zed'])
    df_path = pd.DataFrame([{
        'color': [0, 0, 200, 120],
        'path': path
    }])
    return df_path, df_col

def rhotheta_to_latlon(rho, theta, lat_tg, lon_tg):
    '''
    transforms polar coordinates into lat, lon
    :param rho:
    :param theta:
    :param lat_tg: latitude de la target (0,0)
    :param lon_tg: longitude de la target (0,0)
    :return:
    '''
    z = rho * np.exp(1j * theta)
    lat = np.imag(z)*360/(40075*1000) + lat_tg
    lon = np.real(z)*360/(40075*1000*np.cos(PI/180*lat)) + lon_tg
    return lat, lon


def get_layers(df, df_past, df_target, df_path, df_col):
    '''
    renders the layers to be displayed with df in different formats as entries
    :param df:
    :param df_past:
    :param df_target:
    :param df_path:
    :param df_col:
    :return:
    '''
    return [
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160, 40]',
            get_radius=10,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_target,
            get_position='[lon, lat]',
            get_color='[200, 30, 0]',
            get_radius=65,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_target,
            get_position='[lon, lat]',
            get_color='[255, 255, 255]',
            get_radius=45,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_target,
            get_position='[lon, lat]',
            get_color='[0, 0, 200]',
            get_radius=25,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df_past,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 40]',
            get_radius=10,
        ),
        pdk.Layer(
            type="PathLayer",
            data=df_path,
            pickable=True,
            get_color="color",
            width_scale=20,
            width_min_pixels=1,
            get_path="path",
            get_width=1,
        ),
        pdk.Layer(
            type="ColumnLayer",
            data=df_col,
            get_position=['lon', 'lat'],
            get_elevation="zed",
            elevation_scale=1,
            radius=10,
            get_fill_color=[160, 20, 0, 10],
            pickable=True,
            auto_highlight=True
        ),
    ]


def show():
    '''
    shows the i-PADS in Streamlit
    :return:
    '''
    env = TwoDimEnv()
    model = SAC.load("longModel")
    st.title('Intelligent PADS by hexamind')
    st.write('This is a quick demo of an autonomous Parachute (Precision Air Delivery System) controlled by Reinforcement learning. ')
    st.text('<- Set the starting point')

    st.sidebar.write("Where do you want the parachute to start from?")
    rho = st.sidebar.slider('What distance? (in m)', 0, 3000, 1500) / MOVE_TO_METERS
    theta = 2*PI/360 * st.sidebar.slider('What angle?', 0, 360, 90)
    zed = int(st.sidebar.slider('What elevation? (in m)', 0, 1200, 600) / MOVE_TO_METERS)

    location = st.sidebar.radio("Location", ['San Francisco', 'Paris', 'Puilaurens'])
    lat_tg = LOC[location]['lat']
    lon_tg = LOC[location]['lon']
    df_path, df_col = run_episode(env, model, lat_tg, lon_tg, rho_init=rho, theta_init=theta, zed=zed)
    st.sidebar.write(
        'If you like to play, you will probably find some starting points where the parachute is out of control :) '
        'No worries, we have plenty more efficient models at www.hexamind.com ')

    df_target = pd.DataFrame({'lat': [lat_tg], 'lon': [lon_tg]})
    deck_map = st.empty()
    pitch = st.slider('pitch', 0, 100, 50)
    initial_view_state = pdk.ViewState(
            latitude=lat_tg,
            longitude=lon_tg,
            zoom=12,
            pitch=pitch
    )
    deck_map.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=initial_view_state
    ))
    df_pathi = df_path.copy()
    for i in range(zed):
        df_pathi['path'][0] = df_path['path'][0][0:i+1]
        layers = get_layers(df_col[i:i+1], df_col[0:i], df_target, df_pathi, df_col[0:i+1])
        deck_map.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=initial_view_state,
            layers=layers
        ))
        time.sleep(TIMESLEEP)


show()

# to uncomment for debug
#show_print()
