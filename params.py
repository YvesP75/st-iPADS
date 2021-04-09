import numpy as np

PI = np.float32(np.pi)

TRAINING_LENGTH = 10000 # how many episodes are run for training

EPISODE_LENGTH = 100 # number of steps per episode

START_LIMITS = 50 # the PADS starts within a disk of 50 m
SPACE_LIMITS = 250 # playground limits for the PADS
LANDING_LIMITS = 10 # error tolerated for landing
LANDING_TARGET = 50 # error tolerated for the target when landing

RHO_INIT = 10  # default para's 2D position
THETA_INIT = PI/2
Z_INIT = 100

SPEED_RHO = 5 # moves forward per step
SPEED_ANGLE = 0
SPEED_Z = 1 # moves down per step

MOVE_TO_METERS = 8 # how many meters per move in a step

MAX_ANGLE = PI/3

LOC = {'Paris':
           {'lat': 48.865879, 'lon': 2.319827},
       'Fonsorbes':
           {'lat': 43.54, 'lon': 1.25},
       'San Francisco':
           {'lat': 37.78, 'lon': -122.41}
       }

TIMESLEEP = 0.2
