"""
    Created my_asltest1.py as means to test the various modules
    

"""


import numpy as np
import pandas as pd
from asl_data import AslDb
from asl_utils import test_features_tryit


asl = AslDb() # initializes the database
#print(asl.df.head()) # displays the first five rows of the asl database, indexed by video and frame
#print(asl.df.ix[98,1])  # look at the data available for an individual frame

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

#print(asl.df.head())  # the new feature 'grnd-ry' is now in the frames dictionary
# test the code
#test_features_tryit(asl)

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
 #show a single set of features for a given (video, frame) tuple
#print([asl.df.ix[98,1][v] for v in features_ground])

training = asl.build_training(features_ground)
#print("Training words: {}".format(training.words))
#print(training.get_word_Xlengths('CHOCOLATE'))

df_means = asl.df.groupby('speaker').mean()
#print(df_means)
#asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
df_std = asl.df.groupby('speaker').std()


# TODO add features for normalized by speaker values of left, right, x, y
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd

asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])
asl.df['nose-x-mean']= asl.df['speaker'].map(df_means['nose-x'])
asl.df['nose-y-mean']= asl.df['speaker'].map(df_means['nose-y'])

asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])

asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean'])/asl.df['right-x-std']
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean'])/asl.df['right-y-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean'])/asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean'])/asl.df['left-y-std']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']


# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
#  Note that theta will be in radians, and using arctan of x,y (instead of y,x)
#   reference start is expected to be straight vertical line from origin(nose)

asl.df['polar-rr'] = np.sqrt(np.square(asl.df['grnd-ry']) + np.square(asl.df['grnd-rx']))
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])

asl.df['polar-lr'] = np.sqrt(np.square(asl.df['grnd-lx']) + np.square(asl.df['grnd-ly']))
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']


# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

asl.df['delta-rx'] = asl.df.groupby('speaker')['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df.groupby('speaker')['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df.groupby('speaker')['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df.groupby('speaker')['left-y'].diff().fillna(0)

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']


# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like
#  for now, just offset the delta by 1 to bypass any unittest
#   Assume that values of x & y further away from normalised x,y, of right & left
#   are less likely to be valid


asl.df['custom-rx'] = asl.df['grnd-rx'] / asl.df['norm-rx']
asl.df['custom-ry'] = asl.df['grnd-ry'] / asl.df['norm-ry']
asl.df['custom-lx'] = asl.df['grnd-lx'] / asl.df['norm-lx']
asl.df['custom-ly'] = asl.df['grnd-ly'] / asl.df['norm-ly']


# TODO define a list named 'features_custom' for building the training set

features_custom = ['custom-rx', 'custom-ry', 'custom-lx', 'custom-ly']


print(asl.df.head())





