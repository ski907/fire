#Original code and concept by Christian
#https://scipython.com/blog/the-forest-fire-model/
#Accessed July 2021
#
#Adapted by C. Engel, July 2021

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
import time

EMPTY, TREE, FIRE = 0, 1, 2


x_dim = 50
y_dim = 50

size = (y_dim, x_dim)
inset = tuple([f-2 for f in size])


initial_forest_density = st.sidebar.slider(
                            'Initial Forest Density',
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5
                            )
lightning_prob = st.sidebar.slider(
                            'Probability of Lightning Strike',
                            min_value=0.0,
                            max_value=0.01,
                            value=0.0001,
                            step=0.0001,
                            format='%f'
                            
                            )
growth_prob = st.sidebar.slider(
                            'Probability of Forest Expansion',
                            min_value=0.0,
                            max_value=0.2,
                            value=0.01
                            )
burn_advance_prob = st.sidebar.slider(
                            'Probability of Burn Advancement',
                            min_value=0.0,
                            max_value=1.0,
                            value=0.9
                            )
diagonal_fire_prob = 0.573

X=np.zeros(size)
X[1:y_dim-1, 1:x_dim-1] = np.random.random(size=inset) < initial_forest_density

def sim(X):
    
    X1 = np.zeros(size)
    for x in range(1,x_dim-1):
        for y in range(1,y_dim-1):
            if X[y,x] == EMPTY:
                if TREE in [X[adj] for adj in adjacent(y,x)]:
                    if np.random.random() <= growth_prob:
                        X1[y,x] = TREE
                if TREE in [X[dia] for dia in diagonal(y,x)]:
                    if np.random.random() <= growth_prob:
                        if np.random.random() <= diagonal_fire_prob:
                            X1[y,x] = TREE
                            
            if X[y,x] == TREE:
                if np.random.random() <= lightning_prob:
                    X1[y,x] = FIRE
                elif FIRE in [X[adj] for adj in adjacent(y,x)]:
                    if np.random.random() <= burn_advance_prob:
                        X1[y,x] = FIRE
                elif FIRE in [X[dia] for dia in diagonal(y,x)]:
                    if np.random.random() <= burn_advance_prob:
                        if np.random.random() <= diagonal_fire_prob:
                            X1[y,x] = FIRE
                else:
                    X1[y,x] = TREE
    return X1
                        
                

def adjacent(y: int, x: int):
    adj = [(1,0),(0,1),(-1,0),(0,-1)]
    return [(y+dy,x+dx) for dy,dx in adj]

def diagonal(y: int, x: int):
    dia = [(1,1),(-1,1),(-1,-1),(1,-1)]
    return [(y+dy,x+dx) for dy,dx in dia]

colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'orange']
cmap = colors.ListedColormap(colors_list)
bounds = [0,1,2,3]
norm = colors.BoundaryNorm(bounds, cmap.N)



fig = plt.figure(figsize=(25/3, 6.25))
ax = fig.add_subplot(111)
ax.set_axis_off()
im = ax.imshow(X, cmap=cmap, norm=norm)#, interpolation='nearest')
the_plot = st.pyplot(plt)


# The animation function: called to produce a frame for each generation.
def animate():
    im.set_data(animate.X)
    animate.X = sim(animate.X)
    the_plot.pyplot(plt)
    
# Bind our grid to the identifier X in the animate function's namespace.
animate.X = X

run = True

while run:
    animate()
    #time.sleep(0.001)

