from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
import thesis
from thesis.ants.ants import Graph

from decodes.extensions.cellular_automata import CA
import random

prop_values = (0,1,2)
prop_colors = (Color(1,1,1),Color(1,1,.5),Color(0))
state_list = ('undeveloped','access','built')
no_states = len(state_list)
color_dict = dict(zip(prop_values,prop_colors))
state_dict = dict(zip(prop_values, state_list))


no_gen = 10


import os
path = os.path.expanduser("~") + os.sep + "_ants_export"
f_prefix = "sobe_5_"

sc = 1
p = [1]
m = 100
n = 100
w = False
# p = .5 = sobe_2. p = .75 = sobe_3, p = .25 = sobe_4
p_house = .25

def a_count(v,i,a):
    ret = 0
    for j in a: 
        if j[i] == v : ret += 1
    return ret


def print_attributes(obj):
    for attr in obj.__dict__:
        print attr, getattr(obj, attr)
    

## this component will create a graph assuming an m x n grid of cells with a cell size of sc
## 05.22.13 added cell into data type

DELETE = -1
UNDEVELOPED = 0
ACCESS = 1
BUILT = 11


WEST = 0
NORTH = 1
EAST = 2
SOUTH =3
dirs =["WEST","NORTH","EAST","SOUTH"]
v = [[-1,0,0],[0,1,0],[1,0,0],[0,-1,0]]                                 # contains vector displacements for cardinal directions WNES
scales = [[.5,1],[1,.5],[.5,1],[1,.5]]
deltas = [[.5,0],[0,-.5],[-.5,0],[0,.5]]

MOORE = 0                                   ## 8-cell neighborhood
VON_NEUMANN = 1                             ## 4-cell neighborhood
NEIGHBORHOOD_TYPE = VON_NEUMANN

EPSILON = sc * .001                  ## tolerance for touch


m_domain = list(range(m))
n_domain = list(range(n))
cell_half = sc/2

# initialize lists
pts = []
neighbors = []
cells = []

# create neighbor list for standard rectangular cell grid
for j in n_domain:
    for i in m_domain:
        pts.append([cell_half + sc*i,cell_half+sc*j,0])
        cells.append([sc,sc,1])
        k = i+j*m
        n_new = []
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                if (abs(di)+abs(dj)) > 0:
                    if w :          # wrap is true
                        new_index = n_domain[(j+dj)%n]*m+m_domain[(i+di)%m]
                        if (di == 0) or (dj == 0): n_new.append(new_index)
                        elif NEIGHBORHOOD_TYPE == MOORE : n_new.append(new_index)
                    else:           # wrap is false
                        if ((i+di) in m_domain) and ((j+dj) in n_domain):
                            if (di == 0) or (dj == 0) : n_new.append((j+dj)*m+(i+di))
                            elif NEIGHBORHOOD_TYPE == MOORE : n_new.append((j+dj)*m+(i+di))
        neighbors.append(n_new)


props = []
for i in range(m*n): 
    if random.uniform(0.0,1.0) < .01 :
        props.append([1,0,0])
    else:
        props.append([0,0,0])
#    props.append(random.choice(range(no_states)))

g_temp = []
g_temp = Graph(neighbors,pts,cells,props)


gen = 0
h=[]
h.append(g_temp)

# history loop
while gen < no_gen:
    img = h[gen].to_image(Interval(m,n),color_dict)
    img.save(f_prefix+str(gen), path, True)
    new_props = []
    for j in range(m*n): new_props.append([0,0,0])
    for i in range(m*n):
        if h[gen].prop[i][0] == 0:
            n_temp = h[gen].n_props(i)
            c_temp = a_count(1,0,n_temp)
            if c_temp == 1:
                if random.uniform(0.0,1.0) < p_house:
                    new_props[i][0] = 2
                else:
                    new_props[i][0] = 1
            elif c_temp == 2:
                new_props[i][0] = 1
            elif c_temp == 3:
                new_props[i][0] = 2
            elif c_temp == 4:
                new_props[i][0] = 1
            else:
                new_props[i][0] = 0
        else: new_props[i] = h[gen].prop[i]
    h.append(Graph(neighbors,pts,cells,new_props))
    gen+=1

raw_input("press enter...")