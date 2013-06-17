from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
import thesis
from thesis.ants.ants import Graph

from decodes.extensions.cellular_automata import CA
import random
import datetime

import thesis.PIL as PIL
from thesis.PIL import *


prop_values = (0,1,2)
prop_colors = (Color(1,1,1),Color(1,1,.5),Color(0))
state_list = ('undeveloped','access','built')
no_states = len(state_list)
color_dict = dict(zip(prop_values,prop_colors))
state_dict = dict(zip(prop_values, state_list))

sc = 1
#p = [1]
m = 10
n = 10
w = False
# p = .5 = sobe_2. p = .75 = sobe_3, p = .25 = sobe_4
p1 = .25
no_gen = 10


import os
path = os.path.expanduser("~") + os.sep + "_ants_export"
name = "sobe_5"
base_path = "F:\\2013 Summer\\Thesis\\_ants"+"\\"
init_fname = ""

lines = []
fin = open(base_path+'batch.txt')
for line in fin:
    lines.append(line)
    tokens = line.split("=")
    if len(tokens) > 1 :
        switch = tokens[0].strip()
        arg = tokens[1].strip()
    else:
        switch = ''
        arg = ''
    print switch, arg
    if switch == 'b':
        base_path = arg
        print "path = ",arg
    if switch == 'f':
        name = arg
        print "name = ",arg
    if switch == 'w':
        m = eval(arg)
        print "width = ", arg
    if switch == 'h':
        n = eval(arg)
        print "height = ",arg
    if switch == 'g':
        no_gen = eval(arg)
        print "No of generations = ",arg
    if switch == 'p1':
        p1 = eval(arg)/100.0
        print "Param 1 =",p1,arg
    if switch == 's':
        random.seed(eval(arg))
        print "random seed set to ", eval(arg)
    if switch == "i":
        init_fname = arg.strip()
        print "looking for initial conditions in ", init_fname
fin.close()


# Create the directory if it doesn't already exist
f_prefix = name+"_"
f_name = f_prefix+"_"+str(m)+"x"+str(n)+"_"+str(int(p1*100))
path =base_path+f_name
if not( os.path.exists(path)):
    os.mkdir(path)

# Re-write this text file
fout = open(path+"\\"+f_prefix+".txt",'w')
for line in lines:
    if line[-1] !='\n' : line = line + '\n'
    if line[0] != 'd':
        fout.write(line)
fout.write('d = '+str(datetime.date.today()))
fout.close()

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

temp = []
if init_fname != "":
    fin = open(path+"\\"+init_fname)
    line_in = fin.readline()
    temp = eval(line_in)
if len(temp) == m*n:
    props = temp
else:
    props = []
    for i in range(m*n): 
        if random.uniform(0.0,1.0) < .05 :
            props.append([1,0,0])
        else:
            props.append([0,0,0])
#    props.append(random.choice(range(no_states)))

# Save the initial conditions
fout = open(path+"\\"+f_prefix+"init.txt",'w')
fout.write(str(props))
fout.close()


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
                if random.uniform(0.0,1.0) < p1:
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