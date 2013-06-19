from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
import thesis
from thesis.ants.ants import Graph, History

from decodes.extensions.cellular_automata import CA
import random
import datetime
import cPickle

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
param = []
for i in range(10): param.append(0)
param[1] = .25
no_gen = 10


import os
path = os.path.expanduser("~") + os.sep + "_ants_export"
name = "sobe_5"
base_path = "F:\\2013 Summer\\Thesis\\_ants"+"\\"
init_fname = ""
rule_fname = "default.txt"

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
        param[1] = eval(arg)/100.0
        print "Param 1 =",param[1],arg
    if switch == 's':
        random.seed(eval(arg))
        print "random seed set to ", eval(arg)
    if switch == "i":
        init_fname = arg.strip()
        print "looking for initial conditions in ", init_fname
    if switch == "r":
        rule_fname = arg.strip()
        print "rules loaded from ", rule_fname
fin.close()


# Create the directory if it doesn't already exist
f_prefix = name+"_"
f_name = f_prefix+"_"+str(m)+"x"+str(n)+"_"+str(int(param[1]*100))
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

### Main Function

r = Graph()
r.init_rectgrid(Interval(m,n),include_corners=False,wrap=False,cellsize=1)
init_r = 99 * [0]
init_r.append(1)

r.init_rvals([init_r])
t= History(r)

#execfile(base_path+rule_fname)
#fin = open(base_path+rule_fname)
#test_string = fin.readlines()
#fin.close()

#exec(test_string)

t.set_rule(base_path+rule_fname)
t.set_params(param)

t.generate(no_gen)

t.set_color_dict(prop_colors)
t.write_images(f_name,path)



def a_count(v,i,a):
    ret = 0
    for j in a: 
        if j[i] == v : ret += 1
    return ret


def print_attributes(obj):
    for attr in obj.__dict__:
        print attr, getattr(obj, attr)
    
"""
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
"""

raw_input("press enter...")