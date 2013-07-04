from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
import thesis
from thesis.ants.ants import Graph, History

from decodes.extensions.cellular_automata import CA
import random
import datetime

#import PIL
#from PIL import *
#print PIL




#from PIL import Image as PILImage

names =('undeveloped')
colors = ((0,0,0))

def fun(self,n):
    print self.hist[n]
    print "works!"




sc = 1
#p = [1]
m = 10
n = 10
w = False
# p = .5 = sobe_2. p = .75 = sobe_3, p = .25 = sobe_4
param = []
for i in range(10): param.append(0)
param[0] = 25                           # inital probability
param[1] = 4                            # depth
param[2] = 1                            # 0 = no first gen; 1 = first gen
param[3] = 1                            # 0 = full (synchronous); 1 = select one (asynchronous)
param[5] = 40                           # ms per slide
param[6] = 10                           # step size
param[7] = (40,40)                      # model size
param[8] = 0                            # block size
param[9] = (400,400)                    # display size
no_gen = 10


import os
path = os.path.expanduser("~") + os.sep + "_ants_export"
name = "sobe_5"
base_path = "F:\\2013 Summer\\Thesis\\_ants"+"\\"
init_fname = ""
out_fname = ""
rule_fname = "default.txt"

#im = PILImage.open(base_path+'test.jpg')

#print im.getpixel((0,0))

lines = []
fin = open(base_path+'setup.txt')
for line in fin:
    lines.append(line)
    tokens = line.split("=")
    if len(tokens) > 1 :
        switch = tokens[0].strip()
        arg = tokens[1].strip()
    else:
        switch = ''
        arg = ''
#    print switch, arg
    if switch == 'b':
        base_path = arg
        print "path = ",arg
    if switch == 'c':
        colors = eval(arg)
        print "colors = ", colors
    if switch == 'n':
        names = eval(arg)
        print "names = ",names
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
    if switch != "" and switch[0] == 'p':
        t = eval(switch[1:])
        t = t % 10
        param[t] = eval(arg)
        print "Param "+str(t)+" =",param[t],arg
    if switch == 's':
        random.seed(eval(arg))
        print "random seed set to ", eval(arg)
    if switch == "i":
        init_fname = arg.strip()
        print "looking for initial conditions in ", init_fname
    if switch == "o":
        out_fname = arg.strip()
        print "writing initial conditions in ", out_fname
    if switch == "r":
        rule_fname = arg.strip()
        print "rules loaded from ", rule_fname
fin.close()

# convert param list to working params
model_size = Interval(param[7][0],param[7][1])
if param[8] != 0:
    block_size = Interval(param[8][0],param[8][1])
else:
    block_size = Interval(0,0)
display_size = Interval(param[9][0],param[9][1])

# set up colors and states
no_states = min(len(names),len(colors))
prop_values = range(no_states)
prop_colors = []
for c in colors:
    prop_colors.append(Color(c[0]/255.0,c[1]/255.0,c[2]/255.0))

color_dict = dict(zip(prop_values,prop_colors))
state_dict = dict(zip(prop_values,names))


# Create the directory if it doesn't already exist
f_prefix = name+"_"
f_name = f_prefix+'%03d'%model_size.a+"x"+'%03d'%model_size.b+"_"+str(param[0])
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
#r.init_rectgrid(model_size,include_corners=False,wrap=False,cellsize=1)
# r.init_ppm("untitled3",base_path,color_dict)


if init_fname == "":
    r.init_rectgrid(model_size,include_corners=False,wrap=False,cellsize=1)
    if block_size.a == 0:
        init_r = 30 * [0]
        init_r[0] = 1
        r.init_rvals([init_r,[-1],[0]])
    else:
        r.init_block(model_size,block_size, param[0])
    r.to_csv(f_name,path)
else:
    r.init_ppm(init_fname,base_path,color_dict)
#    r.from_csv(init_fname,base_path)

if out_fname != "":
    r.to_csv(out_fname,base_path)


t= History(r)

t.set_rule(base_path+rule_fname)
t.set_params(param)

t.generate(no_gen)

t.set_color_dict(prop_colors)
t.write_svgs(f_name,path, display_size,state_dict)


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