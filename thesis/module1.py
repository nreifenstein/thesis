import decodes as dc
from decodes.core import *
from decodes.extensions import packing
import random
from time import time
import math

def is_same(line1, line2):
    if line1.spt == line2.spt and line1.ept == line2.ept: return True
    if line1.spt == line2.ept and line1.ept == line2.spt: return True
    return False

def is_on(pt,ln):
    ''' assumes pt is co_linear
    '''
    v = Vec(pt,ln.spt)
    if v == Vec(0,0): v = Vec(pt, ln.ept)
    if ln._vec.is_coincident(v) or ln._vec.is_coincident(v.inverted()):
        return ((pt >= ln.spt) and (pt <= ln.ept)) or ((pt >= ln.ept) and (pt <= ln.spt))
    else:
        return False

def join_segments(line1, line2):
    result = []
    if line1._vec.is_coincident(line2._vec) or line1._vec.is_coincident(line2._vec.inverted()):
        if is_on(line1.spt,line2) or is_on(line1.ept,line2) or is_on(line2.spt,line1) or is_on(line2.ept,line1):
            pts = [line1.spt, line1.ept, line2.spt, line2.ept]
            result = Segment(min(pts), max(pts))
    return result

def join_segments3(line1, line2):
    result = []
    if line1._vec.is_coincident(line2._vec) or line1._vec.is_coincident(line2._vec.inverted()):
        spt = min(line2.spt,line2.ept)
        ept = max(line2.spt,line2.ept)
        if ((line1.spt >= spt) and (line1.spt <= ept)) or ((line1.ept >= spt) and (line1.ept <= ept)):
            pts = [line1.spt, line1.ept, line2.spt, line2.ept]
            result = Segment(min(pts), max(pts))
    return result

def join_segments2(line1, line2):
    result = []
    if (line1.spt == line2.spt) or (line1.spt == line2.ept) or (line1.ept == line2.spt) or (line1.ept == line2.ept):
        if line1._vec.is_coincident(line2._vec) or (line1._vec.is_coincident(line2._vec.inverted())):
            pts = [line1.spt, line1.ept, line2.spt, line2.ept]
            result = Segment(min(pts), max(pts))
    return result

p_list_1 = [Point(0,0),Point(100,0), Point(100,100), Point(0,100)]
a = PGon( p_list_1)
s1 = Segment(Point(0,0),Point(100,0))
s2 = Segment(Point(150,0),Point(100,0))
s3 = Segment(Point(50,0),Point(75,0))
#p1 = PLine([Point(0,0),Point(100,0)])
#p2 =PLine([Point(100,0),Point(200,0)])
#p3 = PLine([Point(100,100),Point(200,100)])
print join_segments(s1,s3)

p_list_1 = [Point(0,0),Point(100,0), Point(100,100), Point(0,100)]
a = PGon( p_list_1)
b = PGon([Point(100,0),Point(200,0), Point(200,100), Point(100,100)])
c = PGon([Point(100,100),Point(200,100), Point(200,200), Point(100,200)])



e = a.edges

e.extend(b.edges)
e.extend(c.edges)

r = [Segment(e[0].spt, e[0].ept)]

for i in range(1,len(e)):
    flag = True
    for t in r:
        if is_same(e[i],t):
            r.remove(t)
            flag = False
            break
    if flag :
        r.append(Segment(e[i].spt, e[i].ept))


'''
for i in range(1,len(e)):
    flag = True
    for t in r:
        test = join_segments(t,e[i])
        if test !=[]:
            r.remove(t)
            r.append(test)
            flag = False
            break
    if flag :
        r.append(Segment(e[i].spt, e[i].ept))
'''

print


outie = dc.makeOut(dc.Outies.SVG, "svg_out", canvas_dimensions=Interval(1000,500), flip_y = True)
scale = 50

for x in r:
    x.set_color(Color(1,1,0))
    x.set_weight(1)
    outie.put(x)
               

outie.draw()





raw_input("Hit any key...")
