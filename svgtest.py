import decodes as dc
from decodes.core import *
from decodes.core import PGon

p = []
xmax = 500
ymax = 500
r = dc.core.PGon.rectangle(Point(xmax/2,ymax/2),xmax,ymax)

for n in range(1):
    outie = dc.makeOut(dc.Outies.SVG, "svg_"+str(n), canvas_dimensions=Interval(xmax,ymax), flip_y = True)
    p.append(outie)

    scale = 50

    p[n].put(r)

    for x in range(10):
        for y in range(10):
            pt = Point(x*scale,y*scale)
            pt.set_color(Color(1,0,0))
            pt.set_weight(2*y+1.5)
            p[n].put(pt)
    p[n].draw()

        
'''   
def func(t):
    return Point(t*scale,math.sin(t)*scale)
crv = Curve(func,Interval(0,math.pi*2))

pl = crv.surrogate
pl.set_color(1,0,0)
pl.set_weight(5.0)

outie.put(pl)        

outie.draw()
'''