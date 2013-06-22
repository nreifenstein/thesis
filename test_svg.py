from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
import thesis
from thesis.ants.ants import Graph

from decodes.extensions.cellular_automata import CA
import random, os
import cStringIO

filepath = os.path.expanduser("~") + os.sep + "test_svg.svg"


c = 10
n = 100

print "drawing svg to "+filepath

buffer = cStringIO.StringIO()
svg_size = ""
svg_size = 'width="'+str(1000)+'" height="'+str(1000)+'"'

buffer.write('<svg '+svg_size+' xmlns="http://www.w3.org/2000/svg" version="1.1">\n')

type = 'polygon'

for i in range(n):
    for j in range(n):
                
        if (i+j)%2 == 0:
            style = 'fill:rgb(255,255,127);stroke-width:0;stroke:none'
        else:
            style = 'fill:rgb(0,127,0);stroke-width:0;stroke:none'
        pts = [[c*i,c*j],[c*(i+1),c*j],[c*(i+1),c*(j+1)],[c*i,c*(j+1)]]
        point_string = " ".join([str(v[0])+","+str(v[1]) for v in pts])
        atts = 'points="'+point_string+'"'
        buffer.write('<polygon '+atts+' style="'+style+'"/>\n')


buffer.write('</svg>')

# write buffer to file
fo = open(filepath, "wb")
fo.write( buffer.getvalue() )
fo.close()
buffer.close()

raw_input("press enter...")