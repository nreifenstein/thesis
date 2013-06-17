from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
import thesis
from thesis.ants.ants import Graph

from decodes.extensions.cellular_automata import CA
import random
import datetime

import PIL
import _imaging
import Image as PILimage

im = PILimage.open("animation.gif")
im.seek(1) # skip to the second frame
try:
    while 1:
        im.seek(im.tell()+1)
        # do something to im
except EOFError:
    pass # end of sequence




raw_input("press enter...")