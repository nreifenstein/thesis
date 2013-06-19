
import decodes as dc
from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
#from decodes.io import *
#from decodes.io import outie
import copy
print "ants.py loaded"
import os

## define graph class
class Graph():
    def __init__(self,_list=[],_pts=[],_cells=[],_vals=[]):
        self.links = _list
        self.pts = _pts
        self.cell = _cells
        self.val = _vals
        self._res = len(_vals)

    def init_rectgrid(self,size=Interval(20,20),include_corners=False,wrap=False,cellsize=1):
        self.size = size
        m = size.a
        n = size.b
        cellhalf = cellsize/2
        # initialize lists
        pts = []
        neighbors = []
        cells = []

        # create neighbor list for standard rectangular cell grid
        for j in range(n):
            for i in range(m):
                pts.append([cellhalf + cellsize*i,cellhalf+cellsize*j,0])
                cells.append([cellsize,cellsize,1])
                k = i+j*m
                n_new = []
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        if (abs(di)+abs(dj)) > 0:
                            new_index = ((j+dj)%n)*m+((i+di)%m)
                            if wrap :          # wrap is true
                                if (di == 0) or (dj == 0): n_new.append(new_index)
                                elif include_corners : n_new.append(new_index)
                            else:           # wrap is false
                                if ((i+di) in range(m)) and ((j+dj) in range(n)):
                                    if (di == 0) or (dj == 0) : n_new.append(new_index)
                                    elif include_corners : n_new.append(new_index)
                neighbors.append(n_new)
        self.links = neighbors
        self.pts = pts
        self.cell = cells
        self.val = []
        self._res = len(pts)

    def init_rvals(self,choices=[[0,1]]):
        v = []
        for i in range(self._res):
            vi = []
            for j in choices:
                vi.append(random.choice(j))
            v.append(vi)
        self.val = v

    def to_file(self,fname=os.path.expanduser("~") + os.sep + 'out.txt'):
#        import os
#        path = os.path.expanduser("~") + os.sep + fname
        print "writing to ",fname
        fout = open(fname,'w')
        fout.write(str(self.links)+'\n')
        fout.write(str(self.pts)+'\n')
        fout.write(str(self.val)+'\n')
        fout.write(str(self.cell)+'\n')
        fout.close()

    def from_file(self,fname=os.path.expanduser("~") + os.sep + 'out.txt'):
 #       path = os.path.expanduser("~") + os.sep + fname
        print "reading from ",fname
        fin = open(fname)
        lines = []
        for line in fin:
            lines.append(line)
        fin.close()
        self.links = eval(lines[0])
        self.pts = eval(lines[1])
        self.val = eval(lines[2])
        self.cell = eval(lines[3])


        
    def addcell(self, _links, _pt, _cell,_val):
        self.links.append(_links)
        self.pts.append(_pt)
        self.cell.append(_cell)
        self.val.append(_val)
        return len(self.pts)-1
        
    def update(self):
        for i in range(len(self.links)): self.links[i] = []
        for i in range(len(self.links)-1):
            for j in range(i+1,len(self.links)):
                if touch(self.pts[i],self.cell[i],self.pts[j],self.cell[j]):
                    self.links[i].append(j)
                    self.links[j].append(i)
                    
    def direction(self,i,j):
        p1 = self.pts[i]
        p2 = self.pts[j]
        dx = p1[0]-p2[0]
        dy = p1[1]-p2[1]
        if abs(dx) > abs(dy):
            if dx > 0 : return EAST
            else: return WEST
        else:
            if dy > 0 : return NORTH
            return SOUTH
            
            
    def neighbor(self, i,j):
        if i == j : return False                   
        p1 = self.pts[i]
        p2 = self.pts[j]
        c1 = self.cell[i]
        c2 = self.cell[j]
        #print "p1 ",p1," p2 ",p2," c1 ",c1," c2 ",c2
        dx = abs(p1[0] - p2[0])             ## get absolute value of distances between basepoints
        dy = abs(p1[1] - p2[1])
        dz = abs(p1[2] - p2[2])
        lx = c1[0]/2 + c2[0]/2     ## get sum of widths, height, lengths
        ly = c1[1]/2 + c2[1]/2 
        lz = 0
        if NEIGHBORHOOD_TYPE == MOORE:
           return (dx <= lx + EPSILON) and (dy <= ly + EPSILON) and (dz <= lz+ EPSILON)
        elif NEIGHBORHOOD_TYPE == VON_NEUMANN:
            if abs(dx - lx) < EPSILON:
                return dy < ly - EPSILON
            if abs(dy - ly) < EPSILON:
                return dx < lx - EPSILON
            return False
        else: return False
            
    def divide(self,i,d,q):     ## i = cell index, d = direction, q = fraction
        s1 = [[1-q,1,1],[1,1-q,1],[1-q,1,1],[1,1-q,1]]                  # contains scale factors for new cell
        s2 = [[q,1,1],[1,q,1],[q,1,1],[1,q,1]]          # contains scale factors for remainder

        c1 = a_mult_like(self.cell[i],s1[d])                    # creates new cell sizes
        c2 = a_mult_like(self.cell[i],s2[d])

        b1 = a_add_like(self.pts[i],a_scalar(.5,a_mult_like(c2,v[(d+2)%4])))            # creates new base points
        b2 = a_add_like(self.pts[i],a_scalar(.5,a_mult_like(c1,v[d])))

        new_i = self.addcell([i],b2,c2,self.val[i])                                                      # make new cells
        
        n = self.links[i]                   # get neighbors
        self.links[i] = [new_i]
        self.pts[i] = b1
        self.cell[i] = c1
        for j in n:
            if i in self.links[j] : self.links[j].remove(i)
            for k in [i,new_i]:
                if self.neighbor(j,k):
                    self.links[j].append(k)
                    self.links[k].append(j)
        return True
        
        
    def combine(self,i,j):
        print "testing ",i,j
        if not(self.neighbor(i,j)) : return False          # are not adjacent
        test = False
        for k in [0,1]:                                 # check if they are same h/w
            #print self.pts[i][k],self.pts[j][k]
            if (abs(self.pts[i][k]-self.pts[j][k]) < EPSILON) and (self.cell[i][k] == self.cell[j][k]): test = True
        if not(test) : return False                     # not same h/w
        print i,j," can be combined"
        n_new = a_rem_dup(self.links[i]+self.links[j])
        if i in n_new : n_new.remove(i)
        if j in n_new : n_new.remove(j)
        d = direction(self.pts[j],self.pts[i])
        if d%2 == 0: self.cell[i][0]+=self.cell[j][0]
        else: self.cell[i][1]+=self.cell[j][1]
        self.pts[i] = a_add_like(self.pts[i],a_scalar(.5,a_mult_like(self.cell[j],v[d])))
        self.links[j] = []
        self.links[i] = n_new
        self.val[j]=DELETE
        print "for cell ",j," :",self.links[j], self.val[j]
        for k in n_new:
            if j in self.links[k] : self.links[k].remove(j)
        return True
        
    def find(self,val):
        result = []
        for i in list(range(len(self.val))):
            if self.val[i] == val : result.append(i)
        return result
        
    def n_vals(self,c):
        result = []
        for i in self.links[c]:
            result.append(self.val[i])
        return result
        
    def have_neighbor(self,mp,np):              # create list of cells with (1) vals in mp and (2) neighbors with np
        result = []
        for i in range(len(self.val)):
            if mp == self.val[i]:
                if np in self.n_vals(i):
                    result.append(i)
        return result

    def to_image(self,pixel_res = Interval(20,20),color_dict = {0:Color(0.0),1:Color(1.0)}, default_color = Color(0.0)):
        img = Image(pixel_res,default_color)
        for n, val in enumerate(self.val):
            img._pixels[n] = color_dict[val[0]]
        return img

    def to_svg(self,f_name="svg_out", color_dict = {0:Color(0.0),1:Color(1.0)}, cdim=Interval(500,500), recs=True,nodes=False,links=False):
        # Given a point list and a graph, construct the edges as lines
        svg_out = dc.makeOut(dc.Outies.SVG, f_name, canvas_dimensions=cdim, flip_y = True)
        lines = []
        pts = []
        pts2 = []   # total list, includes empty nodes - use for line list
        recs = []
        coord = self.pts
        
        for i in range(len(coord)): 
            print "checking node ",i
            p = Point(coord[i][0],coord[i][1],coord[i][2])
            pts2.append(p)
            if len(self.links[i]) > 0:
                pts.append(p)
                r = PGon.rectangle(p, self.cell[i][0], self.cell[i][1])
                r.set_fill(color_dict[self.val[i][0]])
                recs.append(r)

                """
        for i in list(range(len(coord))):
            n = self.links[i]
            for j in n:
                if i < j:
                    lines.append(Segment(pts2[i],pts2[j]))
                    """
        print "putting rectangles, ",
        if recs : svg_out.put(recs)
        print "edges, ",
        if links : svg_out.put(lines)
        print "nodes"
        if nodes : svg_out.put(pts)
        print "drawing tp file ",
        svg_out.draw()
        print "done"





def p_index(test, list_of_points):
    ret_value = -1
    for i in list(range(len(list_of_points))):
        if Segment(test, list_of_points[i]).length < .1 :
            return i
    return -1

def direction(p1,p2):
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    if abs(dx) > abs(dy):
        if dx > 0 : return EAST
        else: return WEST
    else:
        if dy > 0 : return NORTH
        return SOUTH

def touch(p1,c1,p2,c2):
    dx = abs(p1[0] - p2[0])             ## get absolute value of distances between basepoints
    dy = abs(p1[1] - p2[1])
    dz = abs(p1[2] - p2[2])
    #print "in touch. dx,dy,dz: ", dx, dy, dz
    lx = c1[0]/2 + c2[0]/2     ## get sum of widths, height, lengths
    ly = c1[1]/2 + c2[1]/2 
    lz = 0
    #print "in touch. lx,ly,lz: ", lx,ly,lz
    if NEIGHBORHOOD_TYPE == MOORE:
        return (dx <= lx + EPSILON) and (dy <= ly + EPSILON) and (dz <= lz+ EPSILON)
    elif NEIGHBORHOOD_TYPE == VON_NEUMANN:
        if abs(dx - lx) < EPSILON:
            return dy < ly - EPSILON
        if abs(dy - ly) < EPSILON:
            return dx < lx - EPSILON
        return False
    else: return False

    ## define array utility functions
def a_add_like(a1,a2):
    result = []
    for i in range(min(len(a1),len(a2))):
        result.append(a1[i]+a2[i])
    return result

def a_sub_like(a1,a2):
    result = []
    for i in range(min(len(a1),len(a2))):
        result.append(a1[i]-a2[i])
    return result

def a_mult_like(a1,a2):
    result = []
    for i in range(min(len(a1),len(a2))):
        result.append(a1[i]*a2[i])
    return result

def a_scalar(s,a1):
    result = []
    for i in range(len(a1)):
        result.append(s*a1[i])
    return result
    
def a_rem_dup(a1):
    result = []
    for i in range(len(a1)):
        dup = False
        for j in range(len(result)):
            if a1[i] == result[j] : dup = True
        if not dup : result.append(a1[i])
    return result

def a_count(v,i,a):
    ret = 0
    for j in a: 
        if j[i] == v : ret += 1
    return ret


## define history class

class History():
    def __init__(self,init_graph=Graph()):
        self.hist = [init_graph]
        self.rule_text = "default.txt"
        self.param = []

    def set_color_dict(self,_color_dict):
        self.color_dict = _color_dict

    def set_rule(self,_string="default.txt"):
        self.rule_text = _string

    def set_params(self,_param):
        self.param = _param

    def rule(self,n=0):
        execfile(self.rule_text)

    def generate(self,gen=1):
        m = self.hist[0].size.a
        n = self.hist[0].size.b
        g = 1
        while g < gen:
            self.add_gen()
            execfile(self.rule_text)
#            self.hist[g].init_rvals()
            g+=1

    def write_images(self, fname="out", base_path=os.path.expanduser("~") + os.sep):
        size = self.hist[0].size
        for i,g in enumerate(self.hist):
            img = g.to_image(size,self.color_dict)
            img.save(fname+str(i), base_path, True)

    def test(self,n=0):
        print self.hist[n]

    def add_gen(self):
        self.hist.append(copy.copy(self.hist[-1]))

