
import decodes as dc
from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
#from decodes.io import *
#from decodes.io import outie
import copy
import csv
print "ants.py loaded"
import os,cStringIO


def is_same(line1, line2):
    if line1[0] == line2[0] and line1[1] == line2[1]: return True
    if line1[0] == line2[1] and line1[1] == line2[0]: return True
    return False


## define graph class
class Graph():
    def __init__(self,_list=[],_pts=[],_cells=[],_vals=[]):
        new_links = []
        for i in _list:
            new_link = []
            for j in i:
                new_link.append(j)
            new_links.append(new_link)
        self.link = new_links
        self.pts = copy.copy(_pts)
        self.cell = copy.copy(_cells)
        new_vals = []
        for i in _vals:
            new_val = []
            for j in i:
                new_val.append(j)
            new_vals.append(new_val)
        self.val = new_vals
        self.size = Interval(0,len(new_vals))

    @property
    def _res(self):  return len(self.val)

    def init_rectgrid(self,size=Interval(20,20),include_corners=False,wrap=False,cellsize=1):
        self.size = size
        m = int(size.a)
        n = int(size.b)
        cellhalf = cellsize/2.0
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
        self.link = neighbors
        self.pts = pts
        self.cell = cells
        self.val = []
        self._res = len(pts)

    def init_rvals(self,choices=[[0,1],[-1],[0]]):
        v = []
        for i in range(self._res):
            vi = []
            for j in choices:
                vi.append(random.choice(j))
            v.append(vi)
        self.val = v

    def init_block(self, model_size=Interval(40,40), block_size=Interval(20,20), init_val = 1, no_vals = 1, increment = 1.0):
        mod_x = block_size.a + 1
        mod_y = block_size.b + 1
        c_x = (model_size.a + block_size.a) / 2
        c_y = (model_size.b + block_size.b) / 2
        v = int(model_size.a * model_size.b) * [0]
        nx = 0.0
        ny = 0.0

        for i in range(int(model_size.a)):
            if ((i-c_x) % mod_x == 0): nx += 1.0/2.0
            else: nx += increment/2.0
            ny = 0.0
            for j in range(int(model_size.b)):
                
                k = i+j*int(model_size.a)
                v[k] = no_vals * [0]
                v[k][3] = -1
                flag = 0
                if ((i-c_x) % mod_x == 0): flag += 2
                if ((j-c_y) % mod_y == 0): 
                    flag += 1
                    ny += 1.0 / 2.0
                else:
                    ny += increment / 2.0
                if flag == 0:
                    v[k][0] = 0
                    v[k][3] = k
                    self.cell[k] = [increment, increment,1.0]
                if flag == 1:
                    v[k][0] = 1
                    self.cell[k] = [increment, 1.0, 1.0]
                if flag == 2:
                    v[k][0] = 1
                    self.cell[k] = [1.0,increment,1.0]
                if flag == 3:
                    v[k][0] =1
                    self.cell[k] = [1.0,1.0,1.0]
                self.pts[k] = [nx, ny, 0.0]
                ny += self.cell[k][1] / 2.0
            nx += self.cell[k][0]/2.0

        self.val = v

    def init_ppm(self, init_fname = "in", path=os.path.expanduser("~") + os.sep,color_dict = {0:Color(0.0),1:Color(1.0)}):
        fin = open(path+init_fname+'.ppm')
        lines = []
        for line in fin:
            lines.append(line)
        fin.close()
        print len(lines)
        a = 1

        model_dimensions = lines[2]
        dims = model_dimensions.strip().split(' ')
        m = int(dims[0])
        n = int(dims[1])
        self.init_rectgrid(Interval(m,n),include_corners=False,wrap=False,cellsize=1)
        counter = 4
        vals = []
        for k in list(range(m*n)): vals.append([0,-1,0])
        for i in range(m):
            for j in range(n):
                col = []
                for c in range(counter, counter+3):
                    col.append(int(lines[c]))

                v = 0
                for k in color_dict:
                    if (color_dict[k].r*255 == col[0]) and (color_dict[k].g*255 == col[1]) and (color_dict[k].b*255 == col[2]):
                        v = k
#                print "color = ", col, " = ", v
                vals[j+(m-i-1)*m] = [v,-1,0]
                counter +=3
        self.val = vals

    def to_file(self,fname=os.path.expanduser("~") + os.sep + 'out.txt'):
#        import os
#        path = os.path.expanduser("~") + os.sep + fname
        print "writing to ",fname
        fout = open(fname,'w')
        fout.write(str(self.link)+'\n')
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
        self.link = eval(lines[0])
        self.pts = eval(lines[1])
        self.val = eval(lines[2])
        self.cell = eval(lines[3])
      
    def addcell(self, _link, _pt, _cell,_val):
        self.link.append(_link)
        self.pts.append(_pt)
        self.cell.append(_cell)
        self.val.append(_val)
        return len(self.pts)-1
        
    def update(self):
        for i in range(len(self.link)): self.link[i] = []
        for i in range(len(self.link)-1):
            for j in range(i+1,len(self.link)):
                if touch(self.pts[i],self.cell[i],self.pts[j],self.cell[j]):
                    self.link[i].append(j)
                    self.link[j].append(i)
                    
    def direction(self,i,j , tol = .001):
        WEST = 0
        NORTH = 1
        EAST = 2
        SOUTH = 3
        pt_i = self.pts[i]
        cell_i = self.cell[i]
        pt_j = self.pts[j]
        cell_j = self.cell[j]
        if abs((pt_i[0]-cell_i[0]/2.0) - (pt_j[0]+cell_j[0]/2.0)) < tol : return EAST
        if abs((pt_j[0]-cell_j[0]/2.0) - (pt_i[0]+cell_i[0]/2.0)) < tol : return WEST
        if abs((pt_j[1]-cell_j[1]/2.0) - (pt_i[1]+cell_i[1]/2.0)) < tol : return SOUTH
        return NORTH
           
    def neighbor(self, i,j, neighborhood_type = 1, epsilon = .01):
        if i == j : return False                   
        p1 = self.pts[i]
        p2 = self.pts[j]
        c1 = self.cell[i]
        c2 = self.cell[j]
        #print "p1 ",p1," p2 ",p2," c1 ",c1," c2 ",c2
        dx = abs(p1[0] - p2[0])             ## get absolute value of distances between basepoints
        dy = abs(p1[1] - p2[1])
        dz = abs(p1[2] - p2[2])
        lx = c1[0]/2.0 + c2[0]/2.0     ## get sum of widths, height, lengths
        ly = c1[1]/2.0 + c2[1]/2.0 
        lz = 0
        if neighborhood_type == 0:  # MOORE
           return (dx <= lx + epsilon) and (dy <= ly + epsilon) and (dz <= lz+ epsilon)
        elif neighborhood_type == 1: # VON NEUMANN
            if abs(dx - lx) < epsilon:
                return dy < ly - epsilon
            if abs(dy - ly) < epsilon:
                return dx < lx - epsilon
            return False
        else: return False
            
    def divide(self,i,d,q, increment = True):     ## i = cell index, d = direction, q = fraction
        if increment:
            if increment > self.cell[i][d%2] : return i
            q = q / self.cell[i][d%2]
        v = [[-1,0,0],[0,1,0],[1,0,0],[0,-1,0]]                                 # contains vector displacements for cardinal directions WNES
        scales = [[.5,1],[1,.5],[.5,1],[1,.5]]
        deltas = [[.5,0],[0,-.5],[-.5,0],[0,.5]]

        s1 = [[1-q,1,1],[1,1-q,1],[1-q,1,1],[1,1-q,1]]                  # contains scale factors for new cell
        s2 = [[q,1,1],[1,q,1],[q,1,1],[1,q,1]]          # contains scale factors for remainder

        c1 = a_mult_like(self.cell[i],s1[d])                    # creates new cell sizes
        c2 = a_mult_like(self.cell[i],s2[d])

        b1 = a_add_like(self.pts[i],a_scalar(.5,a_mult_like(c2,v[(d+2)%4])))            # creates new base points
        b2 = a_add_like(self.pts[i],a_scalar(.5,a_mult_like(c1,v[d])))

        new_i = self.addcell([i],b2,c2,copy.copy(self.val[i]))                          # make new cells
        
        n = self.link[i]                   # get neighbors
        self.link[i] = [new_i]
        self.pts[i] = b1
        self.cell[i] = c1
        for j in n:
            if i in self.link[j] : self.link[j].remove(i)
            for k in [i,new_i]:
                if self.neighbor(j,k):
                    self.link[j].append(k)
                    self.link[k].append(j)
        return new_i
               
    def combine(self,i,j):
        print "testing ",i,j
        if not(self.neighbor(i,j)) : return False          # are not adjacent
        test = False
        for k in [0,1]:                                 # check if they are same h/w
            #print self.pts[i][k],self.pts[j][k]
            if (abs(self.pts[i][k]-self.pts[j][k]) < epsilon) and (self.cell[i][k] == self.cell[j][k]): test = True
        if not(test) : return False                     # not same h/w
        print i,j," can be combined"
        n_new = a_rem_dup(self.link[i]+self.link[j])
        if i in n_new : n_new.remove(i)
        if j in n_new : n_new.remove(j)
        d = direction(self.pts[j],self.pts[i])
        if d%2 == 0: self.cell[i][0]+=self.cell[j][0]
        else: self.cell[i][1]+=self.cell[j][1]
        self.pts[i] = a_add_like(self.pts[i],a_scalar(.5,a_mult_like(self.cell[j],v[d])))
        self.link[j] = []
        self.link[i] = n_new
        self.val[j]=DELETE
        print "for cell ",j," :",self.link[j], self.val[j]
        for k in n_new:
            if j in self.link[k] : self.link[k].remove(j)
        return True
        
    def find(self,val):
        result = []
        for i in list(range(len(self.val))):
            if self.val[i] == val : result.append(i)
        return result
        
    def n_vals(self,c,k=0):                         # create list of the values of the neighbors of c
        result = []
        for i in self.link[c]:
            if k == -1:
                result.append(self.val[i])
            else:
                result.append(self.val[i][k])
        return result

    def neighbors(self, i, no_states = 2, include_other_parcels = True):      # create a list that stores index/direction pairs for each possible state
        result = no_states * [[]]
        for k in self.link[i]:
            if (self.val[k][0] == 1) or (self.val[k][3] == self.val[i][3]) or include_other_parcels:
                d = self.direction(k,i)
                if result[self.val[k][0]] == [] : result[self.val[k][0]] = [(k,d)]
                else: result[self.val[k][0]].append((k,d))
                a = 1
        return result
        
    def have_neighbor(self,mp,np):              # create list of cells with (1) vals in mp and (2) neighbors with np
        result = []
        for i in range(len(self.val)):
            if mp == self.val[i]:
                if np in self.n_vals(i):
                    result.append(i)
        return result

    def parcel_list(self):
        # get number if parcels
        no_par = -1
        for i in range(len(self.val)):
            if self.val[i][3] > no_par : no_par = self.val[i][3]

        # initialize result
        result = []
        for i in range(no_par+1) : result.append([])

        # fill out list
        for i in range(len(self.val)):
            if self.val[i][3] != -1: result[self.val[i][3]].append(i)

        return result

    def to_dc_svg(self,f_name="svg_out", path = os.path.expanduser("~"), color_dict = {0:Color(0.0),1:Color(1.0)}, cdim=Interval(500,500), draw_recs=True,draw_nodes=False,draw_link=False):
        # this uses Decodes. OK for nodes and link, slow for cells.
        svg_out = dc.makeOut(dc.Outies.SVG, f_name, path, canvas_dimensions=cdim, flip_y = False)
        lines = []
        pts = []
        pts2 = []   # total list, includes empty nodes - use for line list
        recs = []
        coord = self.pts

        for i in range(len(self.pts)):
            p = self.pts[i]
            r = PGon.rectangle(p, self.cell[i][0], self.cell[i][1])
            val = self.val[i]
            r.set_color(color_dict[val[0]][int(val[2])])
            recs.append(r)


        print "putting rectangles, ",
        if draw_recs : svg_out.put(recs)
        print "edges, ",
        if draw_link : svg_out.put(lines)
        print "nodes"
        if draw_nodes : svg_out.put(pts)
        print "drawing tp file ",
        svg_out.draw()
        print "done"

    def to_csv(self,f_name="in",f_path= os.path.expanduser("~")):
        filepath = f_path + os.sep + f_name+".csv"
        print "saving graph to "+filepath
        no_vals = len(self.val[0])
        fout = open(filepath,"w")
        fout.write(str(self.size.a)+','+str(self.size.b)+','+str(no_vals)+'\n')
        fout.write('cell,base,,,values'+no_vals*','+'size,,,link\n')

        for i in range(len(self.pts)):
            vals = [i]
            vals.extend(self.pts[i])
            vals.extend(self.val[i])
            vals.extend(self.cell[i])
            vals.extend(self.link[i])
            out_string = ",".join([str(v) for v in vals])
            fout.write(out_string+'\n')
        fout.close()

    def from_csv(self,f_name="out",f_path=os.path.expanduser("~")):
        filepath = f_path + os.sep + f_name+".csv"
        print "reading graph from "+filepath
        pts = []
        val = []
        cell = []
        link = []
        with open(filepath, 'rb') as f:
            reader = csv.reader(f)
            for n, row in enumerate(reader):
#                print n,row
                if n == 0:
                    no_vals = int(row[2])
                    u = no_vals+4
                    self.size = Interval(int(float(row[0])),int(float(row[1])))
                if n > 1:
                    pts.append([float(row[1]),float(row[2]),float(row[3])])

                    v = []
                    for i in range(4,u): 
                        if row[i] == "" : row[i] = "0"
                        v.append(int(float(row[i])))
                    val.append(v)
                    c = []
                    for i in range(u,u+3): c.append(float(row[i]))
                    cell.append(c)

                    j = u+3
                    flag = True
                    lin = []
                    while j < len(row) and flag:
                        if row[j] <> "":
                            lin.append(int(row[j]))
                            j+=1
                        else:
                            break
                    link.append(lin)

        self.pts = pts
        self.val = val
        self.cell = cell
        self.link = link
        self._res = len(val)


    def to_svg(self,f_name="svg_out", f_path= os.path.expanduser("~"), cdim=Interval(500,500), color_dict = {0:Color(0.0),1:Color(1.0)},state_dict = {0:'none',1:'something'}, parcels = True, cells = True):
        # quick and dirty svg writer
#        ht = cdim.b
        filepath = f_path + os.sep + f_name+".svg"


        max_x = 0
        max_y = 0
        for i in range(len(self.pts)):
            temp_x = self.pts[i][0] + self.cell[i][0]/2.0
            if temp_x > max_x : max_x = temp_x
            temp_y = self.pts[i][1] + self.cell[i][1]/2.0
            if temp_y > max_y : max_y = temp_y


        c = min(cdim.a/max_x,cdim.b/max_y)
        ht = cdim.b

#        c = min(cdim.a/self.size.a,cdim.b/self.size.b)
        print "drawing svg to "+filepath

        buffer = cStringIO.StringIO()
        svg_size = ""
        svg_size = 'width="'+str(cdim.a)+'" height="'+str(cdim.b)+'"'

        buffer.write('<svg '+svg_size+' xmlns="http://www.w3.org/2000/svg" version="1.1">\n')

        type = 'polygon'

        e_list = []
        layer_list = len(state_dict) * [-1]

        for k in range(len(self.val)):
            dx = c * self.cell[k][0]/2
            dy = c * self.cell[k][1]/2
            px = c * self.pts[k][0]
            py = ht - c * self.pts[k][1]
            pts = [[px-dx,py-dy],[px+dx,py-dy],[px+dx,py+dy],[px-dx,py+dy]]
            val = self.val[k]

            # check neighborhood
            if parcels :
                for n in self.link[k]:
                    # if this edge is a border
                    if self.val[n][3] != self.val[k][3]:
                        d = self.direction(n,k)
                        if d == 0:
                            p1 = [self.pts[n][0]+self.cell[n][0]/2.0, min(self.pts[n][1]+self.cell[n][1]/2.0,self.pts[k][1]+self.cell[k][1]/2.0)]
                            p2 = [self.pts[n][0]+self.cell[n][0]/2.0, max(self.pts[n][1]-self.cell[n][1]/2.0,self.pts[k][1]-self.cell[k][1]/2.0)]
                        if d == 1:
                            p1 = [max(self.pts[n][0]-self.cell[n][0]/2.0,self.pts[k][0]-self.cell[k][0]/2.0), self.pts[n][1]-self.cell[n][1]/2.0]
                            p2 = [min(self.pts[n][0]+self.cell[n][0]/2.0,self.pts[k][0]+self.cell[k][0]/2.0), self.pts[n][1]-self.cell[n][1]/2.0]
                        if d == 2:
                            p1 = [self.pts[n][0]-self.cell[n][0]/2.0, min(self.pts[n][1]+self.cell[n][1]/2.0,self.pts[k][1]+self.cell[k][1]/2.0)]
                            p2 = [self.pts[n][0]-self.cell[n][0]/2.0, max(self.pts[n][1]-self.cell[n][1]/2.0,self.pts[k][1]-self.cell[k][1]/2.0)]
                        if d == 3:
                            p1 = [max(self.pts[n][0]-self.cell[n][0]/2.0,self.pts[k][0]-self.cell[k][0]/2.0), self.pts[n][1]+self.cell[n][1]/2.0]
                            p2 = [min(self.pts[n][0]+self.cell[n][0]/2.0,self.pts[k][0]+self.cell[k][0]/2.0), self.pts[n][1]+self.cell[n][1]/2.0]

                        e_list.append([[p1[0]*c, ht-c*p1[1]],[p2[0]*c, ht-c*p2[1]]])

            # put cells
            if cells :
                if layer_list[val[0]] == -1 :
                    layer_list[val[0]] = [k]
                else:
                    layer_list[val[0]].append(k)

# write cells        
        for n in range(len(layer_list)):          
            buffer.write('<g id="'+state_dict[n]+'">\n')
            if layer_list[n] != -1:
                for k in layer_list[n]:
                    dx = c * self.cell[k][0]/2
                    dy = c * self.cell[k][1]/2
                    px = c * self.pts[k][0]
                    py = ht - c * self.pts[k][1]
                    pts = [[px-dx,py-dy],[px+dx,py-dy],[px+dx,py+dy],[px-dx,py+dy]]
                    val = self.val[k]

                    col = color_dict[val[0]][int(val[2])]

                    if val[0] == 3:
                        sw = str(1)
                        st = 'rgb(255,255,255)'
                    else:
                        sw = '0'
                        st = 'none'
                    style = 'fill:rgb('+str(int(255*col.r))+','+str(int(255*col.g))+','+str(int(255*col.b))+');stroke-width:'+sw+';stroke:'+st
                    point_string = " ".join([str(v[0])+","+str(v[1]) for v in pts])
                    atts = 'points="'+point_string+'"'
                    buffer.write('<polygon '+atts+' style="'+style+'"/>\n')
            buffer.write('</g>\n')

        # write parcel edges
        buffer.write('<g id="parcels">\n')
        style = 'fill:none;stroke-width:1;stroke-dasharray:6,2,2,2;stroke:rgb(0,0,0)'
        for e in e_list:           
            point_string = 'x1="'+str(e[0][0])+'" y1="'+str(e[0][1])+'" x2="'+str(e[1][0])+'" y2="'+str(e[1][1])+'" '
            buffer.write('<line '+point_string+' style="'+style+'"/>\n')
        buffer.write('</g>\n')

        buffer.write('</svg>')

        # write buffer to file
        fo = open(filepath, "wb")
        fo.write( buffer.getvalue() )
        fo.close()
        buffer.close()



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
        return (dx <= lx + epsilon) and (dy <= ly + epsilon) and (dz <= lz+ epsilon)
    elif NEIGHBORHOOD_TYPE == VON_NEUMANN:
        if abs(dx - lx) < epsilon:
            return dy < ly - epsilon
        if abs(dy - ly) < epsilon:
            return dx < lx - epsilon
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

def a_count(v,i,a):                     # searches list a, finds value v in sublist # i
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
        self.log = ['start']

    def set_color_dict(self,_color_dict):
        self.color_dict = _color_dict

    def set_dict(self,_color_dict,_state_dict):
        self.color_dict = _color_dict
        self.state_dict = _state_dict

    def set_rule(self,_string="default.txt"):
        self.rule_text = _string

    def set_vis(self,_string="color_default.txt"):
        fin = open(_string)
        self.vis_text = fin.read()
        fin.close()
        print "read"

    def set_params(self,_param):
        self.param = _param

    def rule(self,n=0):
        execfile(self.rule_text)

    def generate(self,gen=1):
        m = self.hist[0].size.a
        n = self.hist[0].size.b
        g = 1
        init_props = copy.copy(self.hist[0].val)
        prob = self.param[0]/100.0
        parcel_mode = (self.param[18] == 1)
        min_size = self.param[19]

#        while g < gen:
#        execfile(self.rule_text)
        # updated 07.02.2013
        # new architecture 07.03.2013 revised - 07.19.2013
        # 07.09.2013 charleston extensions : built or not built at access?
        # 07.16.2013 made to work with variable values

        no_states = len(self.state_dict)

        while g < gen:
            if g == 27:
                print
            # add new generation
            self.add_gen()
            if parcel_mode:
                log_string = 'parcelizing'
                print "p",
                p_list = []
                # get code from ants_130811

            else:
                log_string = 'testing'
                print ".",

                # get target parcel
                p_list = self.hist[g-1].parcel_list()
                p_target = random.randint(0,len(p_list)-1)

                # overall loop
                keep_looking = True
                if keep_looking:
                    c_list = p_list[p_target]

                    # find c_target
                    c_target = -1
                    for i in c_list:
                        if self.hist[g-1].val[i][0] == 0:
                            neighbors = self.hist[g-1].neighbors(i, no_states, include_other_parcels = False)
                
                            street_count = len(neighbors[1])
                            access_count = len(neighbors[2])
                            built_count = len(neighbors[3])
                            os_count = len(neighbors[4])

                            if (access_count + street_count) > 0 : 
                                p = random.choice(neighbors[1]+neighbors[2])
                                c_target = [i,p[0],p[1]]
                                # [0] = index of target cell, [1] = index of enabling cell,[2] = direction, cell

                    # check if c_target is found
                    if c_target != -1:

                        # place a built unit
                        print "working on ", c_target," in parcel ",p_target,": placing ",

                        # initialize
                        depth = self.hist[g-1].cell[c_target[0]][c_target[2]%2]
                        placed = False

                        # loop
                        while not placed:
                            if (random.randint(0,100) < self.param[0]) :
                                # perform operation B
                                print " built"
                                if depth >= 4*min_size: 
                                    new_size = 2*min_size
                                    new_cell = self.hist[g].divide(c_target[0], c_target[2], new_size)
                                else:
                                    new_cell = c_target[0]
                                self.hist[g].val[new_cell][0:3] = [3,-1,0]
                                placed = True
                            else:
                                # perform operation A                            
                                print " access,",

                        keep_looking = False

                    else:
                        # no access enabled sites
                        print "no access-enabled sites :",

                        if (random.randint(0,100) < self.param[14]) :
                            # parcel merge
                            print " merging parcels"

                        else:
                            # building up
                            print " add vertical addition"
                            keep_looking = False

#            print log_string                    
            self.log.append(log_string)            
            g += 1


        if self.param[2] == 1:
            self.hist[0] = self.hist[1]   
#        self.hist[0].val = init_props


    def write_images(self, fname="out", base_path=os.path.expanduser("~") + os.sep):
        size = self.hist[0].size
        for i,g in enumerate(self.hist):
            if i % self.param(6) == 0:
                img = g.to_image(size,self.color_dict)
                img.save(fname+str(i), base_path, True)

    def write_svgs(self,fname="out", base_path=os.path.expanduser("~") + os.sep, size = Interval(500,500)):

        for i,g in enumerate(self.hist):
            if (i%self.param[6] == 0) or (i+1== len(self.hist)):
                g.to_svg(fname+'%03d'%i, base_path,size,self.color_dict,self.state_dict, parcels = (self.param[16] == 1), cells = (self.param[15] == 1))
        for k in range(100//self.param[5]):
            g.to_svg(fname+'%03d'%i+str(k), base_path,size,self.color_dict,self.state_dict, parcels = (self.param[16] == 1), cells = (self.param[15] == 1))
        if len(self.state_dict) != 0:
            print "writing to ",fname+"_m.csv"
            np = len(self.state_dict)
            fout = open(base_path + os.sep + fname+"_m.csv",'w')
            out_string = ",".join(['generation']+self.state_dict.values()+['action'])
            fout.write(out_string+'\n')
            for i,g in enumerate(self.hist):
                v = np * [0]
                for j in range(len(g.val)):
                    v[g.val[j][0]%np]+=(g.cell[j][0] * g.cell[j][1]) * int(g.val[j][2]+1)
                out_string = ",".join([str(n) for n in [i]+v] + [self.log[i]])
                fout.write(out_string+'\n')
            fout.close()
        print "writing to ",fname+".bat"
        fout = open(base_path + os.sep + fname+".bat",'w')
        fout.write('cd '+base_path + '\n')
        fout.write('convert -delay '+str(self.param[5])+' -loop 0 *.svg '+fname+'.gif\n')
        fout.write(fname+'.gif\n')
 #       fout.write('pause\n')
        fout.close()
 #       return base_path + os.sep + fname+".bat"


    def test(self,n=0):
        print self.hist[n]

    def add_gen(self):
        self.hist.append(Graph(copy.copy(self.hist[-1].link),copy.copy(self.hist[-1].pts), copy.copy(self.hist[-1].cell), copy.copy(self.hist[-1].val )))
        self.hist[-1].size = self.hist[-2].size


