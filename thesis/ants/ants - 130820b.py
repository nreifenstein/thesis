
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

    def portion(self,i,j):
        dir = self.direction(i,j)
        dx = self.pts[i][0] - self.pts[j][0]
        dy = self.pts[i][1] - self.pts[j][1]
        if (dir%2) == 0:
            if dy <= 0 : return 1
            else: return 3
        else:
            if dx < 0 : return 2
            else: return 0
        return -1
          
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

    def parcel_fp(self, cells,no_vals = 2):
        # returns a list of areas for each val[0] in cells
        result = [0] * no_vals
        for c in cells:
            a = self.cell[c][0]*self.cell[c][1]
            result[self.val[c][0]] += a
        return result

    def parcel_flr(self, cells, no_vals = 2):
        # returns a list of areas for each val[0] in cells
        result = [0] * no_vals
        for c in cells:
            a = self.cell[c][0]*self.cell[c][1]*(1+self.val[c][2])
            result[self.val[c][0]] += a
        return result

    def best_choice(self,cells, no_vals = 2, max_coverage = .5, min_size = 12):
        # [0] : largest potential access site - could be any size
        # [1] : largest potential building site > 24 x 24
        # [2] : largest built site - for vertical addition
        result = [[],[],[]]
        best_area = [0, 575,0]
        fp = self.parcel_fp(cells, no_vals)
        pa = a_sum(fp)
        fpa = a_sum(fp,[0,0,0,1,0,1])
        coverage = fpa / pa
        for i in cells:
            v = self.val[i]
            area = self.cell[i][0] * self.cell[i][1]
            min_dim = min(self.cell[i][0],self.cell[i][1])
            if (v[0] == 0) and area > best_area[0] and coverage < max_coverage:
                neighbors = self.neighbors(i, no_vals, include_other_parcels = False)
                
                street_count = len(neighbors[1])
                access_count = len(neighbors[2])

                if (access_count + street_count) > 0 : 
                    p = random.choice(neighbors[1]+neighbors[2])
                    result[0] = [i,p[0],p[1]]
                    best_area[0] = area
                    if (area > best_area[1]) and (min_dim > min_size) : result[1] = result[0]
            elif (v[0] == 3):
                result[2] = [i,-1,-1]
                best_area[2] = area
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

    def to_svg(self,f_name="svg_out", f_path= os.path.expanduser("~"), cdim=Interval(500,500), color_dict = {0:Color(0.0),1:Color(1.0)},state_dict = {0:'none',1:'something'}, parcels = True, cells = True, floors = False):
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

                    #col = color_dict[val[0]][int(val[2])]
                    col = color_dict[val[0]][0]

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
                    if ((val[0] == 3) or (val[0]==5)) and floors:
                        string = '<text transform="matrix(1 0 0 1 '+str(pts[3][0])+' '+str(pts[3][1])+')" fill="#FFFFFF"'
                        string = string + ' font-size="16">'+str(val[2]+1)+'</text>\n'
                        #string = string + ' font-family="'+"'Futura'"+'" font-size="24">'+str(val[2]+1)+'</text>\n'
                        buffer.write(string)

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
def a_add_like(a1,a2 = [0]):
    result = []
    if len(a2) < len(a1):
        t = len(a1) * a2
        a2 = t[0:len(a1)]
    for i in range(min(len(a1),len(a2))):
        result.append(a1[i]+a2[i])
    return result

def a_sub_like(a1,a2):
    result = []
    for i in range(min(len(a1),len(a2))):
        result.append(a1[i]-a2[i])
    return result

def a_mult_like(a1,a2 = [1]):
    result = []
    if len(a2) < len(a1):
        t = len(a1) * a2
        a2 = t[0:len(a1)]
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

def a_sum(a, mask = None):
    if mask == None:
        mask = len(a) *[1]
    ret = 0
    for i in range(len(a)): ret+=a[i]*mask[i]
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
        # updated 08.12.2013

        no_states = len(self.state_dict)

        while g < gen:
            # add new generation
            self.add_gen()
            #if g == 28:
                #print
            if parcel_mode:
                log_string = 'parcelizing'
                print "p",
                p_list = []
                # get code from ants_130811

            else:
                log_string = 'testing'
                print g,".",

                # get target parcel
                p_list = self.hist[g-1].parcel_list()
                #p_target = random.randint(0,len(p_list)-1)

                # loop through p_list
                a_list = []
                b_list = []
                c_list = []
                for parcel in p_list:
                    # find target cell : c_target
                    r = self.hist[g-1].best_choice(parcel, no_states, self.param[22])
                    if r[0] != []:
                        a_list.append(r[0])
                        if r[1] != [] : b_list.append(r[1])
                    elif r[2] != []:
                        c_list.append(r[2])

                # check to see if a_list is empty
                if len(a_list) > 0:

                        build = (random.randint(0,100) < self.param[0])
                        if build and b_list == [] : build = False

                        if build : 
                            c_target = random.choice(b_list)
                        else:
                            c_target = random.choice(a_list)

                        p_target = self.hist[g-1].val[c_target[0]][3]

                        # place a built unit
                        print "working on ", c_target," in parcel ",p_target,": placing ",

                        # initialize
                        depth = int(self.hist[g-1].cell[c_target[0]][c_target[2]%2])
                        width = int(self.hist[g-1].cell[c_target[0]][(1+c_target[2])%2])
                        placed = False


                        # determine justification
                        w_dir = self.hist[g-1].portion(c_target[0],c_target[1])

                        # loop
                        while not placed:
                            # make sure cell is proper size
#                            proper_size = ((depth >= 2*min_size)) and ((width >= 2*min_size))
                            if build :
                                # perform operation B
                                print " built"
                                if depth >= 4*min_size: 
                                    new_size = 2*min_size
                                    new_cell = self.hist[g].divide(c_target[0], c_target[2], new_size)
                                else:
                                    new_cell = c_target[0]
                                if width >= 3*min_size:
                                    new_width = new_cell
                                    new_size = 2*min_size
                                    new_cell = self.hist[g].divide(new_width, w_dir, new_size)
                                if width > min_size:
                                    self.hist[g].val[new_cell][0:3] = [3,g,0]
                                else: self.hist[g].val[new_cell][0:3] = [4,g,0]
                                placed = True
                            else:
                                # perform operation A                            
                                print " access,",
                                if (depth == min_size) or (width == min_size):
                                    # if too small, make it access
                                    self.hist[g].val[c_target[0]][0:3] = [2,g,0]
                                    placed = True
                                else:
                                    # carve out access piece
                                    no_zones = int(depth/min_size)
                                    if no_zones < 3:
                                        new_size = depth
                                    else:
                                        zone_list = range(1,no_zones+1)
                                        if no_zones-1 in zone_list: zone_list.remove(no_zones-1)
                                        zones = random.choice(zone_list)
                                        new_size = zones*min_size
                                    if new_size < depth:
                                        new_cell = self.hist[g].divide(c_target[0], c_target[2], new_size)
                                    else:
                                        new_cell = c_target[0]
                                    new_val = 0
                                    if width > min_size:
                                        new_access = self.hist[g].divide(new_cell, w_dir, min_size)
                                    else: new_access = new_cell
                                    self.hist[g].val[new_access][0:3] = [2,g,0]
                                    self.hist[g].val[new_cell][0:3] = [new_val,g,0]
                                    #placed = True
                                    # select new candidate cells
                                    p_list = self.hist[g].parcel_list()
                                    parcel = p_list[p_target]
                                    r = self.hist[g].best_choice(parcel, no_states, self.param[22])
                                    if r[0] == []:
                                        # no more minimal sites are left
                                        placed = True
                                    else:
                                        #build = (random.randint(0,100) > self.param[20])
                                        build = True
                                        if r[1] == [] : placed = True
                                        if build : 
                                            c_target = r[1]
                                        else:
                                            c_target = r[0]
                                        if c_target != []:
                                            depth = int(self.hist[g].cell[c_target[0]][c_target[2]%2])
                                            width = int(self.hist[g].cell[c_target[0]][(1+c_target[2])%2])
                                            w_dir = self.hist[g].portion(c_target[0],c_target[1])

                else:
                        # no access enabled sites
                        print "no access-enabled sites :",
                        if self.param[21] == 0 :
                            print " terminating"
                            self.log.append('terminating')
                            break
                        d_list = random.choice(c_list)
                        p_target = self.hist[g-1].val[d_list[0]][3]

                        if (random.randint(0,100) < self.param[14]) :
                            # parcel merge
                            print " merging parcels"

                            # get parcel to merge with
                            merge_list = []
                            for i in d_list:
                                for j in self.hist[g].link[i]:
                                    if (self.hist[g].val[j][3] > -1) and (self.hist[g].val[j][3] != p_target):
                                        merge_list.append(self.hist[g].val[j][3])

                            # select random neighboring parcel
                            if len(merge_list) > 0:
                                merge_target = random.choice(merge_list)

                                for k in p_list[merge_target]:
                                    self.hist[g].val[k][3] = p_target
                                    p_list[p_target].append(k)
                                p_list[merge_target] = []

                        else:
                            # building up
                            print " add vertical addition"

                            # find c_target that is built
                            for i in c_list:
                                if (self.hist[g-1].val[i][0] == 3) or (self.hist[g-1].val[i][0] == 5):
                                    if self.hist[g-1].val[i][2] >= self.param[23]: self.hist[g].val[i][0] = 5
                                    self.hist[g].val[i][2] = 1 + self.hist[g-1].val[i][2]
                                    break
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

    def write_parcels(self, fname="out", base_path=os.path.expanduser("~") + os.sep, gen = 0):
        print "writing to ",fname+'%03d'%gen+"_p.csv"
        np = len(self.state_dict)
        fout = open(base_path + os.sep + fname+'%03d'%gen+"_p.csv",'w')
        out_string = "parcel, total area, footprint3,footprint5,coverage,total built,far"
        fout.write(out_string+'\n')
        parcels = self.hist[gen].parcel_list()
        total_area = 0
        total_built = 0
        total_footprint3 = 0
        total_footprint5 = 0
        for i,parcel in enumerate(parcels):
            fp = self.hist[gen].parcel_fp(parcel,np)
            ba = self.hist[gen].parcel_flr(parcel,np)
            area = a_sum(fp)
            built = a_sum(ba,[0,0,0,1,0,1])
            if area > 0:
                out_string = str(i)+','+str(area)+','+str(fp[3])+','+str(fp[5])+','+str(round((fp[3]+fp[5])/area,2))+','+str(built)+','+str(round(built/area,2))
                fout.write(out_string+'\n')
            total_area += area
            total_built += built
            total_footprint3 += fp[3]
            total_footprint5 += fp[5]
        total_footprint = total_footprint3 + total_footprint5
        out_string = ','+str(total_area)+','+str(total_footprint3)+','+str(total_footprint5)+','+str(round(total_footprint/total_area,2))+','+str(total_built)+','+str(round(total_built/total_area,2))
        fout.write(out_string+'\n')

        fout.close()


    def write_svgs(self,fname="out", base_path=os.path.expanduser("~") + os.sep, size = Interval(500,500)):
        parcels = (self.param[16] == 1)
        cells = (self.param[15] == 1)
        floors = (self.param[24] == 1)
        for i,g in enumerate(self.hist):
            if (i%self.param[6] == 0) or (i+1== len(self.hist)):
                g.to_svg(fname+'%03d'%i, base_path,size,self.color_dict,self.state_dict, parcels, cells, floors)
        for k in range(100//self.param[5]):
            g.to_svg(fname+'%03d'%i+str(k), base_path,size,self.color_dict,self.state_dict, parcels, cells, floors)
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


