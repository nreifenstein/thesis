
import decodes as dc
from decodes.core import *
from decodes.core import dc_color, dc_base, dc_vec, dc_point, dc_cs, dc_line, dc_mesh, dc_pgon, dc_xform
#from decodes.io import *
#from decodes.io import outie
import copy
import csv
print "ants.py loaded"
import os,cStringIO
import PIL

## define graph class
class Graph():
    def __init__(self,_list=[],_pts=[],_cells=[],_vals=[]):
        self.link = copy.copy(_list)
        self.pts = copy.copy(_pts)
        self.cell = copy.copy(_cells)
        self.val = copy.copy(_vals)

    @property
    def _res(self):  return len(self.val)

    def init_rectgrid(self,size=Interval(20,20),include_corners=False,wrap=False,cellsize=1):
        self.size = size
        m = size.a
        n = size.b
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

    def init_block(self, model_size=Interval(40,40), block_size=Interval(20,20), init_val = 1, prob = 50):
        mod_x = block_size.a + 1
        mod_y = block_size.b + 1
        c_x = (model_size.a + block_size.a) / 2
        c_y = (model_size.b + block_size.b) / 2
        v = (model_size.a * model_size.b) * [0]

        for i in range(model_size.a):
            for j in range(model_size.b):
                k = i+j*model_size.a
                if ((i-c_x) % mod_x == 0) or ((j-c_y) % mod_y == 0):
                    v[k] = [1,0,0]
                else:
                    v[k] = [0,0,0]

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
                    
    def direction(self,i,j):
        WEST = 0
        NORTH = 1
        EAST = 2
        SOUTH = 3
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
        lx = c1[0]/2 + c2[0]/2     ## get sum of widths, height, lengths
        ly = c1[1]/2 + c2[1]/2 
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
            
    def divide(self,i,d,q):     ## i = cell index, d = direction, q = fraction
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

    def to_dc_svg(self,f_name="svg_out", color_dict = {0:Color(0.0),1:Color(1.0)}, cdim=Interval(500,500), draw_recs=True,draw_nodes=False,draw_link=False):
        # this uses Decodes. OK for nodes and link, slow for cells.
        svg_out = dc.makeOut(dc.Outies.SVG, f_name, canvas_dimensions=cdim, flip_y = False)
        lines = []
        pts = []
        pts2 = []   # total list, includes empty nodes - use for line list
        recs = []
        coord = self.pts
        
        for i in range(len(coord)): 
#            print "checking node ",i
            p = Point(coord[i][0],coord[i][1],coord[i][2])
            pts2.append(p)
            if len(self.link[i]) > 0:
                if draw_nodes:
                    p.set_color(Color(0.0))
                    p.set_weight(self.cell[i][0]*.1)
                pts.append(p)
                if draw_recs:
                    r = PGon.rectangle(p, self.cell[i][0], self.cell[i][1])
                    r.set_color(color_dict[self.val[i][0]])
                    recs.append(r)

        if draw_link :
            for i in list(range(len(coord))):
                n = self.link[i]
                for j in n:
                    if i < j:
                        line = Segment(pts2[i],pts2[j])
                        line.set_color(Color(0.0))
                        lines.append(line)

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
                    self.size = Interval(int(row[0]),int(row[1]))
                if n > 1:
                    pts.append([float(row[1]),float(row[2]),float(row[3])])

                    v = []
                    for i in range(4,u): 
                        if row[i] == "" : row[i] = "0"
                        v.append(int(row[i]))
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


    def to_svg(self,f_name="svg_out", f_path= os.path.expanduser("~"), cdim=Interval(500,500), color_dict = {0:Color(0.0),1:Color(1.0)}):
        # quick and dirty svg writer
        ht = cdim.b
        filepath = f_path + os.sep + f_name+".svg"

        c = min(cdim.a/self.size.a,cdim.b/self.size.b)
        print "drawing svg to "+filepath

        buffer = cStringIO.StringIO()
        svg_size = ""
        svg_size = 'width="'+str(cdim.a)+'" height="'+str(cdim.b)+'"'

        buffer.write('<svg '+svg_size+' xmlns="http://www.w3.org/2000/svg" version="1.1">\n')

        type = 'polygon'

        for k in range(len(self.val)):
            dx = c * self.cell[k][0]/2
            dy = c * self.cell[k][1]/2
            px = c * self.pts[k][0]
            py = ht - c * self.pts[k][1]
            pts = [[px-dx,py-dy],[px+dx,py-dy],[px+dx,py+dy],[px-dx,py+dy]]
            col = color_dict[self.val[k][0]]
            style = 'fill:rgb('+str(int(255*col.r))+','+str(int(255*col.g))+','+str(int(255*col.b))+');stroke-width:0;stroke:none'
            point_string = " ".join([str(v[0])+","+str(v[1]) for v in pts])
            atts = 'points="'+point_string+'"'
            buffer.write('<polygon '+atts+' style="'+style+'"/>\n')


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
        init_props = copy.copy(self.hist[0].val)
        prob = self.param[0]/100.0

        while g < gen:
#            execfile(self.rule_text)
            # modified 06.26.2013 to create incremental additions
            # updated 07.02.2013
            # new architecture 07.03.2013

            # add new generation
            self.add_gen()
            log_string = 'testing'
            print ".",
            # create list of enablers [access sites]
            a_list = []
            for j in range(len(self.hist[g-1].val)):
                if (self.hist[g-1].val[j][0] == 1 or self.hist[g-1].val[j][0] == 2) and (0 in self.hist[g-1].n_vals(j)):
                    a_list.append(j)

            if len(a_list) == 0 : 
                self.log.append('no more building sites')
                break

            # create iterator list - i_list
            if self.param[3] == 0 :
                i_list = a_list
            else:
                i_list = [random.choice(a_list)]


            # loop through i_list
            for i in i_list:

                # make list of neighbors with val == 0
                n_list = []
                for n in self.hist[g-1].link[i]:
                    if (self.hist[g-1].val[n][0] == 0):
                        n_list.append(n)

                # create iterator list for neighbors
                if self.param[3] == 0 :
                    j_list = n_list
                else:
                    j_list = [random.choice(n_list)]

                # loop through j_list
                for j in j_list:

                    # check for special cases
                    flag = False
                    neighbor_values = self.hist[g].n_vals(j,k=-1)
                    street_count = a_count(1,0,neighbor_values)
                    access_count = a_count(2,0,neighbor_values)

                    #       case 1: site is at intersection [built corner rule]
                    if street_count > 1 : flag = True

                    #       case 2: site is on the street but already has access adjacent [alternation rule]
                    #               Note: this uses the current generation (g not g-1 see above) to test emergent order in this generation
                    if (street_count == 1) and (access_count > 0) : flag = True

                    # main loop
                    if flag or (self.hist[g-1].val[i][2] > self.param[1]) or (random.uniform(0.0,1.0) < prob):
                        # greedy [built] only
                        self.hist[g].val[j] = [3,-1,0]
                    else:
                        # charleston [access]
                        t = []
                        t.append(2)
                        d = self.hist[g].direction(j,i)
                        t.append(d)
                        if d%2 == self.hist[g].val[i][1] :
                            t.append(self.hist[g].val[i][2] + (1.0/self.param[4]))
                        else:
                            t.append(self.hist[g].val[i][2] + 1.0 )
                        self.hist[g].val[j] = t
            self.log.append(log_string)        
            g+=1
        print
            
#        self.hist[0].val = init_props


    def write_images(self, fname="out", base_path=os.path.expanduser("~") + os.sep):
        size = self.hist[0].size
        for i,g in enumerate(self.hist):
            if i % self.param(6) == 0:
                img = g.to_image(size,self.color_dict)
                img.save(fname+str(i), base_path, True)

    def write_svgs(self,fname="out", base_path=os.path.expanduser("~") + os.sep, size = Interval(500,500),state_dict=dict()):
        for i,g in enumerate(self.hist):
            if i%self.param[6] == 0:
                g.to_svg(fname+'%03d'%i, base_path,size,self.color_dict)
        if len(state_dict) != 0:
            print "writing to ",fname+"_m.csv"
            np = len(state_dict)
            fout = open(base_path + os.sep + fname+"_m.csv",'w')
            out_string = ",".join(['generation']+state_dict.values()+['action'])
            fout.write(out_string+'\n')
            for i,g in enumerate(self.hist):
                v = np * [0]
                for j in g.val:
                    v[j[0]%np]+=1
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

    def write_animated_svgs(self, f_name="out", f_path=os.path.expanduser("~") + os.sep):
        filepath = f_path + os.sep + f_name+".svg"
        dur = 1
        size = Interval(500,500)
        for i,g in enumerate(self.hist):
            g.to_svg(f_name+str(i), f_path,size,self.color_dict)

        buffer = cStringIO.StringIO()
        svg_size = ""
        svg_size = 'width="'+str(1000)+'" height="'+str(1000)+'"'

        buffer.write('<svg '+svg_size+' xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1">\n')
        
        for i in range(len(self.hist)):
            buffer.write('<image '+svg_size+' xlink:href="'+f_name+str(i)+'.svg">\n')
            values = len(self.hist) * ["none"]
            values[i] = "inline"
            v_string = ";".join(values)
            line_string ="<animate id='frame_"+str(i)+"' attributeName='display' values='"+v_string+"'"
            line_string = line_string + " dur = '"+str(dur)+"s' fill='freeze' begin='"+str(i*dur)+"s' repeatCount='indefinite' />\n"
            buffer.write(line_string)
            buffer.write('</image>\n')
       

        buffer.write('</svg>')

        # write buffer to file
        fo = open(filepath, "wb")
        fo.write( buffer.getvalue() )
        fo.close()
        buffer.close()


    def test(self,n=0):
        print self.hist[n]

    def add_gen(self):
        self.hist.append(Graph(self.hist[-1].link,self.hist[-1].pts, self.hist[-1].cell, self.hist[-1].val ))
        self.hist[-1].size = self.hist[-2].size

