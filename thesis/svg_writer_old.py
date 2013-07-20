  ## code for svg writer 07.18.2013 
  
   def to_svg(self,f_name="svg_out", f_path= os.path.expanduser("~"), cdim=Interval(500,500), color_dict = {0:Color(0.0),1:Color(1.0)},v_rule = 'col=color_dict[val[0]]'):
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

        e_list = []

        parcels = True

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


#            exec(v_rule)
            col = color_dict[val[0]][int(val[2])]

#            col = color_dict[self.val[k][0]]
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