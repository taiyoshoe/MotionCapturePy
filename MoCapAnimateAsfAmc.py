"""
A Python class for storing and accessing motion capture data from ASF/AMC files

Authors: Ari Herman, Marissa Major, Taiyo Terada

"""
import numpy as np
from numpy import array,dot,zeros, eye, savetxt
import math
from math import sin, cos, pi
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


# MotionCapture class description:
# Data: Deg2Rad, ASF_lines, AMC_lines, bones, axis, length, direction, dof, 
# hierarchy, id, limbs, angleData, num_bones, C, Cinv, v, num_frames, data
# Methods: Rx, Ry, Rz, Rxyz, globalPosition, globalRotation, Animate, makePoints

class MotionCapture(object):
    def __init__(self,ASF_file,AMC_file):
        self.Deg2Rad = pi/180
        self.ASF_lines = open(ASF_file,"r").readlines()
        self.AMC_lines = open(AMC_file,"r").readlines()
        self.bones=["root"]
        self.axis = {"root":(0,0,0)}
        self.length = {"root":0}
        self.direction = {"root":(0,0,0)}
        self.dof = {"root":[1,1,1]}
        self.hierarchy = {"root":[]}
        self.id = {"root":0}
        self.limbs = []
        self.angleData = {"root":[]}
        
        # Read ASF file
        ######################################################################################
        n = len(self.ASF_lines)
        for i in range(n):
            if self.ASF_lines[i].split()[0] == ":bonedata":
                k = i
                break
        for i in range(k,n):   
            line = self.ASF_lines[i].split()
            if line[0] == "id":
                ID = int(line[1])
            elif line[0] == "name":
                bone = line[1]
                self.bones.append(bone)
                self.axis[bone] = None
                self.dof[bone] = [0,0,0]
                self.hierarchy[bone] = []
                self.angleData[bone] = []
                self.id[bone] = ID
            elif line[0] == "direction":
                self.direction[bone] = (float(line[1]),float(line[2]),float(line[3]))
            elif line[0] == "length":
                self.length[bone] = float(line[1])
            elif line[0] == "axis":
                self.axis[bone] = (float(line[1]),float(line[2]),float(line[3]))
            elif line[0] == "dof":
                for rot in line[1:]:
                    if rot=="rx":
                        self.dof[bone][0]=1
                    elif rot=="ry":
                        self.dof[bone][1]=1
                    elif rot=="rz":
                        self.dof[bone][2]=1              
            if line[0] == ":hierarchy":
                j = i
                break
        for i in range(j,n):
            line = self.ASF_lines[i].split()
            for j in range(1,len(line)):
                self.hierarchy[line[j]].append(line[0])
                self.hierarchy[line[j]]+=self.hierarchy[line[0]]
                self.limbs.append((self.id[line[0]],self.id[line[j]]))
        self.num_bones =len(self.bones)
        self.C = {bone:dot(dot(self.Rz(self.axis[bone][2]),self.Ry(self.axis[bone][1])),self.Rx(self.axis[bone][0])) for bone in self.bones}
        self.Cinv = {bone:dot(dot(self.Rx(-self.axis[bone][0]),self.Ry(-self.axis[bone][1])),self.Rz(-self.axis[bone][2])) for bone in self.bones}        
        self.v={bone:self.length[bone]*array(self.direction[bone]).T for bone in self.bones}
        #######################################################################
                
        # Read AMC file
        #######################################################################
        n = len(self.AMC_lines)
        for i in range(n):
            line = self.AMC_lines[i].split()
            if line[0] == "1":
                j = i
                break
        for i in range(j,n):
            line = self.AMC_lines[i].split()
            bone = line[0]
            try:
                idx = int(bone)
            except ValueError:
                if bone == "root":
                    self.angleData["root"].append([float(line[4]),float(line[5]),float(line[6])])
                else:
                    angles = [0.0,0.0,0.0]
                    angles2 = [float(num) for num in line[1:]]
                    for k in range(3):
                        if self.dof[bone][k] == 1:
                            angles[k] = angles2.pop(0)
                    self.angleData[line[0]].append(angles)
        self.num_frames = idx - 1
        #######################################################################
    
        self.data = zeros((self.num_frames,len(self.bones),3))
        for i in range(self.num_frames):
            framei=self.globalPosition(i)
            for bone in self.bones:
                self.data[i,self.id[bone],:] = 0.05*framei[self.id[bone]]
        
    def Rx(self,r):
        r *= self.Deg2Rad
        return array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
        
    def Ry(self,r):
        r *= self.Deg2Rad
        return array([[cos(r),0,sin(r)],[0,1,0],[-sin(r),0,cos(r)]])
        
    def Rz(self,r):
        r *= self.Deg2Rad
        return array([[cos(r),-sin(r),0],[sin(r),cos(r),0],[0,0,1]])

    def Rxyz(self,rv):
        return dot(dot(self.Rz(rv[2]),self.Ry(rv[1])),self.Rx(rv[0]))

    def globalPosition(self,frame):
        vout=[]
        RotateFrame=self.globalRotation(frame)
        for bone in self.bones:
            M=dot(RotateFrame[self.id[bone]],self.v[bone])
            if len(self.hierarchy[bone])>0:
                parent=self.hierarchy[bone][0]
                M=M+vout[self.id[parent]]
            vout.append(M)
        return vout
        
    def globalRotation(self,frame):
        output=[]
        for bone in self.bones:
            if self.dof[bone] == [0,0,0]:
                rotate = eye(3,3)
            else:
                rotate=dot(dot(self.C[bone],self.Rxyz(self.angleData[bone][frame])),self.Cinv[bone])
            if len(self.hierarchy[bone])>0:
                parent=self.hierarchy[bone][0]
                rotate=dot(rotate,output[self.id[parent]])
            output.append(rotate)
        return output  
    
    def makePoints(self, bone_subset=None):
        if bone_subset == None:
            bone_subset = self.bones
        idx = [self.id[bone] for bone in bone_subset]
        return self.data[:,idx,:].reshape(self.num_frames,3*len(idx))
        
    def Animate(self,X):
        n = len(X)
        def anim(i):
            for j in range(n):
                line[j].set_data(self.data[i,X[j],0:2].T)
                line[j].set_3d_properties(self.data[i,X[j],2].T)
            time_text.set_text('time = %.1d' % i)
            return line
            return time_text
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        line = [ax.plot([],[],[])[0] for j in range(n)]
        time_text = ax.text(0.0,0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
        line_ani = animation.FuncAnimation(fig, anim, self.num_frames, interval=20, blit=False)
        plt.show()
   
   
# Create MotionCapture object
###############################################################################

#motion = MotionCapture("02.asf","02_01.amc")  # Ex 1
motion = MotionCapture("02.asf","02_01.amc")  # Ex 2
#points= motion.makePoints()
motion.Animate(motion.limbs)
  
# Animate     
###############################################################################
       
#motion.Animate(motion.limbs)



#######################################################
#savetxt("datadance.txt",points)
####################################################3

# Plot path of a single bone
###############################################################################
#
#bone = "lowerback"
#
#points = motion.makePoints([bone])
#points = array(points)
#
#fig = plt.figure()
#ax = p3.Axes3D(fig)
#ax.plot(points[:,0],points[:,1],points[:,2])
#plt.show()


