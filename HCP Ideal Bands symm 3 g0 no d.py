# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:22:35 2017

@author: tigan_5ytncvu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:21:55 2017

@author: tigan_5ytncvu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:59:04 2017

@author: tigan_5ytncvu
"""
##############################################################################
############    DIAGONALIZATION CODE FOR HAMILTONIAN MATRIX    ###############
##############################################################################


import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
import scipy as sp


#####    FCC   #####


class Hconstruct:
    
    def __init__(self, k=np.array([0,0,0]), a = 1., d = np.sqrt(2)*((16*np.pi/3.)**(1./3)), ca_ratio = np.sqrt(8/3.),#1.61, #a is in Angstrom
                
                 d1=np.array([1,0,0]), d2=np.array([1,np.sqrt(3),0]), 
                 d3=np.array([0,0,1]), d4 = np.array([1,-np.sqrt(3),0]),
                 d5=np.array([-1,-np.sqrt(3),0]), d6=np.array([-1,np.sqrt(3),0]), 
                 d7=np.array([0,0,-1]),d8 = np.array([-1,0,0]),
                Es=1, Ep=1, 
                 Ed=1, sssig=-2, spsig=2*np.sqrt(3), ppsig=12., pppi=-6., sdsig=-3.16, pdsig=-6*np.sqrt(15), #sssig to pppi from Harrison and general, d from Titanium solid state table
                 pdpi=6*np.sqrt(5), ddsigma= -60., ddpi= 40., dddelta= -10., rd = 1.08 ):

        self.a = a
        self.d = d
        self.ca_ratio = ca_ratio
        self.c = ca_ratio*self.a
        
        
        self.d1 = (self.a)*d1
        self.d2 = (self.a/2.)*d2
        #self.d3 = (self.c)*d3
        self.d3 = (self.a/2.)*d4
        
        self.d4 = (self.a/2.)*d5
        self.d5 = (self.a/2.)*d6
        #self.d7 = (self.c)*d7
        self.d6 = (self.a)*d8
        self.b1 = np.array([1./2.*self.a, ((1/6.)*np.sqrt(3))*self.a, 0.5*self.c])
        #self.b1 = 2./3*self.d1 + 1./3*self.d2 + 1./2*self.d3
        self.b2 = self.b1 - self.d1
        self.b3 = self.b1 - self.d2
        
        self.b4 = np.array([1./2.*self.a, ((1/6.)*np.sqrt(3))*self.a, -0.5*self.c])
        self.b5 = self.b4 - self.d1
        self.b6 = self.b4 - self.d2
        
        
        
        
        self.bmag = np.sqrt((1/3.)*self.a**2 + 0.25*(self.c**2))
                            
                            
        self.darr = np.asarray([self.d1,self.d2,self.d3,self.d4,self.d5,self.d6,
                           self.b1/self.bmag, self.b2/self.bmag,self.b3/self.bmag,self.b4/self.bmag,self.b5/self.bmag,self.b6/self.bmag])
        self.l = (self.darr[:, :1].astype(np.complex_))
        self.m = (self.darr[:, 1:2].astype(np.complex_))
        self.n = (self.darr[:, 2:3].astype(np.complex_))
        
        self.ll = self.l*self.l
        self.mm = self.m*self.m
        self.nn = self.n*self.n
        self.lm = (self.darr[:, 1:2]*self.darr[:, :1]).astype(np.complex_)
        self.ln = (self.darr[:, 2:3]*self.darr[:, :1]).astype(np.complex_)
        self.mn = (self.darr[:, 2:3]*self.darr[:, 1:2]).astype(np.complex_)
        #self.d4 = (self.a/np.sqrt(2))*d4
        self.Es = Es
        self.Ep = Ep
        self.Ed = Ed
        self.sssig = sssig/(self.d)
        self.spsig = spsig/(self.d**2)
        self.ppsig = ppsig/(self.d**3)
        self.pppi = pppi/(self.d**3)
        self.sdsig = sdsig/(self.d)
        self.pdsig = pdsig/(self.d**4)
        self.pdpi = pdpi/(self.d**4)
        self.ddsig = ddsigma/(self.d**5)
        self.ddpi = ddpi/(self.d**5)
        self.dddelta = dddelta/(self.d**5)
        self.k = k
        self.rd = rd
        self.energiess1 = []
        self.energiess2 = []
        
        self.px_energies = []
        self.py_energies = []
        self.pz_energies = []
        
        self.px_energies2 = []
        self.py_energies2= []
        self.pz_energies2 = []
        
        self.xy_energies = []
        self.yz_energies = []
        self.zx_energies = []
        self.xxyy_energies = []
        self.zr_energies = []
        
        self.xy_energies2 = []
        self.yz_energies2 = []
        self.zx_energies2 = []
        self.xxyy_energies2 = []
        self.zr_energies2 = []
        
        self.kvals = []
        
#
    
    
        
    def phasefactors(self, kv):
        
        
        #dot products of k vectors with distance vectors of unit cell 
        kd1 = np.dot(kv, self.d1) 
        kd2 = np.dot(kv, self.d2) 
        kd3 = np.dot(kv, self.d3)
        kd4 = np.dot(kv, self.d4) 
        kd5 = np.dot(kv, self.d5) 
        kd6 = np.dot(kv, self.d6) 
        
        kd7 = np.dot(kv, self.b1) 
        kd8 = np.dot(kv, self.b2) 
        kd9 = np.dot(kv, self.b3) 
        kd10 = np.dot(kv, self.b4) 
        kd11 = np.dot(kv, self.b5) 
        kd12 = np.dot(kv, self.b6) 
        
        
        
        gxxpm = np.array([1, 1, 1,      1, 1, 1 ])
        gyypm = np.array([1, 1, 1,      1, 1, 1 ])
        gzzpm = np.array([1, 1, 1,      1, 1, 1 ])
        
        gxypm = np.array([1, 1, -1,      1, -1, 1 ])
        gyxpm = np.array([1, 1, -1,      1, -1, 1 ])
        
        gxzpm = np.array([1, 1, 1,      1, 1, 1 ])
        gzxpm = np.array([1, 1, 1,      1, 1, 1 ])
        
        gyzpm = np.array([1, 1, 1,      1, 1, 1 ])
        gzypm = np.array([1, 1, 1,      1, 1, 1 ])
        
        
        gxxpm2 = np.array([1, 1, 1,      1, 1, 1 ])
        gyypm2 = np.array([1, 1, 1,      1, 1, 1 ])
        gzzpm2 = np.array([1, 1, 1,      1, 1, 1 ])
        
        gxypm2 = np.array([1, -1, 1,      1, -1, 1 ])
        gyxpm2 = np.array([1, -1, 1,      1, -1, 1 ])
        
        gxzpm2 = np.array([1, -1, 1,      -1, 1, 1 ])
        gzxpm2 = np.array([1, -1, 1,      -1, 1, 1 ])
        
        gyzpm2 = np.array([1, 1, -1,      -1, -1, 1 ])
        gzypm2 = np.array([1, 1, -1,      -1, -1, 1 ])
        
        
        
        
        
        
        gspxpm = np.array([1, 1, 1, 1,         -1, 1, -1, 1,     -1, 1, -1, 1 ])
        gspypm = np.array([-1, 1, -1, 1,      1, 1, 1, 1,        -1, 1, -1, 1 ])
        gspzpm = np.array([-1, 1, -1, 1,     -1, 1, -1, 1,     -1, 1, -1, 1 ])
        
        
        ########################## d bands #####################################
        
        
        gxy_xy = np.array([1, 1, 1,      1, 1, 1 ])
        gxy_yz = np.array([1, 1, 1,      1, 1, 1 ])
        gxy_zx = np.array([1, 1, 1,      1, 1, 1 ])
        
        gxy_xxyy =np.array([1, 1, 1,      1, 1, 1 ])
        
        gyz_zx = np.array([1, 1, 1,      1, 1, 1 ])
        gyz_xxyy = np.array([1, 1, 1,      1, 1, 1 ])
        gzx_xxyy = np.array([1, 1, 1,      1, 1, 1 ])
        
        gxxyy_xxyy = np.array([1, 1, 1,      1, 1, 1 ])
        gzr_zr = np.array([1, 1, 1,      1, 1, 1 ])
        
        gzx_zr = np.array([1, 1, 1,      1, 1, 1 ])
        gxy_zr = np.array([1, 1, 1,      1, 1, 1 ])
        gyz_zr = np.array([1, 1, 1,      1, 1, 1 ])
        gxxyy_zr = np.array([1, 1, 1,      1, 1, 1 ])
        
        
        gxy_xy2 = np.array([1, 1, 1,      1, 1, 1 ])
        gxy_yz2 = np.array([-1, 1, 1,      1, -1, 1 ])
        gxy_zx2 = np.array([-1, -1, -1,      1, 1, 1 ])
        
        gxy_xxyy2 =np.array([1, -1, 1,      1, -1, 1 ])
        
        gyz_zx2 = np.array([-1, 1, 1,      -1, 1, 1 ])
        gyz_xxyy2 = np.array([1, 1, -1,      -1, -1, 1 ])
        gzx_xxyy2 = np.array([1, -1, 1,      -1, 1, 1 ])
        
        gxxyy_xxyy2 = np.array([1, 1, 1,      1, 1, 1 ])
        gzr_zr2 = np.array([1, 1, 1,      1, 1, 1 ])
        
        gzx_zr2 = np.array([-1, 1, 1,      1, -1, -1 ])
        gxy_zr2 = np.array([-1, 1, 1,      -1, 1, 1 ])
        gyz_zr2 = np.array([-1, -1, 1,      1, 1, -1 ])
        gxxyy_zr2 = np.array([1, 1, 1,      1, 1, 1 ])
        
        #Phase Factors for the Bloch Sum 
        self.g0_arr= np.array([np.exp(complex(0,kd1)),  #phase terms
                         np.exp(complex(0,kd2)), 
                         np.exp(complex(0,kd3)), 
                         np.exp(complex(0,kd4)), 
                         np.exp(complex(0,kd5)), 
                         np.exp(complex(0,kd6))
                         ])
    
        self.g0_arr2= np.array([
                         np.exp(complex(0,kd7)), 
                         np.exp(complex(0,kd8)), 
                         np.exp(complex(0,kd9)), 
                         np.exp(complex(0,kd10)), 
                         np.exp(complex(0,kd11)), 
                         np.exp(complex(0,kd12))
                         ])
    
    
        self.g0_arrT= np.array([np.exp(complex(0,kd1)),  #phase terms
                         np.exp(complex(0,kd2)), 
                         np.exp(complex(0,kd3)), 
                         np.exp(complex(0,kd4)), 
                         np.exp(complex(0,kd5)), 
                         np.exp(complex(0,kd6)),
                         np.exp(complex(0,kd7)), 
                         np.exp(complex(0,kd8)), 
                         np.exp(complex(0,kd9)), 
                         np.exp(complex(0,kd10)), 
                         np.exp(complex(0,kd11)), 
                         np.exp(complex(0,kd12))
                         ])
    
        self.g0c_arr= np.array([np.exp(-complex(0,kd1)),  #phase terms
                         np.exp(-complex(0,kd2)), 
                         np.exp(-complex(0,kd3)), 
                         np.exp(-complex(0,kd4)), 
                         np.exp(-complex(0,kd5)), 
                         np.exp(-complex(0,kd6))
                         ])
    
        self.g0c_arr2 = np.array([
                         np.exp(-complex(0,kd7)), 
                         np.exp(-complex(0,kd8)), 
                         np.exp(-complex(0,kd9)), 
                         np.exp(-complex(0,kd10)), 
                         np.exp(-complex(0,kd11)), 
                         np.exp(-complex(0,kd12))
                         ])
    
        self.gxy_arr=self.g0_arr*gxypm #Atoms of the same type
        self.gyx_arr=self.g0c_arr*gyxpm
        
        self.gxy_arr2=self.g0_arr2*gxypm2 #A to B atoms
        self.gyx_arr2=self.g0c_arr2*gyxpm2
        
        self.gxy_arrT = np.hstack((self.gxy_arr, self.gxy_arr2 ))
        self.gxy_arrT2 = np.hstack((self.gyx_arr, self.gyx_arr2 ))
        
        self.gxz_arr=self.g0_arr*gxzpm
        self.gzx_arr=self.g0c_arr*gzxpm
        
        self.gxz_arr2=self.g0_arr2*gxzpm2
        self.gzx_arr2=self.g0c_arr2*gzxpm2
        
        
        self.gyz_arr=self.g0_arr*gyzpm
        self.gzy_arr=self.g0c_arr*gzypm   
        
        self.gyz_arr2=self.g0_arr2*gyzpm2
        self.gzy_arr2=self.g0c_arr2*gzypm2
        
        

        
        
        self.gxx_arr=self.g0_arr*gxxpm
        self.gyy_arr=self.g0_arr*gyypm
        self.gzz_arr=self.g0_arr*gzzpm
        
        
        
        ############################ d bands #################################
        
        
        self.gdxy_xy_arr=self.g0_arr*gxy_xy
        
        self.gdyz_yz_arr=self.gdxy_xy_arr
        self.gdzx_zx_arr=self.gdxy_xy_arr
        
        self.gdxy_yz_arr=self.g0_arr*gxy_yz
        self.gdyz_xy_arr=self.gdxy_yz_arr
        
        self.gdxy_zx_arr=self.g0_arr*gxy_zx
        self.gdzx_xy_arr=self.gdxy_zx_arr
        
        self.gdxy_xxyy_arr=self.g0_arr*gxy_xxyy
        self.gdxxyy_xy_arr = self.gdxy_xxyy_arr
        
        self.gdyz_xxyy_arr=self.g0_arr*gyz_xxyy
        self.gdxxyy_yz_arr = self.gdyz_xxyy_arr
        
        self.gdzx_xxyy_arr=self.g0_arr*gzx_xxyy
        self.gdxxyy_zx_arr = self.gdzx_xxyy_arr
        
        self.gdyz_zx_arr=self.g0_arr*gyz_zx
        self.gdzx_yz_arr = self.gdyz_zx_arr
        
        self.gdxy_zr_arr=self.g0_arr*gxy_zr
        self.gdzr_xy_arr = self.gdxy_zr_arr
        
        self.gdyz_zr_arr=self.g0_arr*gyz_zr
        self.gdzr_yz_arr = self.gdyz_zr_arr
        self.gdzx_zr_arr=self.g0_arr*gzx_zr
        self.gdzr_zx_arr = self.gdzx_zr_arr
        
        self.gdxxyy_xxyy_arr=self.g0_arr*gxxyy_xxyy
        self.gdzr_zr_arr=self.g0_arr*gzr_zr
        
        self.gdxxyy_zr_arr=self.g0_arr*gxxyy_zr
        self.gdzr_xxyy_arr = self.gdxxyy_zr_arr
        
        ################     CONJUGATES     ###############
        
        
        self.gdxy_xy_arrc=self.g0c_arr*gxy_xy
        
        self.gdyz_yz_arrc=self.gdxy_xy_arrc
        self.gdzx_zx_arrc=self.gdxy_xy_arrc
        
        self.gdxy_yz_arrc=self.g0c_arr*gxy_yz
        self.gdyz_xy_arrc=self.gdxy_yz_arrc
        
        self.gdxy_zx_arrc=self.g0c_arr*gxy_zx
        self.gdzx_xy_arrc=self.gdxy_zx_arrc
        
        self.gdxy_xxyy_arrc=self.g0c_arr*gxy_xxyy
        self.gdxxyy_xy_arrc = self.gdxy_xxyy_arrc
        
        self.gdyz_xxyy_arrc=self.g0c_arr*gyz_xxyy
        self.gdxxyy_yz_arrc = self.gdyz_xxyy_arrc
        
        self.gdzx_xxyy_arrc=self.g0c_arr*gzx_xxyy
        self.gdxxyy_zx_arrc = self.gdzx_xxyy_arrc
        
        self.gdyz_zx_arrc=self.g0c_arr*gyz_zx
        self.gdzx_yz_arrc = self.gdyz_zx_arrc
        
        self.gdxy_zr_arrc=self.g0c_arr*gxy_zr
        self.gdzr_xy_arrc = self.gdxy_zr_arrc
        
        self.gdyz_zr_arrc=self.g0c_arr*gyz_zr
        self.gdzr_yz_arrc = self.gdyz_zr_arrc
        self.gdzx_zr_arrc=self.g0c_arr*gzx_zr
        self.gdzr_zx_arrc = self.gdzx_zr_arrc
        
        self.gdxxyy_xxyy_arrc=self.g0c_arr*gxxyy_xxyy
        self.gdzr_zr_arrc=self.g0c_arr*gzr_zr
        
        self.gdxxyy_zr_arrc=self.g0c_arr*gxxyy_zr
        self.gdzr_xxyy_arrc = self.gdxxyy_zr_arrc
        
        ############################ d bands #################################
        
        
        self.gdxy_xy_arr2=self.g0_arr2*gxy_xy2
        
        self.gdyz_yz_arr2=self.gdxy_xy_arr2
        self.gdzx_zx_arr2=self.gdxy_xy_arr2
        
        self.gdxy_yz_arr2=self.g0_arr2*gxy_yz2
        self.gdyz_xy_arr2=self.gdxy_yz_arr2
        
        self.gdxy_zx_arr2=self.g0_arr2*gxy_zx2
        self.gdzx_xy_arr2=self.gdxy_zx_arr2
        
        self.gdxy_xxyy_arr2=self.g0_arr2*gxy_xxyy2
        self.gdxxyy_xy_arr2 = self.gdxy_xxyy_arr2
        
        self.gdyz_xxyy_arr2=self.g0_arr2*gyz_xxyy2
        self.gdxxyy_yz_arr2 = self.gdyz_xxyy_arr2
        
        self.gdzx_xxyy_arr2=self.g0_arr2*gzx_xxyy2
        self.gdxxyy_zx_arr2 = self.gdzx_xxyy_arr2
        
        self.gdyz_zx_arr2=self.g0_arr2*gyz_zx2
        self.gdzx_yz_arr2 = self.gdyz_zx_arr2
        
        self.gdxy_zr_arr2=self.g0_arr2*gxy_zr2
        self.gdzr_xy_arr2 = self.gdxy_zr_arr2
        
        self.gdyz_zr_arr2=self.g0_arr2*gyz_zr2
        self.gdzr_yz_arr2 = self.gdyz_zr_arr2
        self.gdzx_zr_arr2=self.g0_arr2*gzx_zr2
        self.gdzr_zx_arr2 = self.gdzx_zr_arr2
        
        self.gdxxyy_xxyy_arr2=self.g0_arr2*gxxyy_xxyy2
        self.gdzr_zr_arr2=self.g0_arr2*gzr_zr2
        
        self.gdxxyy_zr_arr2=self.g0_arr2*gxxyy_zr2
        self.gdzr_xxyy_arr2 = self.gdxxyy_zr_arr2
        
        ################     CONJUGATES     ###############
        
        
        self.gdxy_xy_arrc2=self.g0c_arr2*gxy_xy2
        
        self.gdyz_yz_arrc2=self.gdxy_xy_arrc2
        self.gdzx_zx_arrc2=self.gdxy_xy_arrc2
        
        self.gdxy_yz_arrc2=self.g0c_arr2*gxy_yz2
        self.gdyz_xy_arrc2=self.gdxy_yz_arrc2
        
        self.gdxy_zx_arrc2=self.g0c_arr2*gxy_zx2
        self.gdzx_xy_arrc2=self.gdxy_zx_arrc2
        
        self.gdxy_xxyy_arrc2=self.g0c_arr2*gxy_xxyy2
        self.gdxxyy_xy_arrc2 = self.gdxy_xxyy_arrc2
        
        self.gdyz_xxyy_arrc2=self.g0c_arr2*gyz_xxyy2
        self.gdxxyy_yz_arrc2 = self.gdyz_xxyy_arrc2
        
        self.gdzx_xxyy_arrc2=self.g0c_arr2*gzx_xxyy2
        self.gdxxyy_zx_arrc2 = self.gdzx_xxyy_arrc2
        
        self.gdyz_zx_arrc2=self.g0c_arr2*gyz_zx2
        self.gdzx_yz_arrc2 = self.gdyz_zx_arrc2
        
        self.gdxy_zr_arrc2=self.g0c_arr2*gxy_zr2
        self.gdzr_xy_arrc2 = self.gdxy_zr_arrc2
        
        self.gdyz_zr_arrc2=self.g0c_arr2*gyz_zr2
        self.gdzr_yz_arrc2 = self.gdyz_zr_arrc2
        self.gdzx_zr_arrc2=self.g0c_arr2*gzx_zr2
        self.gdzr_zx_arrc2 = self.gdzx_zr_arrc2
        
        self.gdxxyy_xxyy_arrc2=self.g0c_arr2*gxxyy_xxyy2
        self.gdzr_zr_arrc2=self.g0c_arr2*gzr_zr2
        
        self.gdxxyy_zr_arrc2=self.g0c_arr2*gxxyy_zr2
        self.gdzr_xxyy_arrc2 = self.gdxxyy_zr_arrc2
    
    def initial_energies(self, signifier, band):
        l = self.l
        m = self.m 
        n = self.n 
        ll = self.ll
        mm = self.mm
        nn = self.nn 
        
        
        Kp = (7.62/(self.a**2)) #From eqn V_{ll'm} = n_{ll'm}*h**2/m*d**2
        Kd = (7.62*((self.rd**3)/(self.a**5))**2)
        ones = np.ones((12,1))
                
        if signifier == 'fcc':
            
            if band ==  'p':
            
                const = (7.62/(1.61**2)) #From eqn V_{ll'm} = n_{ll'm}*h**2/m*d**2
                ones = np.ones((12,1))
                #ones2 = np.ones((6,1))
                
                self.Es = self.sssig*ones
         
                self.Espx = self.darr[:, : 1]*self.spsig
                self.Espy = self.darr[:, 1: 2]*self.spsig
                self.Espz = self.darr[:, 2: 3]*self.spsig       
                
                
                self.Epxx = (self.ll*self.ppsig + (ones - self.ll)*self.pppi)
                self.Epyy = (self.mm*self.ppsig + (ones - self.mm)*self.pppi)
                self.Epzz = (self.nn*self.ppsig + (ones - self.nn)*self.pppi)
                
                self.Exy = self.l*self.m*(self.ppsig - self.pppi)
                self.Exz = self.l*self.n*(self.ppsig - self.pppi)
                self.Eyz = self.m*self.n*(self.ppsig - self.pppi)
                
            if band == 'd':
                
                self.Edxy_xy = ( 3*ll*mm*self.ddsig     +   (ll + mm -4*ll*mm)*self.ddpi    +    (nn + ll*mm)*self.dddelta )
                self.Edyz_yz = (3*mm*nn*self.ddsig      +   (mm + nn -4*mm*nn)*self.ddpi    +    (ll + mm*nn)*self.dddelta)
                self.Edzx_zx = (3*nn*ll*self.ddsig      +   (nn + ll -4*nn*ll)*self.ddpi    +    (mm + nn*ll)*self.dddelta)
                
                self.Edxy_yz = (3*l*mm*n*self.ddsig     +   l*n*(ones -  4*mm )*self.ddpi   +    l*n*(mm - ones)*self.dddelta)
                self.Edxy_zx = (3*ll*m*n*self.ddsig     +   m*n*(ones -  4*ll )*self.ddpi   +    m*n*(ll - ones)*self.dddelta)
                self.Edyz_zx = (3*nn*m*l*self.ddsig     +   m*l*(ones -  4*nn )*self.ddpi   +    m*l*(nn - ones)*self.dddelta)
                
                self.Edxy_xxyy = ((3./2)*l*m*(ll - mm)*self.ddsig       +       2*l*m*(mm -  ll )*self.ddpi                 +    0.5*l*m*(ll - mm)*self.dddelta)
                self.Edyz_xxyy = ((3./2)*m*n*(ll - mm)*self.ddsig       -       m*n*(ones + 2*(ll -  mm) )*self.ddpi        +    m*n*(ones + 0.5*(ll - mm))*self.dddelta)
                self.Edzx_xxyy = ((3./2)*n*l*(ll - mm)*self.ddsig       +       n*l*(ones - 2*(ll -  mm) )*self.ddpi        -    n*l*(ones - 0.5*(ll - mm))*self.dddelta)
                
                self.Edxy_zr = (np.sqrt(3)*l*m*(nn - 0.5*(ll + mm))*self.ddsig      -   np.sqrt(3)*2*l*m*nn*self.ddpi               +   0.5*np.sqrt(3)*l*m*(ones + nn)*self.dddelta)
                self.Edyz_zr = (np.sqrt(3)*m*n*(nn - 0.5*(ll + mm))*self.ddsig      +   np.sqrt(3)*m*n*(ll + mm - nn)*self.ddpi     -   0.5*np.sqrt(3)*m*n*(ll + mm)*self.dddelta)
                self.Edzx_zr = (np.sqrt(3)*l*n*(nn - 0.5*(ll + mm))*self.ddsig      +   np.sqrt(3)*l*n*(ll + mm - nn)*self.ddpi     -   0.5*np.sqrt(3)*l*n*(ll + mm)*self.dddelta)
                
                self.Edxxyy_xxyy = (0.75*((ll - mm)**2)*self.ddsig      +     (ll + mm - (ll -  mm)**2 )*self.ddpi    +    (nn + 0.25*(ll -  mm)**2)*self.dddelta)
                self.Edxxyy_zr = (0.5*np.sqrt(3)*(ll - mm)*(nn - 0.5*(ll + mm))*self.ddsig     +   np.sqrt(3)*nn*(mm - ll)*self.ddpi     +   0.25*np.sqrt(3)*(ones + nn)*(ll -  mm)*self.dddelta)
                self.Edzr_zr = (((nn - 0.5*(ll + mm))**2)*self.ddsig       +      3*nn*(mm + ll)*self.ddpi      +       0.75*((ll +  mm)**2)*self.dddelta)
                
                
            
            
            
            
            
        
    def Hamiltonian_d(self, kv):
        
        """
        #Form of the Hamiltonian Matrix in terms of orbitals
        
            s         px         py        pz        xy     yz     zx     (x^2 - y^2)     (3z^2 - r^2)
        s   sssig*g0  spsig*g1  spsig*g2  spsig*g3  sdsig*g1
        
        px
        
        py
        
        pz
        
        xy
        
        yz
        
        zx
        
        (x^2 - y^2)
        
        (3z^2 - r^2)
        
        
        """
        #self.initial_energies('fcc', 'd')
        self.phasefactors(kv)
    
    
        M = 12*12.*np.asarray([
                [np.dot(self.g0_arr,self.Edxy_xy[:6])[0],  np.dot(self.gdxy_yz_arr,self.Edxy_yz[:6])[0], np.dot(self.gdxy_zx_arr,self.Edxy_zx[:6])[0],
                         np.dot(self.gdxy_xxyy_arr,self.Edxy_xxyy[:6])[0],  np.dot(self.gdxy_zr_arr,self.Edxy_zr[:6])[0] ,
                    np.dot(self.g0_arr2,self.Edxy_xy[6:])[0],  np.dot(self.gdxy_yz_arr2,self.Edxy_yz[6:])[0], np.dot(self.gdxy_zx_arr2,self.Edxy_zx[6:])[0],
                         np.dot(self.gdxy_xxyy_arr2,self.Edxy_xxyy[6:])[0],  np.dot(self.gdxy_zr_arr2,self.Edxy_zr[6:])[0] ], 
                
                [np.dot(self.gdxy_yz_arr,self.Edxy_yz[:6])[0],  np.dot(self.g0_arr,self.Edyz_yz[:6])[0], np.dot(self.gdyz_zx_arr,self.Edyz_zx[:6])[0],
                         np.dot(self.gdyz_xxyy_arr,self.Edyz_xxyy[:6])[0],  np.dot(self.gdyz_zr_arr,self.Edyz_zr[:6])[0], 
                    np.dot(self.gdxy_yz_arr2,self.Edxy_yz[6:])[0],  np.dot(self.g0_arr2,self.Edyz_yz[6:])[0], np.dot(self.gdyz_zx_arr2,self.Edyz_zx[6:])[0],
                         np.dot(self.gdyz_xxyy_arr2,self.Edyz_xxyy[6:])[0],  np.dot(self.gdyz_zr_arr2,self.Edyz_zr[6:])[0]], 
                
                [np.dot(self.gdxy_zx_arr,self.Edxy_zx[:6])[0],  np.dot(self.gdyz_zx_arr,self.Edyz_zx[:6])[0], np.dot(self.g0_arr,self.Edzx_zx[:6])[0],
                         np.dot(self.gdzx_xxyy_arr,self.Edzx_xxyy[:6])[0],  np.dot(self.gdzx_zr_arr,self.Edzx_zr[:6])[0],
                    np.dot(self.gdxy_zx_arr2,self.Edxy_zx[6:])[0],  np.dot(self.gdyz_zx_arr2,self.Edyz_zx[6:])[0], np.dot(self.g0_arr2,self.Edzx_zx[6:])[0],
                         np.dot(self.gdzx_xxyy_arr2,self.Edzx_xxyy[6:])[0],  np.dot(self.gdzx_zr_arr2,self.Edzx_zr[6:])[0]], 
                           
                [np.dot(self.gdxy_xxyy_arr,self.Edxy_xxyy[:6])[0], np.dot(self.gdyz_xxyy_arr,self.Edyz_xxyy[:6])[0], np.dot(self.gdzx_xxyy_arr,self.Edzx_xxyy[:6])[0],
                         np.dot(self.g0_arr,self.Edxxyy_xxyy[:6])[0],  np.dot(self.gdxxyy_zr_arr,self.Edxxyy_zr[:6])[0],
                    np.dot(self.gdxy_xxyy_arr2,self.Edxy_xxyy[6:])[0], np.dot(self.gdyz_xxyy_arr2,self.Edyz_xxyy[6:])[0], np.dot(self.gdzx_xxyy_arr2,self.Edzx_xxyy[6:])[0],
                         np.dot(self.g0_arr2,self.Edxxyy_xxyy[6:])[0],  np.dot(self.gdxxyy_zr_arr2,self.Edxxyy_zr[6:])[0]], 
                 
                [np.dot(self.gdxy_zr_arr,self.Edxy_zr[:6])[0],  np.dot(self.gdyz_zr_arr,self.Edyz_zr[:6])[0], np.dot(self.gdzx_zr_arr,self.Edzx_zr[:6])[0], 
                         np.dot(self.gdxxyy_zr_arr,self.Edxxyy_zr[:6])[0],  np.dot(self.g0_arr,self.Edzr_zr[:6])[0],
                    np.dot(self.gdxy_zr_arr2,self.Edxy_zr[6:])[0],  np.dot(self.gdyz_zr_arr2,self.Edyz_zr[6:])[0], np.dot(self.gdzx_zr_arr2,self.Edzx_zr[6:])[0], 
                         np.dot(self.gdxxyy_zr_arr2,self.Edxxyy_zr[6:])[0],  np.dot(self.g0_arr2,self.Edzr_zr[6:])[0] ], 
                
                
                
                
                
                [np.dot(self.g0c_arr2,self.Edxy_xy[:6])[0],  np.dot(self.gdxy_yz_arrc2,self.Edxy_yz[:6])[0], np.dot(self.gdxy_zx_arrc2,self.Edxy_zx[:6])[0],
                         np.dot(self.gdxy_xxyy_arr2,self.Edxy_xxyy[:6])[0],  np.dot(self.gdxy_zr_arrc2,self.Edxy_zr[:6])[0] ,
                    np.dot(self.g0c_arr,self.Edxy_xy[6:])[0],  np.dot(self.gdxy_yz_arrc,self.Edxy_yz[6:])[0], np.dot(self.gdxy_zx_arrc,self.Edxy_zx[6:])[0],
                         np.dot(self.gdxy_xxyy_arrc,self.Edxy_xxyy[6:])[0],  np.dot(self.gdxy_zr_arrc,self.Edxy_zr[6:])[0] ], 
                
                [np.dot(self.gdxy_yz_arrc2,self.Edxy_yz[:6])[0],  np.dot(self.g0c_arr2,self.Edyz_yz[:6])[0], np.dot(self.gdyz_zx_arrc2,self.Edyz_zx[:6])[0],
                         np.dot(self.gdyz_xxyy_arrc2,self.Edyz_xxyy[:6])[0],  np.dot(self.gdyz_zr_arrc2,self.Edyz_zr[:6])[0], 
                    np.dot(self.gdxy_yz_arrc,self.Edxy_yz[6:])[0],  np.dot(self.g0c_arr,self.Edyz_yz[6:])[0], np.dot(self.gdyz_zx_arr,self.Edyz_zx[6:])[0],
                         np.dot(self.gdyz_xxyy_arrc,self.Edyz_xxyy[6:])[0],  np.dot(self.gdyz_zr_arrc,self.Edyz_zr[6:])[0]], 
                
                [np.dot(self.gdxy_zx_arrc2,self.Edxy_zx[:6])[0],  np.dot(self.gdyz_zx_arrc2,self.Edyz_zx[:6])[0], np.dot(self.g0c_arr2,self.Edzx_zx[:6])[0],
                         np.dot(self.gdzx_xxyy_arrc2,self.Edzx_xxyy[:6])[0],  np.dot(self.gdzx_zr_arrc2,self.Edzx_zr[:6])[0],
                    np.dot(self.gdxy_zx_arrc,self.Edxy_zx[6:])[0],  np.dot(self.gdyz_zx_arrc,self.Edyz_zx[6:])[0], np.dot(self.g0c_arr,self.Edzx_zx[6:])[0],
                         np.dot(self.gdzx_xxyy_arrc,self.Edzx_xxyy[6:])[0],  np.dot(self.gdzx_zr_arrc,self.Edzx_zr[6:])[0]], 
                           
                [np.dot(self.gdxy_xxyy_arrc2,self.Edxy_xxyy[:6])[0], np.dot(self.gdyz_xxyy_arrc2,self.Edyz_xxyy[:6])[0], np.dot(self.gdzx_xxyy_arrc2,self.Edzx_xxyy[:6])[0],
                         np.dot(self.g0c_arr2,self.Edxxyy_xxyy[:6])[0],  np.dot(self.gdxxyy_zr_arrc2,self.Edxxyy_zr[:6])[0],
                    np.dot(self.gdxy_xxyy_arrc,self.Edxy_xxyy[6:])[0], np.dot(self.gdyz_xxyy_arrc,self.Edyz_xxyy[6:])[0], np.dot(self.gdzx_xxyy_arrc,self.Edzx_xxyy[6:])[0],
                         np.dot(self.g0c_arr,self.Edxxyy_xxyy[6:])[0],  np.dot(self.gdxxyy_zr_arrc,self.Edxxyy_zr[6:])[0]], 
                 
                [np.dot(self.gdxy_zr_arrc2,self.Edxy_zr[:6])[0],  np.dot(self.gdyz_zr_arrc2,self.Edyz_zr[:6])[0], np.dot(self.gdzx_zr_arrc2,self.Edzx_zr[:6])[0], 
                         np.dot(self.gdxxyy_zr_arrc2,self.Edxxyy_zr[:6])[0],  np.dot(self.g0c_arr2,self.Edzr_zr[:6])[0],
                    np.dot(self.gdxy_zr_arrc,self.Edxy_zr[6:])[0],  np.dot(self.gdyz_zr_arrc,self.Edyz_zr[6:])[0], np.dot(self.gdzx_zr_arrc,self.Edzx_zr[6:])[0], 
                         np.dot(self.gdxxyy_zr_arrc,self.Edxxyy_zr[6:])[0],  np.dot(self.g0c_arr,self.Edzr_zr[6:])[0] ]
                 
                 
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
        
    

    
    def Hamiltonian_sp(self, kv):
        
        """
        #Form of the Hamiltonian Matrix in terms of orbitals
        
               s         px         py        pz      
        s   sssig*g0  spsig*g1  spsig*g2  spsig*g3  
        
        px  -spsig*g1  Ep  spsig*g2  spsig*g3  
        
        py  -spsig*g2  spsig*g1  Ep  spsig*g3  
        
        pz  -spsig*g3  spsig*g1  spsig*g2  Ep  
        
        """
        
        self.initial_energies('fcc')
        self.phasefactors(kv)
    
        M = 12.*np.asarray([[np.dot(self.g0_arr, self.Es)[0], np.dot(self.gspx_arr,self.Espx)[0],  np.dot(self.gspy_arr,self.Espy)[0],  np.dot(self.gspz_arr,self.Espz)[0] ]
                [np.dot(self.gspx_arr,self.Espx)[0], np.dot(self.g0_arr,self.Epxx)[0],  np.dot(self.gxy_arr,self.Exy)[0],  np.dot(self.gxz_arr,self.Exz)[0] ], 
                [np.dot(self.gspy_arr,self.Espy)[0], np.dot(self.gyx_arr,self.Exy)[0],  np.dot(self.g0_arr,self.Epyy)[0],  np.dot(self.gyz_arr,self.Eyz)[0] ], 
                [np.dot(self.gspz_arr,self.Espz)[0], np.dot(self.gzx_arr,self.Exz)[0],  np.dot(self.gzy_arr,self.Eyz)[0],  np.dot(self.g0_arr,self.Epzz)[0] ]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    
    def Hamiltonian_p(self, kv):
                
        
        self.phasefactors(kv)
    
        #Not sure if this should have zeros in due to the second atom
        M = 12.*np.asarray([
                [np.dot(self.g0_arr,self.Epxx[:6])[0],  np.dot(self.g0_arr,self.Exy[:6])[0],  np.dot(self.g0_arr,self.Exz[:6])[0],
                        np.dot(self.g0_arr2,self.Epxx[6:])[0],  np.dot(self.g0_arr2,self.Exy[6:])[0],  np.dot(self.g0_arr2,self.Exz[6:])[0]], 
                [np.dot(self.g0_arr,self.Exy[:6])[0],  np.dot(self.g0_arr,self.Epyy[:6])[0],  np.dot(self.g0_arr,self.Eyz[:6])[0], 
                        np.dot(self.g0_arr2,self.Exy[6:])[0],  np.dot(self.g0_arr2,self.Epyy[6:])[0],  np.dot(self.g0_arr2,self.Eyz[6:])[0]], 
                [np.dot(self.g0_arr,self.Exz[:6])[0],  np.dot(self.g0_arr,self.Eyz[:6])[0],  np.dot(self.g0_arr,self.Epzz[:6])[0],
                        np.dot(self.g0_arr2,self.Exz[6:])[0],  np.dot(self.g0_arr2,self.Eyz[6:])[0],  np.dot(self.g0_arr2,self.Epzz[6:])[0]],
                
                [np.dot(self.g0c_arr2,self.Epxx[6:])[0],  np.dot(self.g0c_arr2,self.Exy[6:])[0],  np.dot(self.g0c_arr2,self.Exz[6:])[0],
                        np.dot(self.g0c_arr,self.Epxx[:6])[0],  np.dot(self.g0c_arr,self.Exy[:6])[0],  np.dot(self.g0c_arr,self.Exz[:6])[0]], 
                [np.dot(self.g0c_arr2,self.Exy[6:])[0],  np.dot(self.g0c_arr2,self.Epyy[6:])[0],  np.dot(self.g0c_arr2,self.Eyz[6:])[0], 
                        np.dot(self.g0c_arr,self.Exy[:6])[0],  np.dot(self.g0c_arr,self.Epyy[:6])[0],  np.dot(self.g0c_arr,self.Eyz[:6])[0]], 
                [np.dot(self.g0c_arr2,self.Exz[6:])[0],  np.dot(self.g0c_arr2,self.Eyz[6:])[0],  np.dot(self.g0c_arr2,self.Epzz[6:])[0],
                        np.dot(self.g0c_arr,self.Exz[:6])[0],  np.dot(self.g0c_arr,self.Eyz[:6])[0],  np.dot(self.g0c_arr,self.Epzz[:6])[0]]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    
    def Hamiltonian_s(self, kv):
        
        """
        #Form of the Hamiltonian Matrix in terms of orbitals
        
               s        
        s   sssig*g0  

        """
        
        
        self.phasefactors(kv)
    
    
        M = np.array([
                [np.dot(self.g0_arr,self.Es[:6])[0], np.dot(self.g0_arr2,self.Es[6:])[0] ],
                [np.dot(self.g0c_arr2,self.Es[6:])[0], np.dot(self.g0c_arr,self.Es[:6])[0] ]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M

    
    def eigenvalues(self, H):
    
        w, v = la.eig(H)
        return w
    
    def band_structure_s(self, ki, kf, ax, reverse, n1, n2):
        """ kf and ki must be 1x3 arrays
        """
        
        k_diff = kf-ki
        self.kvals = []
        self.energiess1 = []
        self.energiess2 = []
        self.initial_energies('fcc', 'p')
        
        
        loops = 200
        for i in range(loops):
            kr = ki + (i/loops)*k_diff
            #self.phasefactors(kr )
            self.M = self.Hamiltonian_s(kr)
            eigenvals = np.linalg.eigh(self.M)[0]
            print('s eigenvalues', eigenvals)
            self.energiess1.append(eigenvals[0])
            self.energiess2.append(eigenvals[1])
            self.kvals.append(i/float(loops))
            

        ax.plot(self.kvals, self.energiess1)
        ax.plot(self.kvals, self.energiess2)
        ax.set_title('%s to %s'%(n1, n2)) 
        if reverse==True:
            ax.set_xlim([np.max(self.kvals), np.min(self.kvals)])
        #plt.show()
            
            
    def band_structure_p(self, ki, kf, ax, reverse, n1, n2):
        """ kf and ki must be 1x3 arrays
        """

        self.kvals = []
        self.px_energies = []
        self.py_energies = []
        self.pz_energies = []
        self.px_energies2 = []
        self.py_energies2 = []
        self.pz_energies2 = []
        k_diff = kf-ki
        self.initial_energies('fcc', 'p')
        loops = 200
        for i in range(loops):
            kr = ki + (i/float(loops))*k_diff
            #self.phasefactors(kr )
            self.M = self.Hamiltonian_p(kr)
            eigenvals = np.linalg.eigh(self.M)[0]
            print('p eigenvalues', eigenvals)
            self.px_energies.append(eigenvals[0])
            self.py_energies.append(eigenvals[1])
            self.pz_energies.append(eigenvals[2])
            self.px_energies2.append(eigenvals[3])
            self.py_energies2.append(eigenvals[4])
            self.pz_energies2.append(eigenvals[5])
            self.kvals.append((i/float(loops)))

        ax.plot(self.kvals, self.px_energies)#, marker=style, color=k)
        ax.plot(self.kvals, self.py_energies)#, bo)
        ax.plot(self.kvals, self.pz_energies)#, r*)
        ax.plot(self.kvals, self.px_energies2)#, marker=style, color=k)
        ax.plot(self.kvals, self.py_energies2)#, bo)
        ax.plot(self.kvals, self.pz_energies2)#, r*)
        
        ax.set_title('%s to %s'%(n1, n2)) 
        if reverse==True:
            ax.set_xlim([np.max(self.kvals), np.min(self.kvals)])
            
            
    def band_structure_d(self, ki, kf, ax, reverse, n1, n2):
        """ kf and ki must be 1x3 arrays
        """

        self.kvals = []

        self.xy_energies = []
        self.yz_energies = []
        self.zx_energies = []
        self.xxyy_energies = []
        self.zr_energies = []
        
        self.xy_energies2 = []
        self.yz_energies2 = []
        self.zx_energies2 = []
        self.xxyy_energies2 = []
        self.zr_energies2 = []
        k_diff = kf-ki
        self.initial_energies('fcc', 'd')
        loops = 200
        for i in range(loops):
            kr = ki + (i/float(loops))*k_diff
            #self.phasefactors(kr )
            self.M = self.Hamiltonian_d(kr)
            eigenvals = np.linalg.eigh(self.M)[0]
            print('d eigenvalues', eigenvals)
            self.xy_energies.append(eigenvals[0])
            self.yz_energies.append(eigenvals[1])
            self.zx_energies.append(eigenvals[2])
            self.xxyy_energies.append(eigenvals[3])
            self.zr_energies.append(eigenvals[4])
            
            self.xy_energies2.append(eigenvals[5])
            self.yz_energies2.append(eigenvals[6])
            self.zx_energies2.append(eigenvals[7])
            self.xxyy_energies2.append(eigenvals[8])
            self.zr_energies2.append(eigenvals[9])
            self.kvals.append((i/float(loops)))

        ax.plot(self.kvals, self.xy_energies)#, marker=style, color=k)
        ax.plot(self.kvals, self.yz_energies)#, bo)
        ax.plot(self.kvals, self.zx_energies)#, r*)
        ax.plot(self.kvals, self.xxyy_energies)#, r*)
        ax.plot(self.kvals, self.zr_energies)#, r*)
        ax.plot(self.kvals, self.xy_energies2)#, marker=style, color=k)
        ax.plot(self.kvals, self.yz_energies2)#, bo)
        ax.plot(self.kvals, self.zx_energies2)#, r*)
        ax.plot(self.kvals, self.xxyy_energies2)#, r*)
        ax.plot(self.kvals, self.zr_energies2)#, r*)
        
        ax.set_title('%s to %s'%(n1, n2)) 
        if reverse==True:
            ax.set_xlim([np.max(self.kvals), np.min(self.kvals)])
                
                
#def band_structure_plotting_fcc(con):
    
    
def sband_script(con):
    
    
    Gamma = np.array([0,0,0])
    K = (2*np.pi/con.a)*np.array([0,(2/3.),0])
    K_2 = (2*np.pi/con.a)*np.array([(1/3),(1/np.sqrt(3.)),0])
    M = (np.pi/con.a)*np.array([1,(-1/np.sqrt(3)),0])
    A = (np.pi/con.c)*np.array([0,0,1])
    H = (2*np.pi)*np.array([(2./(3*con.a)),0,(1./(2*con.c))])
    H_2 = (2*np.pi)*np.array([(1./(3*con.a)),(1/(con.a*np.sqrt(3.))),(1./(2*con.c))])
    L = (np.pi)*np.array([(1./(con.a)),(-1./(np.sqrt(3)*con.a)),(1./(con.c))])
    
    fig, axes = plt.subplots(1,8,sharey='all')

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax6 = axes[5]
    ax7 = axes[6]
    ax8 = axes[7]
    
    con.band_structure_s(Gamma, K_2, ax1, False, 'Gamma', 'K')
    con.band_structure_s(K_2, M, ax2, False, 'K', 'M')
    con.band_structure_s(M, Gamma, ax3, False, 'M', 'Gamma')
    con.band_structure_s(Gamma, A, ax4, False, 'Gamma', 'A')
    con.band_structure_s(K_2, H_2, ax5, False, 'K', 'H')
    con.band_structure_s(H_2, L, ax6, False, 'H', 'L')
    con.band_structure_s(L, A, ax7, False, 'L', 'A')
    con.band_structure_s(A, H_2, ax8, False, 'A', 'H')
    
    ax1.set_ylabel('E (eV)')
    ax3.set_xlabel('¦K¦')
    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)

    plt.suptitle('s-bands:hcp')
    plt.show()  



def pband_script(con):
    
    
    Gamma = np.array([0,0,0])
    K = (2*np.pi/con.a)*np.array([0,(2/3.),0])
    K_2 = (2*np.pi/con.a)*np.array([(1/3),(1/np.sqrt(3.)),0])
    M = (np.pi/con.a)*np.array([1,(-1/np.sqrt(3)),0])
    A = (np.pi/con.c)*np.array([0,0,1])
    H = (2*np.pi)*np.array([(2./(3*con.a)),0,(1./(2*con.c))])
    H_2 = (2*np.pi)*np.array([(1./(3*con.a)),(1/(con.a*np.sqrt(3.))),(1./(2*con.c))])
    L = (np.pi)*np.array([(1./(con.a)),(-1./(np.sqrt(3)*con.a)),(1./(con.c))])
    
    fig, axes = plt.subplots(1,8,sharey='all')

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax6 = axes[5]
    ax7 = axes[6]
    ax8 = axes[7]
    
    con.band_structure_p(Gamma, K_2, ax1, False, 'Gamma', 'K')
    con.band_structure_p(K_2, M, ax2, False, 'K', 'M')
    con.band_structure_p(M, Gamma, ax3, False, 'M', 'Gamma')
    con.band_structure_p(Gamma, A, ax4, False, 'Gamma', 'A')
    con.band_structure_p(K_2, H_2, ax5, False, 'K', 'H')
    con.band_structure_p(H_2, L, ax6, False, 'H', 'L')
    con.band_structure_p(L, A, ax7, False, 'L', 'A')
    con.band_structure_p(A, H_2, ax8, False, 'A', 'H')
    
    ax3.set_xlabel('¦K¦')
    ax1.set_ylabel('E (eV)')
    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)
    plt.suptitle('p-bands:hcp')
    plt.show()  
    

def dband_script(con):
    
    
   
    
    
    Gamma = np.array([0,0,0])
    K = (2*np.pi/con.a)*np.array([0,(2/3.),0])
    K_2 = (2*np.pi/con.a)*np.array([(1/3),(1/np.sqrt(3.)),0])
    M = (np.pi/con.a)*np.array([1,(-1/np.sqrt(3)),0])
    A = (np.pi/con.c)*np.array([0,0,1])
    H = (2*np.pi)*np.array([(2./(3*con.a)),0,(1./(2*con.c))])
    H_2 = (2*np.pi)*np.array([(1./(3*con.a)),(1/(con.a*np.sqrt(3.))),(1./(2*con.c))])
    L = (np.pi)*np.array([(1./(con.a)),(-1./(np.sqrt(3)*con.a)),(1./(con.c))])
    
    fig, axes = plt.subplots(1,8,sharey='all')

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax6 = axes[5]
    ax7 = axes[6]
    ax8 = axes[7]
    
    con.band_structure_d(Gamma, K_2, ax1, False, 'Gamma', 'K')
    con.band_structure_d(K_2, M, ax2, False, 'K', 'M')
    con.band_structure_d(M, Gamma, ax3, False, 'M', 'Gamma')
    con.band_structure_d(Gamma, A, ax4, False, 'Gamma', 'A')
    con.band_structure_d(K_2, H_2, ax5, False, 'K', 'H')
    con.band_structure_d(H_2, L, ax6, False, 'H', 'L')
    con.band_structure_d(L, A, ax7, False, 'L', 'A')
    con.band_structure_d(A, H_2, ax8, False, 'A', 'H')
    
    ax5.set_xlabel('¦K¦')
    ax1.set_ylabel('E (eV)')
    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)
    plt.suptitle('d-bands:hcp')
    plt.show()  
    

con = Hconstruct() 
p
band_script(con) 



