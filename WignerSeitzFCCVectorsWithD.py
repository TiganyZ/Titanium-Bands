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


#####    FCC   #####


class Hconstruct:
    
    def __init__(self, k=np.array([0,0,0]), a = 1.61,#1.61, #a is in Angstrom
                 d1=np.array([0,1,1]), d2=np.array([1,0,1]), 
                 d3=np.array([1,1,0]), d4=np.array([0,-1,1]),
                 d5=np.array([0,1,-1]), d6=np.array([0,-1,-1]),
                 d7=np.array([-1,0,1]), d8=np.array([1,0,-1]), d9=np.array([-1,0,-1]), 
                d10=np.array([-1,1,0]), d11=np.array([1,-1,0]), d12=np.array([-1,-1,0]), 
                Es=1, Ep=1, 
                 Ed=1, sssig=-1.40, spsig=1.84, ppsig=3.24, pppi=-0.81, sdsig=-3.16, pdsig=-2.95, #sssig to pppi from Harrison and general, d from Titanium solid state table
                 pdpi=1.36, ddsigma=-16.2, ddpi=8.75, dddelta=0, rd = 1.08 ):

        self.a = a
        self.d1 = (self.a/2.)*d1
        self.d2 = (self.a/2.)*d2
        self.d3 = (self.a/2.)*d3
        self.d4 = (self.a/2.)*d4
        self.d5 = (self.a/2.)*d5
        self.d6 = (self.a/2.)*d6
        self.d7 = (self.a/2.)*d7
        self.d8 = (self.a/2.)*d8
        self.d9 = (self.a/2.)*d9
        self.d10 = (self.a/2.)*d10
        self.d11 = (self.a/2.)*d11
        self.d12 = (self.a/2.)*d12
        
        self.darr = np.asarray([self.d1,self.d4,self.d5,self.d6,self.d2,self.d7,
                           self.d8,self.d9,self.d3,self.d10, self.d11, self.d12])/np.sqrt(2)
        self.l = self.darr[:, :1].astype(np.complex_)
        self.m = self.darr[:, 1:2].astype(np.complex_)
        self.n = self.darr[:, 2:3].astype(np.complex_)
        
        self.ll = (self.darr[:, :1]*self.darr[:, :1]).astype(np.complex_)
        self.mm = (self.darr[:, 1:2]*self.darr[:, 1:2]).astype(np.complex_)
        self.nn = (self.darr[:, 2:3]*self.darr[:, 2:3]).astype(np.complex_)
        self.lm = (self.darr[:, 1:2]*self.darr[:, :1]).astype(np.complex_)
        self.ln = (self.darr[:, 2:3]*self.darr[:, :1]).astype(np.complex_)
        self.mn = (self.darr[:, 2:3]*self.darr[:, 1:2]).astype(np.complex_)
        #self.d4 = (self.a/np.sqrt(2))*d4
        self.Es = Es
        self.Ep = Ep
        self.Ed = Ed
        self.sssig = sssig
        self.spsig = spsig
        self.ppsig = ppsig
        self.pppi = pppi
        self.sdsig = sdsig
        self.pdsig = pdsig
        self.pdpi = pdpi
        self.ddsig = ddsigma
        self.ddpi = ddpi
        self.dddelta = dddelta
        self.k = k
        self.rd = rd
        self.energies = []
        self.px_energies = []
        self.py_energies = []
        self.pz_energies = []
        self.xy_energies = []
        self.yz_energies = []
        self.zx_energies = []
        self.xxyy_energies = []
        self.zr_energies = []
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
        kd7 = np.dot(kv, self.d7) 
        kd8 = np.dot(kv, self.d8) 
        kd9 = np.dot(kv, self.d9) 
        kd10 = np.dot(kv, self.d10) 
        kd11 = np.dot(kv, self.d11) 
        kd12 = np.dot(kv, self.d12) 
        
        
        
        gxxpm = np.array([1, 1, 1, 1,         -1, -1, -1, -1,     -1, -1, -1, -1 ])
        gyypm = np.array([-1, -1, -1, -1,      1, 1, 1, 1,        -1, -1, -1, -1 ])
        gzzpm = np.array([-1, -1, -1, -1,     -1, -1, -1, -1,     1, 1, 1, 1 ])
        
        gxypm = np.array([-1, 1, -1, 1,      1, -1, 1, -1,       -1, 1, 1, -1 ])
        gyxpm = np.array([1, -1, 1, -1,     -1, 1, -1, 1,       1, -1, -1, 1 ])
        
        gxzpm = np.array([-1, -1, 1, 1,         -1, 1, 1, -1,         1, -1, 1, -1 ])
        gzxpm = np.array([1, 1, -1, -1,         -1, 1, 1, -1,        -1, 1, -1, 1 ])
        
        
        gyzpm = np.array([-1, 1, 1, -1,      -1, -1, 1, 1,       1, 1, -1, -1 ])
        gzypm = np.array([-1, 1, 1, -1,      1, 1, -1, -1,       -1, -1, 1, 1 ])
        
        
        gspxpm = np.array([1, 1, 1, 1,         -1, 1, -1, 1,     -1, 1, -1, 1 ])
        gspypm = np.array([-1, 1, -1, 1,      1, 1, 1, 1,        -1, 1, -1, 1 ])
        gspzpm = np.array([-1, 1, -1, 1,     -1, 1, -1, 1,     -1, 1, -1, 1 ])
        
        
        ########################## d bands #####################################
        
        
        gxy_xy = np.array([-1, -1, -1, -1,         -1, -1, -1, -1,     -1, -1, -1, -1 ])
        gxy_yz = np.array([1, -1, -1, 1,         -1, 1, 1, -1,     1, -1, -1, 1 ])
        gxy_zx = np.array([-1, 1, 1, -1,         1, -1, -1, 1,     1, -1, -1, 1 ])
        
        gxy_xxyy = np.array([1, 1, 1, 1,         1, 1, 1, 1,     -1, 1, 1, -1 ])
        
        gyz_zx = np.array([1, -1, -1, 1,         1, -1, -1, 1,     -1, 1, 1, -1 ])
        gyz_xxyy = np.array([-1, 1, 1, -1,         1, 1, 1, 1,     1, 1, 1, 1 ])
        gzx_xxyy = np.array([-1, -1, -1, -1,         1, -1, -1, 1,     -1, -1, -1, -1 ])
        
        gxxyy_xxyy = np.array([1, 1, 1, 1,         1, 1, 1, 1,     -1, -1, -1, -1 ])
        gzr_zr = np.array([1, 1, 1, 1,         1, 1, 1, 1,     -1, -1, -1, -1 ])
        
        gzx_zr = np.array([1, 1, 1, 1,         1, -1, -1, 1,     -1, -1, -1, -1 ])
        gxy_zr = np.array([1, 1, 1, 1,         1, 1, 1, 1,     -1, 1, 1, -1 ])
        gyz_zr = np.array([1, -1, 1, -1,         1, 1, 1, 1,     -1, -1, -1, -1 ])
        gxxyy_zr = np.array([-1, -1, -1, -1,         1, 1, 1, 1,     -1, -1, -1, -1 ])
        
        #Phase Factors for the Bloch Sum 
        self.g0_arr= np.array([np.exp(complex(0,kd1)),  #phase terms
                         np.exp(complex(0,kd4)), 
                         np.exp(complex(0,kd5)), 
                         np.exp(complex(0,kd6)), 
                         np.exp(complex(0,kd2)), 
                         np.exp(complex(0,kd7)), 
                         np.exp(complex(0,kd8)), 
                         np.exp(complex(0,kd9)), 
                         np.exp(complex(0,kd3)), 
                         np.exp(complex(0,kd10)), 
                         np.exp(complex(0,kd11)), 
                         np.exp(complex(0,kd12))
                         ])
    
        self.g0c_arr= np.array([np.exp(-complex(0,kd1)),  #phase terms
                         np.exp(-complex(0,kd4)), 
                         np.exp(-complex(0,kd5)), 
                         np.exp(-complex(0,kd6)), 
                         np.exp(-complex(0,kd2)), 
                         np.exp(-complex(0,kd7)), 
                         np.exp(-complex(0,kd8)), 
                         np.exp(-complex(0,kd9)), 
                         np.exp(-complex(0,kd3)), 
                         np.exp(-complex(0,kd10)), 
                         np.exp(-complex(0,kd11)), 
                         np.exp(-complex(0,kd12))
                         ])
    
        self.gxy_arr=self.g0_arr*gxypm
        self.gyx_arr=self.g0_arr*gyxpm
        
        self.gxz_arr=self.g0_arr*gxzpm
        self.gzx_arr=self.g0_arr*gzxpm
        
        self.gyz_arr=self.g0_arr*gyzpm
        self.gzy_arr=self.g0_arr*gzypm
        
        
        self.gxyc_arr=self.g0c_arr*gxypm
        self.gyxc_arr=self.g0c_arr*gyxpm
        
        self.gxzc_arr=self.g0c_arr*gxzpm
        self.gzxc_arr=self.g0c_arr*gzxpm
        
        self.gyzc_arr=self.g0c_arr*gyzpm
        self.gzyc_arr=self.g0c_arr*gzypm
        
        
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
        
        
        
        
        

    
    def initial_energies(self, signifier, band):
        l = self.l
        m = self.m 
        n = self.n 
        ll = self.ll
        mm = self. mm
        nn = self.nn 
        
        
        Kp = (7.62/(self.a**2)) #From eqn V_{ll'm} = n_{ll'm}*h**2/m*d**2
        Kd = (7.62*((self.rd**3)/(self.a**5))**2)
        ones = np.ones((12,1))
                
        if signifier == 'fcc':
            
            if band ==  'p':
            
                const = (7.62/(1.61**2)) #From eqn V_{ll'm} = n_{ll'm}*h**2/m*d**2
                ones = np.ones((12,1))
                
                self.Es = self.sssig*const*ones
         
                self.Espx = self.darr[:, : 1]*self.spsig*const
                self.Espy = self.darr[:, 1: 2]*self.spsig*const
                self.Espz = self.darr[:, 2: 3]*self.spsig*const        
                
                
                self.Epxx = (self.ll*self.ppsig + (ones - self.ll)*self.pppi)*const
                self.Epyy = (self.mm*self.ppsig + (ones - self.mm)*self.pppi)*const
                self.Epzz = (self.nn*self.ppsig + (ones - self.nn)*self.pppi)*const
                
                self.Exy = self.lm*(self.ppsig - self.pppi)*const
                self.Exz = self.ln*(self.ppsig - self.pppi)*const
                self.Eyz = self.mn*(self.ppsig - self.pppi)*const
                
            if band == 'd':
                
                self.Edxy_xy = ( 3*ll*mm*self.ddsig + (ll + mm -4*ll*mm)*self.ddpi + (nn + ll*mm)*self.dddelta )*Kd
                self.Edyz_yz = (3*mm*nn*self.ddsig + (mm + nn -4*mm*nn)*self.ddpi + (ll + mm*nn)*self.dddelta)*Kd
                self.Edzx_zx = (3*nn*ll*self.ddsig + (nn + ll -4*nn*ll)*self.ddpi + (mm + nn*ll)*self.dddelta)*Kd
                
                self.Edxy_yz = (3*l*mm*n*self.ddsig + l*n*(ones -  4*mm )*self.ddpi + l*n*(mm - ones)*self.dddelta)*Kd
                self.Edxy_zx = (3*ll*m*n*self.ddsig + m*n*(ones -  4*ll )*self.ddpi + m*n*(ll - ones)*self.dddelta)*Kd
                self.Edyz_zx = (3*nn*m*l*self.ddsig + m*n*(ones -  4*nn )*self.ddpi + m*l*(nn - ones)*self.dddelta)*Kd
                
                self.Edxy_xxyy = ((3./2)*l*m*(ll - mm)*self.ddsig - 2*l*m*(mm -  ll )*self.ddpi + 0.5*l*m*(ll - mm)*self.dddelta)*Kd
                self.Edyz_xxyy = ((3./2)*m*n*(ll - mm)*self.ddsig - m*n*(ones + 2*(ll -  mm) )*self.ddpi + m*n*(ones + 0.5*(ll - mm))*self.dddelta)*Kd
                self.Edzx_xxyy = ((3./2)*n*l*(ll - mm)*self.ddsig + n*l*(ones - 2*(ll -  mm) )*self.ddpi + n*l*(ones - 0.5*(ll - mm))*self.dddelta)*Kd
                
                self.Edxy_zr = (np.sqrt(3)*l*m*(nn - 0.5*(ll + mm))*self.ddsig - np.sqrt(3)*2*l*m*nn*self.ddpi + 0.5*np.sqrt(3)*l*m*(ones + nn)*self.dddelta)*Kd
                self.Edyz_zr = (np.sqrt(3)*m*n*(nn - 0.5*(ll + mm))*self.ddsig + np.sqrt(3)*m*n*(ll + mm - nn)*self.ddpi - 0.5*np.sqrt(3)*m*n*(ll + nn)*self.dddelta)*Kd
                self.Edzx_zr = (np.sqrt(3)*l*n*(nn - 0.5*(ll + mm))*self.ddsig + np.sqrt(3)*l*n*(ll + mm - nn)*self.ddpi - 0.5*np.sqrt(3)*l*n*(ll + nn)*self.dddelta)*Kd
                
                self.Edxxyy_xxyy = (0.75*((ll - mm)**2)*self.ddsig - (ll + mm - (ll -  mm)**2 )*self.ddpi + (nn + 0.25*(ll -  mm)**2)*self.dddelta)*Kd
                self.Edxxyy_zr = ((3./2)*np.sqrt(3)*(ll - mm)*(nn - 0.5*(ll + mm))*self.ddsig - np.sqrt(3)*nn*(mm - ll)*self.ddpi + 0.25*np.sqrt(3)*(ones + nn)*(ll -  mm)*self.dddelta)*Kd
                self.Edzr_zr = ((nn - 0.5*(ll + mm))*self.ddsig + 3*nn*(mm + ll)*self.ddpi + 0.75*(ll +  mm)*(ll +  mm)*self.dddelta)*Kd
                
                
            
            
            
            
            
        
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
    
    
        M = np.asarray([
                [np.dot(self.gdxy_xy_arr,self.Edxy_xy)[0],  np.dot(self.gdxy_yz_arr,self.Edxy_yz)[0], 
                         np.dot(self.gdxy_zx_arr,self.Edxy_zx)[0],  np.dot(self.gdxy_xxyy_arr,self.Edxy_xxyy)[0],  np.dot(self.gdxy_zr_arr,self.Edxy_zr)[0]  ], 
                
                [np.dot(self.gdyz_xy_arr,self.Edxy_yz)[0],  np.dot(self.gdyz_yz_arr,self.Edyz_yz)[0], 
                         np.dot(self.gdyz_zx_arr,self.Edyz_zx)[0],  np.dot(self.gdyz_xxyy_arr,self.Edyz_xxyy)[0],  np.dot(self.gdyz_zr_arr,self.Edyz_zr)[0] ], 
                
                [np.dot(self.gdzx_xy_arr,self.Edxy_zx)[0],  np.dot(self.gdzx_yz_arr,self.Edyz_zx)[0], 
                         np.dot(self.gdzx_zx_arr,self.Edzx_zx)[0],  np.dot(self.gdzx_xxyy_arr,self.Edzx_xxyy)[0],  np.dot(self.gdzx_zr_arr,self.Edzx_zr)[0] ], 
                           
                [np.dot(self.gdxxyy_xy_arr,self.Edxy_xxyy)[0],  np.dot(self.gdxxyy_yz_arr,self.Edyz_xxyy)[0], 
                         np.dot(self.gdxxyy_zx_arr,self.Edzx_xxyy)[0],  np.dot(self.gdxxyy_xxyy_arr,self.Edxxyy_xxyy)[0],  np.dot(self.gdxxyy_zr_arr,self.Edxxyy_zr)[0] ], 
                 
                [np.dot(self.gdzr_xy_arr,self.Edxy_zr)[0],  np.dot(self.gdzr_yz_arr,self.Edyz_zr)[0], 
                         np.dot(self.gdzr_zx_arr,self.Edzx_zr)[0],  np.dot(self.gdzr_xxyy_arr,self.Edxxyy_zr)[0],  np.dot(self.gdzr_zr_arr,self.Edzr_zr)[0] ], 
                 
                 
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
    
        M = np.asarray([[np.dot(self.g0_arr, self.Es)[0], np.dot(self.gspx_arr,self.Espx)[0],  np.dot(self.gspy_arr,self.Espy)[0],  np.dot(self.gspz_arr,self.Espz)[0] ]
                [np.dot(self.gspx_arr,self.Espx)[0], np.dot(self.g0_arr,self.Epxx)[0],  np.dot(self.gxy_arr,self.Exy)[0],  np.dot(self.gxz_arr,self.Exz)[0] ], 
                [np.dot(self.gspy_arr,self.Espy)[0], np.dot(self.gyx_arr,self.Exy)[0],  np.dot(self.g0_arr,self.Epyy)[0],  np.dot(self.gyz_arr,self.Eyz)[0] ], 
                [np.dot(self.gspz_arr,self.Espz)[0], np.dot(self.gzx_arr,self.Exz)[0],  np.dot(self.gzy_arr,self.Eyz)[0],  np.dot(self.g0_arr,self.Epzz)[0] ]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    
    def Hamiltonian_p(self, kv):
                
        
        self.phasefactors(kv)
    
    
        M = np.asarray([
                [np.dot(-self.g0_arr,self.Epxx)[0],  np.dot(self.gxy_arr,self.Exy)[0],  np.dot(self.gxz_arr,self.Exz)[0] ], 
                [np.dot(self.gyx_arr,self.Exy)[0],  np.dot(-self.g0_arr,self.Epyy)[0],  np.dot(self.gyz_arr,self.Eyz)[0] ], 
                [np.dot(self.gzx_arr,self.Exz)[0],  np.dot(self.gzy_arr,self.Eyz)[0],  np.dot(-self.g0_arr,self.Epzz)[0] ]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    
    def Hamiltonian_s(self):
        
        """
        #Form of the Hamiltonian Matrix in terms of orbitals
        
               s        
        s   sssig*g0  

        """
        
        self.initial_energies('fcc', 'p')
        self.phasefactors(self.k)
    
    
        M = np.array(
                [self.Es*self.g0]
                    )
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
        self.energies = []
        self.initial_energies('fcc', 'p')
        
        
        loops = 200
        for i in range(loops):
            kr = ki + (i/loops)*k_diff
            self.phasefactors(kr )
            self.energies.append((1/12)*np.dot(self.g0_arr, self.Es)[0])
            self.kvals.append(i/float(loops))
            

        ax.plot(self.kvals, self.energies)
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
        k_diff = kf-ki
        self.initial_energies('fcc', 'p')
        loops = 200
        for i in range(loops):
            kr = ki + (i/float(loops))*k_diff
            #self.phasefactors(kr )
            self.M = (1/12.)*self.Hamiltonian_p(kr)
            eigenvals = self.eigenvalues(self.M)
            print(eigenvals)
            self.px_energies.append(eigenvals[0])
            self.py_energies.append(eigenvals[1])
            self.pz_energies.append(eigenvals[2])
            self.kvals.append((i/float(loops)))

        ax.plot(self.kvals, self.px_energies)#, marker=style, color=k)
        ax.plot(self.kvals, self.py_energies)#, bo)
        ax.plot(self.kvals, self.pz_energies)#, r*)
        
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
        k_diff = kf-ki
        self.initial_energies('fcc', 'd')
        loops = 200
        for i in range(loops):
            kr = ki + (i/float(loops))*k_diff
            #self.phasefactors(kr )
            self.M = (1/4.)*self.Hamiltonian_d(kr)
            eigenvals = self.eigenvalues(self.M)
            print(eigenvals)
            self.xy_energies.append(eigenvals[0])
            self.yz_energies.append(eigenvals[1])
            self.zx_energies.append(eigenvals[2])
            self.xxyy_energies.append(eigenvals[3])
            self.zr_energies.append(eigenvals[4])
            self.kvals.append((i/float(loops)))

        ax.plot(self.kvals, self.xy_energies)#, marker=style, color=k)
        ax.plot(self.kvals, self.yz_energies)#, bo)
        ax.plot(self.kvals, self.zx_energies)#, r*)
        ax.plot(self.kvals, self.xxyy_energies)#, r*)
        ax.plot(self.kvals, self.zr_energies)#, r*)
        
        ax.set_title('%s to %s'%(n1, n2)) 
        if reverse==True:
            ax.set_xlim([np.max(self.kvals), np.min(self.kvals)])
                
                
#def band_structure_plotting_fcc(con):
    
    
def sband_script(con):

    X = (2*np.pi/con.a)*np.array([0,1,0])
    L = (np.pi/con.a)*np.array([1,1,1])
    W = (2*np.pi/con.a)*np.array([0.5,1,0])
    K = (2*np.pi/con.a)*np.array([0.25,1,0.25])
    U = (np.pi/con.a)*np.array([1.5,1.5,0])
    Gamma = np.array([0,0,0])
    
    fig, axes = plt.subplots(1,6,sharey='all')

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax6 = axes[5]
    
    con.band_structure_s(Gamma, X, ax1, False, 'Gamma', 'X')
    con.band_structure_s(X, W, ax2, False, 'X', 'W')
    con.band_structure_s(W, L, ax3, False, 'W', 'L')
    con.band_structure_s(L, Gamma, ax4, False, 'L', 'Gamma')
    con.band_structure_s(Gamma, K, ax5, False, 'Gamma', 'K')
    con.band_structure_s(U, X, ax6, False, 'U', 'X')
    
    ax1.set_ylabel('E (eV)')
    ax3.set_xlabel('¦K¦')
    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)

    plt.suptitle('s-bands:fcc')
    plt.show()  



def pband_script(con):
    
    
    X = (2*np.pi/con.a)*np.array([0,1,0])
    L = (np.pi/con.a)*np.array([1,1,1])
    W = (2*np.pi/con.a)*np.array([0.5,1,0])
    K = (2*np.pi/con.a)*np.array([0.25,1,0.25])
    U = (np.pi/con.a)*np.array([1.5,1.5,0])
    Gamma = np.array([0,0,0])


    fig, axes = plt.subplots(1,6,sharey='all')

    ax1 = axes[0] 
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax6 = axes[5]

    con.band_structure_p(Gamma, X, ax1, False, 'Gamma', 'X')
    con.band_structure_p(X, W, ax2, False, 'X', 'W')
    con.band_structure_p(W, L, ax3, False, 'W', 'L')
    con.band_structure_p(L, Gamma, ax4, False, 'L', 'Gamma')
    con.band_structure_p(Gamma, K, ax5, False, 'Gamma', 'K')
    con.band_structure_p(U, X, ax6, False, 'U', 'X')
    
    ax3.set_xlabel('¦K¦')
    ax1.set_ylabel('E (eV)')
    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)
    plt.suptitle('p-bands:fcc')
    plt.show()  
    

def dband_script(con):
    
    
    X = (2*np.pi/con.a)*np.array([0,1,0])
    L = (np.pi/con.a)*np.array([1,1,1])
    W = (2*np.pi/con.a)*np.array([0.5,1,0])
    K = (2*np.pi/con.a)*np.array([0.25,1,0.25])
    U = (np.pi/con.a)*np.array([1.5,1.5,0])
    Gamma = np.array([0,0,0])


    fig, axes = plt.subplots(1,6,sharey='all')

    ax1 = axes[0] 
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    ax5 = axes[4]
    ax6 = axes[5]

    con.band_structure_d(Gamma, X, ax1, False, 'Gamma', 'X')
    con.band_structure_d(X, W, ax2, False, 'X', 'W')
    con.band_structure_d(W, L, ax3, False, 'W', 'L')
    con.band_structure_d(L, Gamma, ax4, False, 'L', 'Gamma')
    con.band_structure_d(Gamma, K, ax5, False, 'Gamma', 'K')
    con.band_structure_d(U, X, ax6, False, 'U', 'X')
    
    ax3.set_xlabel('¦K¦')
    ax1.set_ylabel('E (eV)')
    fig.subplots_adjust(wspace=0)
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)
    plt.suptitle('d-bands:fcc')
    plt.show()  
    

con = Hconstruct() 
dband_script(con) 



