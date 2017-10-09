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


#####    FCC   #####


class Hconstruct:
    
    def __init__(self, k=np.array([0,0,0]), a = 1.,#1.61, #a is in Angstrom
                 d1=np.array([0,1,1]), d2=np.array([1,0,1]), 
                 d3=np.array([1,1,0]), d4=np.array([0,-1,1]),
                 d5=np.array([0,1,-1]), d6=np.array([0,-1,-1]),
                 d7=np.array([-1,0,1]), d8=np.array([1,0,-1]), d9=np.array([-1,0,-1]), 
                d10=np.array([-1,1,0]), d11=np.array([1,-1,0]), d12=np.array([-1,-1,0]), 
                Es=1, Ep=1, 
                 Ed=1, sssig=-1.39, spsig=1.84, ppsig=3.24, pppi=-0.93, sdsig=1, pdsig=1, #sssig to pppi from Harrison and general, d from Titanium solid state table
                 pdpi=1, ddsig=-11.04, ddpi=1, dddel=1 ):

        self.a = a
        self.d1 = (self.a/4)*d1*2
        self.d2 = (self.a/4)*d2*2
        self.d3 = (self.a/4)*d3*2
        self.d4 = (self.a/4)*d4*2
        self.d5 = (self.a/4)*d5*2
        self.d6 = (self.a/4)*d6*2
        self.d7 = (self.a/4)*d7*2
        self.d8 = (self.a/4)*d8*2
        self.d9 = (self.a/4)*d9*2
        self.d10 = (self.a/4)*d10*2
        self.d11 = (self.a/4)*d11*2
        self.d12 = (self.a/4)*d12*2
        
        self.darr = np.asarray([self.d1,self.d4,self.d5,self.d6,self.d2,self.d7,
                           self.d8,self.d9,self.d3,self.d10, self.d11, self.d12])
        self.l = (1/np.sqrt(2))*self.darr[:, :1].astype(np.complex_)
        self.ll = (1/2.)*(self.darr[:, :1]*self.darr[:, :1]).astype(np.complex_)
        self.mm = (1/2.)*(self.darr[:, 1:2]*self.darr[:, 1:2]).astype(np.complex_)
        self.nn = (1/2.)*(self.darr[:, 2:3]*self.darr[:, 2:3]).astype(np.complex_)
        self.lm = (1/2.)*(self.darr[:, 1:2]*self.darr[:, :1]).astype(np.complex_)
        self.ln = (1/2.)*(self.darr[:, 2:3]*self.darr[:, :1]).astype(np.complex_)
        self.mn = (1/2.)*(self.darr[:, 2:3]*self.darr[:, 1:2]).astype(np.complex_)
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
        self.ddsig = ddsig
        self.ddpi = ddpi
        self.dddel = dddel
        self.k = k
        self.energies = []
        self.px_energies = []
        self.py_energies = []
        self.pz_energies = []
        self.kvals = []
        
#
    
    
        
    def phasefactors(self, kv):
        
        
        #dot products of k vectors with distance vectors of unit cell 
        kd1 = np.dot(kv, self.d1) 
        kd2 = np.dot(kv, self.d2) 
        kd3 = np.dot(kv, self.d3)
        kd4= np.dot(kv, self.d4) 
        kd5 = np.dot(kv, self.d5) 
        kd6 = np.dot(kv, self.d6) 
        kd7 = np.dot(kv, self.d7) 
        kd8 = np.dot(kv, self.d8) 
        kd9 = np.dot(kv, self.d9) 
        kd10 = np.dot(kv, self.d10) 
        kd11 = np.dot(kv, self.d11) 
        kd12 = np.dot(kv, self.d12) 
        
        
        
        gxxpm = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1 ])
        gyypm = np.array([-1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1 ])
        gzzpm = np.array([-1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1 ])
        
        gxypm = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1 ])
        gyxpm = np.array([1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1 ])
        
        gxzpm = np.array([-1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1 ])
        gzxpm = np.array([1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1 ])
        
        
        gyzpm = np.array([-1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1 ])
        gzypm = np.array([-1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1 ])
        
        
        
        
        gxxpm = np.array([1, 1, 1, 1,         -1, -1, -1, -1,     -1, -1, -1, -1 ])
        gyypm = np.array([-1, -1, -1, -1,      1, 1, 1, 1,        -1, -1, -1, -1 ])
        gzzpm = np.array([-1, -1, -1, -1,     -1, -1, -1, -1,     1, 1, 1, 1 ])
        
        gxypm = np.array([-1, 1, -1, 1,      1, -1, 1, -1,       -1, 1, 1, -1 ])
        gyxpm = np.array([1, -1, 1, -1,     -1, 1, -1, 1,       -1, 1, 1, -1 ])
        
        gxzpm = np.array([-1, -1, 1, 1,         -1, 1, 1, -1,         1, -1, 1, -1 ])
        gzxpm = np.array([1, 1, -1, -1,         -1, 1, 1, -1,        -1, 1, -1, 1 ])
        
        
        gyzpm = np.array([-1, 1, 1, -1,      -1, -1, 1, 1,       1, 1, -1, -1 ])
        gzypm = np.array([-1, 1, 1, -1,      1, 1, -1, -1,       -1, -1, 1, 1 ])
        
        
        
        #Phase Factors for the Bloch Sum 
        self.g0_arr=np.array([np.exp(complex(0,kd1)),  #phase terms
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
    
        self.gxy_arr=self.g0_arr*gxypm
        self.gyx_arr=self.g0_arr*gyxpm
        
        self.gxz_arr=self.g0_arr*gxzpm
        self.gzx_arr=self.g0_arr*gzxpm
        
        self.gyz_arr=self.g0_arr*gyzpm
        self.gzy_arr=self.g0_arr*gzypm
        
        self.gxx_arr=self.g0_arr*gxxpm
        self.gyy_arr=self.g0_arr*gyypm
        self.gzz_arr=self.g0_arr*gzzpm
        
        
        #self.g0_arr = (self.g0_arr)
    

    
        self.g0c_arr=np.array([[ np.exp(-complex(0,kd1))],  #phase terms
                         [np.exp(-complex(0,kd2))], 
                         [np.exp(-complex(0,kd3))], 
                         [np.exp(-complex(0,kd4))], 
                         [np.exp(-complex(0,kd5))], 
                         [np.exp(-complex(0,kd6))], 
                         [np.exp(-complex(0,kd7))], 
                         [np.exp(-complex(0,kd8))], 
                         [np.exp(-complex(0,kd9))], 
                         [np.exp(-complex(0,kd10))], 
                         [np.exp(-complex(0,kd11))], 
                         [np.exp(-complex(0,kd12))]
                         ],dtype=np.complex_)
  
  
    
    def initial_energies(self, signifier):
        
        #cosines
        if signifier == 'fcc':
            
            const = (7.62/(1.61**2)) #From eqn V_{ll'm} = n_{ll'm}*h**2/m*d**2
            ones = np.ones((12,1))
            
            self.Es = self.sssig*const*ones
            self.Epxx = (self.ll*self.ppsig + (ones - self.ll)*self.pppi)*const
            self.Epyy = (self.mm*self.ppsig + (ones - self.mm)*self.pppi)*const
            self.Epzz = (self.nn*self.ppsig + (ones - self.nn)*self.pppi)*const
            
            self.Ep
            #self.Epionly = self.pppi*const
            self.Espx = -self.darr[:, : 1]*self.spsig*const
            self.Espy = -self.darr[:, 1: 2]*self.spsig*const
            self.Espz = -self.darr[:, 2: 3]*self.spsig*const
            
            self.Exy = self.lm*(self.ppsig - self.pppi)*const
            self.Exz = self.ln*(self.ppsig - self.pppi)*const
            self.Eyz = self.mn*(self.ppsig - self.pppi)*const
            
            
        
    def Hamiltonian(self):
        
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
        
        
    
    
        M = np.array([[1,2,3,1,1], [4,5,6, 2,2],[7,8,9, 3,3], [7,8,9, 3,3],[7,8,9, 3,3]])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    def Hamiltonian_sp(self):
        
        """
        #Form of the Hamiltonian Matrix in terms of orbitals
        
               s         px         py        pz      
        s   sssig*g0  spsig*g1  spsig*g2  spsig*g3  
        
        px  -spsig*g1  Ep  spsig*g2  spsig*g3  
        
        py  -spsig*g2  spsig*g1  Ep  spsig*g3  
        
        pz  -spsig*g3  spsig*g1  spsig*g2  Ep  
        
        """
        
        self.initial_energies('fcc')
        self.phasefactors()
    
    
        M = np.array([
                [self.Es*self.g0, self.Esp*self.g1, self.Esp*self.g2, self.Esp*self.g3],
                [self.Esp*self.g1c,  self.Ep*self.g0,  self.Exy*self.g3,  self.Exy*self.g2  ], 
                [self.Esp*self.g2c, self.Exy*self.g3c,  self.Ep*self.g0,  self.Exy*self.g1  ], 
                [self.Esp*self.g3c,  self.Exy*self.g2c,  self.Exy*self.g1c,  self.Ep*self.g0 ]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    
    def Hamiltonian_p(self, kv):
                
        self.initial_energies('fcc')
        self.phasefactors(kv)
    
    
        M = np.asarray([
                [np.dot(self.gxx_arr,self.Epxx)[0],  np.dot(self.gxy_arr,self.Exy)[0],  np.dot(self.gxz_arr,self.Exz)[0] ], 
                [np.dot(self.gyx_arr,self.Exy)[0],  np.dot(self.gyy_arr,self.Epyy)[0],  np.dot(self.gyz_arr,self.Eyz)[0] ], 
                [np.dot(self.gzx_arr,self.Exz)[0],  np.dot(self.gzy_arr,self.Eyz)[0],  np.dot(self.gzz_arr,self.Epzz)[0] ]
                    ])
        #Array of Hamiltonian matrix with energy values 
        return M
    
    
    def Hamiltonian_s(self):
        
        """
        #Form of the Hamiltonian Matrix in terms of orbitals
        
               s        
        s   sssig*g0  

        """
        
        self.initial_energies('fcc')
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
        self.initial_energies('fcc')
        
        
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
        #M = self.Hamiltonian_p(ki)
        #self.energies = []
        self.kvals = []
        self.px_energies = []
        self.py_energies = []
        self.pz_energies = []
        k_diff = kf-ki
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
            #self.kvals.append(np.sqrt(kr[0]**2 + kr[1]**2 + kr[2]**2))

            
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        ax.plot(self.kvals, np.real(self.px_energies))#, marker=style, color=k)
        ax.plot(self.kvals, np.real(self.py_energies))#, bo)
        ax.plot(self.kvals, np.real(self.pz_energies))#, r*)
        
        ax.set_title('%s to %s'%(n1, n2)) 
        if reverse==True:
            ax.set_xlim([np.max(self.kvals), np.min(self.kvals)])
                
                
#def band_structure_plotting_fcc(con):
    
    
def sband_script(con):

    #M = con.Hamiltonian_s()
    #print('M', M)
    X = (2*np.pi/con.a)*np.array([0,1,0])
    L = (np.pi/con.a)*np.array([1,1,1])
    W = (2*np.pi/con.a)*np.array([0.5,1,0])
    K = (2*np.pi/con.a)*np.array([0.25,1,0.25])
    U = (np.pi/con.a)*np.array([1.5,1.5,0])
    Gamma = np.array([0,0,0])
    
    ki = L#(2*np.pi/con.a)*np.array([0,0,0])
    kf = Gamma#(2*np.pi/con.a)*np.array([0,1,0])
    fig, axes = plt.subplots(1,6,sharey='all')
    #print(axes)
    ax1 = axes[0] #fig.add_subplot(141)
    ax2 = axes[1]#fig.add_subplot(142)
    ax3 = axes[2]#fig.add_subplot(143)
    ax4 = axes[3]#3]#fig.add_subplot(144)
    ax5 = axes[4]
    ax6 = axes[5]
    #ax5 = fig.add_subplot(165)
    #ax6 = fig.add_subplot(166)
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
    #xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels() 
    #plt.setp(xticklabels, visible=False)
    plt.suptitle('s-bands:fcc')
    plt.show()  
    #con.band_structure_s(ki, kf, M)
    
def pband_script(con):
    
    
    X = (2*np.pi/con.a)*np.array([0,1,0])
    L = (np.pi/con.a)*np.array([1,1,1])
    W = (2*np.pi/con.a)*np.array([0.5,1,0])
    K = (2*np.pi/con.a)*np.array([0.25,1,0.25])
    U = (np.pi/con.a)*np.array([1.5,1.5,0])
    Gamma = np.array([0,0,0])

    #ki = Gamma#(2*np.pi/con.a)*np.array([0,0,0])
    #kf = X #(2*np.pi/con.a)*np.array([0,1,0])
    #M = con.Hamiltonian_p(ki)
    #print('M', M)
    #fig = plt.figure()
    fig, axes = plt.subplots(1,6,sharey='all')
    print(axes)
    ax1 = axes[0] #fig.add_subplot(141)
    ax2 = axes[1]#fig.add_subplot(142)
    ax3 = axes[2]#fig.add_subplot(143)
    ax4 = axes[3]#3]#fig.add_subplot(144)
    ax5 = axes[4]
    ax6 = axes[5]
    #ax5 = fig.add_subplot(165)
    #ax6 = fig.add_subplot(166)
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
    

con = Hconstruct() 
pband_script(con) 

#con.phasefactors()
#H = con.Hamiltonian()
#eigenvals = con.eigenvalues(H)


#print("eigenvals")
#print(eigenvals)

