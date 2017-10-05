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
    
    def __init__(self, k=np.array([0,0,0]), H = np.ones(5), a = 1.,#1.61, #a is in Angstrom
                 d1=np.array([0,1,1]), d2=np.array([1,0,1]), 
                 d3=np.array([1,1,0]), d4=np.array([0,-1,1]),
                 d5=np.array([0,1,-1]), d6=np.array([0,-1,-1]),
                 d7=np.array([-1,0,1]), d8=np.array([1,0,-1]), d9=np.array([-1,0,-1]), 
                d10=np.array([-1,1,0]), d11=np.array([1,-1,0]), d12=np.array([-1,-1,0]), 
                Es=1, Ep=1, 
                 Ed=1, sssig=-1.40, spsig=1.84, ppsig=3.24, pppi=-0.81, sdsig=1, pdsig=1, #sssig to pppi from Harrison and general, d from Titanium solid state table
                 pdpi=1, ddsig=-11.04, ddpi=1, dddel=1 ):

        self.a = a
        self.d1 = (self.a/2)*d1
        self.d2 = (self.a/2)*d2
        self.d3 = (self.a/2)*d3
        self.d4 = (self.a/2)*d4
        self.d5 = (self.a/2)*d5
        self.d6 = (self.a/2)*d6
        self.d7 = (self.a/2)*d7
        self.d8 = (self.a/2)*d8
        self.d9 = (self.a/2)*d9
        self.d10 = (self.a/2)*d10
        self.d11 = (self.a/2)*d11
        self.d12 = (self.a/2)*d12
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
        self.H = H
        self.k = k
        self.energies = []
        self.px_energies = []
        self.py_energies = []
        self.pz_energies = []
        self.kvals = []
        
        
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
        
        
        #kd4 = np.dot(kv, self.d4) 
        
        
        #Phase Factors for the Bloch Sum 
        b1 = np.exp(complex(0,kd1)) #phase terms
        b2 = np.exp(complex(0,kd2)) 
        b3 = np.exp(complex(0,kd3)) 
        b4 = np.exp(complex(0,kd4)) #phase terms
        b5 = np.exp(complex(0,kd5)) 
        b6 = np.exp(complex(0,kd6)) 
        b7 = np.exp(complex(0,kd7)) #phase terms
        b8 = np.exp(complex(0,kd8)) 
        b9 = np.exp(complex(0,kd9)) 
        b10 = np.exp(complex(0,kd10)) #phase terms
        b11 = np.exp(complex(0,kd11)) 
        b12 = np.exp(complex(0,kd12)) 
        
        self.b1 = np.exp(complex(0,kd1)) #phase terms
        self.b2 = np.exp(complex(0,kd2)) 
        self.b3 = np.exp(complex(0,kd3)) 
        self.b4 = np.exp(complex(0,kd4)) #phase terms
        self.b5 = np.exp(complex(0,kd5)) 
        self.b6 = np.exp(complex(0,kd6)) 
        self.b7 = np.exp(complex(0,kd7)) #phase terms
        self.b8 = np.exp(complex(0,kd8)) 
        self.b9 = np.exp(complex(0,kd9)) 
        self.b10 = np.exp(complex(0,kd10)) #phase terms
        self.b11 = np.exp(complex(0,kd11)) 
        self.b12 = np.exp(complex(0,kd12)) 
        #b4 = np.exp(complex(0,kd4)) 
        
        self.c1 = np.exp(-complex(0,kd1)) #phase terms
        self.c2 = np.exp(-complex(0,kd2)) 
        self.c3 = np.exp(-complex(0,kd3)) 
        self.c4 = np.exp(-complex(0,kd4)) #phase terms
        self.c5 = np.exp(-complex(0,kd5)) 
        self.c6 = np.exp(-complex(0,kd6)) 
        self.c7 = np.exp(-complex(0,kd7)) #phase terms
        self.c8 = np.exp(-complex(0,kd8)) 
        self.c9 = np.exp(-complex(0,kd9)) 
        self.c10 = np.exp(-complex(0,kd10)) #phase terms
        self.c11 = np.exp(-complex(0,kd11)) 
        self.c12 = np.exp(-complex(0,kd12)) 
        
        c1 = np.exp(-complex(0,kd1)) #phase terms
        c2 = np.exp(-complex(0,kd2)) 
        c3 = np.exp(-complex(0,kd3)) 
        c4 = np.exp(-complex(0,kd4)) #phase terms
        c5 = np.exp(-complex(0,kd5)) 
        c6 = np.exp(-complex(0,kd6)) 
        c7 = np.exp(-complex(0,kd7)) #phase terms
        c8 = np.exp(-complex(0,kd8)) 
        c9 = np.exp(-complex(0,kd9)) 
        c10 = np.exp(-complex(0,kd10)) #phase terms
        c11 = np.exp(-complex(0,kd11)) 
        c12 = np.exp(-complex(0,kd12))
        #c4 = np.exp(complex(0,kd4))
        
        self.g0 = b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12   #+ #b4 #Actual phase factors
        self.gxx = 0*b1 + b2 + b3 + 0*b4 + 0*b5 + 0*b6 + b7 + b8 + b9 + b10 + b11 + b12#- b4
        self.gzz = b1 + b2 + 0*b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12#- b4
        self.gyy = b1 + 0*b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12#+ b4
        
        self.gxy = 0*b1 + 0*b2 + b3 + 0*b4 + 0*b5 + 0*b6 + 0*b7 + 0*b8 + 0*b9 - b10 - b11 + b12#- b4
        self.gxz = 0*b1 + b2 + 0*b3 + 0*b4 + 0*b5 + 0*b6 - b7 - b8 + b9 + 0*b10 + 0*b11 + 0*b12#- b4
        self.gyz = b1 + 0*b2 + 0*b3 - b4 - b5 + b6 + 0*b7 + 0*b8 + 0*b9 + 0*b10 + 0*b11 + 0*b12#+ b4
        
        self.gxyc = 0*c1 + 0*c2 + c3 + 0*c4 + 0*c5 + 0*c6 + 0*c7 + 0*c8 + 0*c9 - c10 - c11 + c12#- b4#- b4
        self.gxzc = 0*c1 + c2 + 0*c3 + 0*c4 + 0*c5 + 0*c6 - c7 - c8 + c9 + 0*c10 + 0*c11 + 0*c12#- b4#- b4
        self.gyzc = c1 + 0*c2 + 0*c3 - c4 - c5 + c6 + 0*c7 + 0*c8 + 0*c9 + 0*c10 + 0*c11 + 0*c12#+ b4#+ b4
        
        self.g0c = c1 + c2 + c3 #+ c4 #Actual phase factors
        self.gxxc = 0*c1 + c2 + c3 #- c4
        self.gzzc = c1 + c2 + 0*c3 #- c4
        self.gyyc = c1 + 0*c2 + c3 #+ c4
        
        #return g0, g1, g2, g3
    
    def initial_energies(self, signifier):
        
        if signifier == 'fcc':
            
            const = (7.62/(1.61**2)) #From eqn V_{ll'm} = n_{ll'm}*h**2/m*d**2
            self.Es = self.sssig*const
            self.Ep = (1/2)*(self.ppsig + self.pppi)*const
            self.Epionly = self.pppi*const
            self.Esp = -(1/np.sqrt(2))*self.spsig*const
            self.Exy = (1/2)*(self.ppsig - self.pppi)*const
            
            
        
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
    
        self.Epxx = self.Ep*self.gxx + self.Epionly*(self.b1 + self.b4 + self.b5 + self.b6)
        self.Epyy = self.Ep*self.gyy + self.Epionly*(self.b2 + self.b7 + self.b8 + self.b9)
        self.Epzz = self.Ep*self.gzz + self.Epionly*(self.b3 + self.b10 + self.b11 + self.b12)
        M = np.array([
                [self.Epxx,  self.Exy*self.gxy,  self.Exy*self.gxz ], 
                [self.Exy*self.gxy,  self.Epyy,  self.Exy*self.gyz ], 
                [self.Exy*self.gxz,  self.Exy*self.gyz,  self.Epzz ]
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
        #self.Hamiltonian_s()
        #self.energies = []
        k_diff = kf-ki
        self.kvals = []
        self.energies = []
        
        #self.energies.append(M[0])
        #self.kvals.append(np.sqrt(ki[0]**2 + ki[1]**2 + ki[2]**2))
        loops = 150
        for i in range(loops):
            kr = ki + (i/loops)*k_diff
            self.phasefactors(kr )
            self.energies.append(-self.Es*self.g0)
            #self.kvals.append(np.sqrt(kr[0]**2 + kr[1]**2 + kr[2]**2) )
            self.kvals.append(i/float(loops))
            
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        ax.plot(self.kvals, self.energies)
        #ax.set_xlabel('k')
        #ax.set_ylabel('E')
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
            self.M = self.Hamiltonian_p(kr)
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
    L = (2*np.pi/con.a)*np.array([1,1,1])
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
    
    con.band_structure_s(L, Gamma, ax4, True, 'L', 'Gamma')
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
    L = (2*np.pi/con.a)*np.array([1,1,1])
    W = (2*np.pi/con.a)*np.array([0.5,1,0])
    K = (2*np.pi/con.a)*np.array([0.25,1,0.25])
    U = (np.pi/con.a)*np.array([1.5,1.5,0])
    Gamma = np.array([0,0,0])

    ki = Gamma#(2*np.pi/con.a)*np.array([0,0,0])
    kf = X #(2*np.pi/con.a)*np.array([0,1,0])
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
    con.band_structure_p(L, Gamma, ax4, True, 'L', 'Gamma')
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

