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
    
    def __init__(self, k=np.array([0,0,1]), H = np.ones(5), a = 1./4, 
                 d1=np.array([1,1,1]), d2=np.array([1,-1,-1]), 
                 d3=np.array([-1,1,-1]), d4=np.array([-1,-1,1]), Es=1, Ep=1, 
                 Ed=1, sssig=-1.40, spsig=1.84, ppsig=3.24, pppi=-0.81, sdsig=1, pdsig=1, #sssig to pppi from Harrison and general, d from Titanium solid state table
                 pdpi=1, ddsig=-11.04, ddpi=1, dddel=1 ):

    
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
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
        self.kvals = []
        
        
    def phasefactors(self, kv):
        
        #dot products of k vectors with distance vectors of unit cell 
        kd1 = np.dot(kv, self.d1) 
        kd2 = np.dot(kv, self.d2) 
        kd3 = np.dot(kv, self.d3) 
        kd4 = np.dot(kv, self.d4) 
        
        
        #Phase Factors for the Bloch Sum 
        b1 = np.exp(complex(0,kd1)) #phase terms
        b2 = np.exp(complex(0,kd2)) 
        b3 = np.exp(complex(0,kd3)) 
        b4 = np.exp(complex(0,kd4)) 
        
        c1 = np.exp(complex(0,kd1)) # conjugate phase terms
        c2 = np.exp(complex(0,kd2)) 
        c3 = np.exp(complex(0,kd3)) 
        c4 = np.exp(complex(0,kd4))
        
        self.g0 = b1 + b2 + b3 + b4 #Actual phase factors
        self.g1 = b1 + b2 - b3 - b4
        self.g2 = b1 - b2 + b3 - b4
        self.g3 = b1 - b2 - b3 + b4
        
        self.g0c = c1 + c2 + c3 + c4 #Actual phase factors
        self.g1c = c1 + c2 - c3 - c4
        self.g2c = c1 - c2 + c3 - c4
        self.g3c = c1 - c2 - c3 + c4
        
        #return g0, g1, g2, g3
    
    def initial_energies(self, signifier):
        
        if signifier == 'fcc':
            
            self.Es = self.sssig
            self.Ep = (1/3)*(self.ppsig + 2*self.pppi)
            self.Esp = -(1/np.sqrt(3))*self.spsig
            self.Exy = (1/3)*(self.ppsig - self.pppi)
            
            
        
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
                [self.Esp*self.g1c,  self.Ep*self.g1,  self.Exy*self.g3,  self.Esp*self.g3 ], 
                [self.Esp*self.g2c,  self.Esp*self.g1,  self.Ep*self.g2,  self.Esp*self.g3 ], 
                [self.Esp*self.g3c,  self.Esp*self.g1,  self.Esp*self.g2,  self.Ep*self.g3 ]
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
    
    def band_structure_s(self, ki, kf, M):
        """ kf and ki must be 1x3 arrays
        """
        self.Hamiltonian_s()
        #self.energies = []
        k_diff = kf-ki
        self.energies.append(M[0])
        self.kvals.append(np.sqrt(ki[0]**2 + ki[1]**2 + ki[2]**2))
        for i in range(20):
            kr = ki + (i/20)*k_diff
            self.phasefactors(kr )
            self.energies.append(self.Es*self.g0)
            self.kvals.append(np.sqrt(kr[0]**2 + kr[1]**2 + kr[2]**2) )
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.kvals, self.energies)
        ax.set_xlabel('k')
        ax.set_ylabel('E')
        ax.set_title('s-band: fcc')
        plt.show()
            
            
                
                
            
            
    
    
con = Hconstruct()
M = con.Hamiltonian_s()
ki = np.array([0,0,0])
kf = np.array([0,2*np.pi,0])
con.band_structure_s(ki, kf, M )
#con.phasefactors()
#H = con.Hamiltonian()
#eigenvals = con.eigenvalues(H)


#print("eigenvals")
#print(eigenvals)

