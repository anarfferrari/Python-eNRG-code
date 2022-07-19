import numpy as np
from scipy.linalg import block_diag 
from operator import itemgetter 
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import pandas as pd
import invariants_loop as inv
plt.style.use(["seaborn"])

def u_v_vg_w():
    """
    Receives: 
    Performs: Reads the initial parameters from the file "initial_parameters.txt".
    Rerturns: An array [U, V, Vg, W].
    """
    return np.loadtxt("initial_parameters.txt", comments = "#")

class Subspace:
    def __init__(self, init_eigblock, invariants, init_gender):
        self.eigblock = init_eigblock
        self.invariants = invariants
        self.gender = np.array(init_gender)


def initial_Hamiltonian():
    """
    Receives:
    Perfoms: Defines the initial Hamiltonian. 
    Returns: The entire initial Hamiltonian and its blocks in an array.
    """
    
    U, V, Vg, w = u_v_vg_w()
    H_block = []
    
    H_block.append(np.array([[0]]))
    H_block.append(np.array([[Vg, V], [V, w]])/4)
    H_block.append(np.array([[w+Vg]])/4)
    H_block.append(np.array([[2*Vg+U, -np.sqrt(2)*V, 0], [-np.sqrt(2)*V, Vg+w, -np.sqrt(2)*V], [0, -np.sqrt(2)*V, 2*w]])/4)
    H_block.append(np.array([[2*Vg+U+w, -V], [-V, 2*w+Vg]])/4)
    H_block.append(np.array([[2*Vg+U+2*w]])/4)
    
    #The Hamiltonians must follow specific construction rules. The subspaces are aranged in 
    #diagonal order, and inside each subspace, the primitive states follow the SWEN order.
    
    return block_diag(*H_block), H_block


def ll_(bra, ket):
    """
    Receives: Bra and ket with different l to be pairwise multiplied.
    Performs: Pairwise multiplication of the vectors in a certain order. 
              We fix a value of l in the ket, and multiply by all the values of l' in the bra.
    Returns: Flat array with the multiplied values. 
    """

    return (np.outer(ket, bra)).flatten()
    

def eig_blocks(H0block, n_dim0):
    """
    Receives: An array with the Hamiltonians of each Subspace.
    Performs: Diagonalizes each block.
    Returns: An array with the Eigenvalue matrix and an array with the Eigenvector matrix, 
    in diagonal order.
    """
    E0block = []
    Vec0block = []
    nuv = []
    
    H0 = block_diag(*np.delete(H0block, n_dim0))
    eigval, eigvec = eigh(H0)

    for H_ in H0block: 
        if len(H_) == 0:
            E0block.append([])
            Vec0block.append([])
            nuv.append(0)
        else:
            aux_val, aux_vet = eigh(H_)
            i_uv = np.argwhere(aux_val-eigval[0] <= 30.).flatten() #40/35 normalmente
            E0block.append(block_diag(*aux_val[i_uv]-eigval[0]))
            Vec0block.append(aux_vet[:, i_uv])
            nuv.append(len(aux_val[i_uv]))
    
    return E0block, Vec0block, nuv

def iteration_0():
    """
    Receives: 
    Performs: Initializates the subspaces. 
    Returns: The initial Space matrix, an array with the initial energies, invariants.
    """
    
    Space =  np.full((5, 3), Subspace([], [], [0,0,0,0])) 
    fn_invar = []

    H0, H0block = initial_Hamiltonian()
    E0block, Vec0block, nuv0 = eig_blocks(H0block, [])
    
    Space[4, 0] = Subspace([], [],  [0, 0, 0, 1])        #Q = 2, S = 0
    Space[3, 1] = Subspace([], [],  [0, 0, 1, 1])        #Q = 1, S = 1/2
    Space[2, 2] = Subspace([], [],  [0, 0, 1, 0])        #Q = 0, S = 1
    Space[2, 0] = Subspace([], [], [1, 1, 0, 1])         #Q = 0, S = 0
    Space[1, 1] = Subspace([], [],  [1, 0, 1, 0])        #Q = -1, S = 1/2
    Space[0,0] =  Subspace([], [], [1, 0, 0, 0])         #Q = -2, S = 0
    
    fn_invar.append([[],[]])
    #Q = -2, S = 0
    fn_invar.append([[], Vec0block[1][1,:]]) 
    #Q = -1, S = 1/2
    fn_invar.append([-np.sqrt(2)*ll_(Vec0block[3][2, :],Vec0block[1][1,:])+ll_(Vec0block[3][1, :],Vec0block[1][0,:]),Vec0block[1][0,:]]) 
    #Q = 0, S = 1
    fn_invar.append([-np.sqrt(3/2)*Vec0block[4][1,:], []])
    #Q = 0, S = 0
    fn_invar.append([[], ll_(Vec0block[4][0, :], Vec0block[3][0, :])+ll_(Vec0block[4][1, :], Vec0block[3][1, :])/np.sqrt(2)])
    #Q = 1, S = 1/2
    fn_invar.append([-np.sqrt(2)*Vec0block[4][0,:], []])
    
    #The first element of the energy block array of matrixes is a null matrix. This was made 
    #to facilitate the construction of the next iteration Hamiltonian. 
        
    return Space, [[]] + E0block, np.array(fn_invar), [0] + nuv0

def around_Descendants(n, xQ, yS, Space, Space0, nuv0_n):
    """
    Receives: Position (xQ,yS) in Space0, Space and Space0.
    Performs: Given a point (xQ,yS) in Space0, fills its contribution to the gender vector of the
    neighboring positions in Space.
    Returns: The fixed Space matrix.
    """    
    xQ_ = xQ+1 #A position in Space0 is one position higher in Space.
    n_father = nuv0_n
    
    Space[xQ_-1, yS].gender[0] = n_father           #South
    Space[xQ_-1, yS].eigblock[0] = n    

    Space[xQ_, yS+1].gender[2] = n_father           #East
    Space[xQ_, yS+1].eigblock[2] = n  
    Space[xQ_, yS+1].invariants[1] = (n, 1)

    Space[xQ_+1, yS].gender[3] = n_father           #North   
    Space[xQ_+1, yS].eigblock[3] = n
    Space[xQ_+1, yS].invariants[2] = (n, 0)
    Space[xQ_+1, yS].invariants[3] = (n, 1)
    
    if yS-1 >= 0:
        Space[xQ_, yS-1].gender[1] = n_father       #West
        Space[xQ_, yS-1].eigblock[1] = n
        Space[xQ_, yS-1].invariants[0] = (n, 0)
    return Space

def empty_Space(N):
    """
    Receives: The iteration number N.
    Performs: Creates and fills the Space matrix with DIFFERENT objects. 
    Returns: Empty Space matrix.
    """
    Space =  np.full((5+2*N, 3+N), Subspace([], [], [])) 
    for i in range(5+2*N):
        for j in range(3+N):
            Space[i,j] = Subspace([0,0,0,0], [(0,0),(0,0),(0,0),(0,0)], [0,0,0,0])
    #The loop ensures that each object has a different location.
    return Space


def new_Space(Space0, N, nuv0):
    """
    Receives: Old space matrix Space0 and iteration number N.
    Performs: 
    Returns: 
    """
    N0 = N-1
    Qmax = 2+N0
    
    n = 1
    #Creates the new empty Space with the correct dimensions.
    Space = empty_Space(N)
    
    #The loop below defines the diagonal ordering. 
    for i, q in enumerate(range(0,2*Qmax+1, 2)):
        for d in range(0, 3+N0-i):
            Space = around_Descendants(n, q+d, d, Space, Space0, nuv0[n])
            n += 1 
    return Space


def HN(N, Space, Space0, E0, fn0, nuv0):
    """
    Receives: Iteration number N, previous energy array E0, invariants. 
    Performs: Calculates the Hamiltonian of the new iteration.
    Returns: An array whose entries are the new Hamiltonian of each subspace (q,s)
    in diagonal order.
    """
    n = 0
    n_dim0 = []
    Qmax0 = 2+N
    H_block = []
    
    for i, q in enumerate(range(0,2*Qmax0+1, 2)):
        for d in range(0, 3+N-i):
            index = np.array(Space[q+d,d].eigblock) 
            index = index[np.nonzero(np.array(nuv0)[index])] #Eliminates from E_father subspaces with no base elements.
            if len(index) == 0: #Fixes diagonal ordering for these subspaces. 
                H_block.append([])
                n_dim0.append(n)
            else:
                #E_father = block_diag(*list(list(itemgetter(*index)(E0))))     #  Evaluates E in index.
                E_father = block_diag(*np.array(E0)[index])
                H_block.append(fills_NonDiagonal(2*E_father/2, Space[q+d,d], fn0, d))    
            n += 1
    
    Eblock, Vecblock, nuv = eig_blocks(H_block, n_dim0)
        
    return [[]] + Eblock, [[]] +  Vecblock, [0]+nuv
    
def fills_NonDiagonal(E, QS, fn, yS):
    """
    Receives: A matrix E with the diagonal part of the new Hamiltonian; the subpace QS;
    the array of invariants, Spin S. 
    Performs: Fills the non-diagonal part of the Hamiltonian of the Subspace (Q,S).
    Returns: The Hamiltonian of the subspace (Q,S).
    """
    ns, nw, ne, nn = QS.gender
    i_w, i_e, i_n, dim = np.cumsum(QS.gender).astype(int)
   
    coef = 1/np.sqrt(2)  #coeficient that accompanies non diagonal term.

    if (ns*nw) != 0: #SW
        E[:i_w,i_w:i_e] = coef*np.reshape(fn[QS.invariants[0]], (ns,nw), 'F')
        
    if (ns*ne) != 0: #SE
        E[:i_w, i_e:i_n] = coef*np.reshape(fn[QS.invariants[1]], (ns,ne), 'F')
    
    if (ne*nn) != 0: #EN
        E[i_e:i_n, i_n:] = coef*np.sqrt(yS/(yS+1))*np.reshape(fn[QS.invariants[2]], (ne,nn), 'F')
    
    if (nw*nn) != 0: #WN
        E[i_w:i_e, i_n:] = -coef*np.sqrt((yS+2)/(yS+1))*np.reshape(fn[QS.invariants[3]], (nw,nn), 'F')

    
    return np.transpose(np.conjugate(E)) + E
        
def iteration_N(N, Space0, E0, fn0, nuv0):
    fn = []
    fn.append([[],[]])
    
    Space = new_Space(Space0, N, nuv0)
    Eblock, Vecblock, nuv = HN(N, Space, Space0, E0, fn0, nuv0)
    
    n = 1
    Qmax = 2+N    
    for i, q in enumerate(range(0,2*Qmax, 2)):
        for d in range(0, 3+N-i):
            fn = invariants(N, q+d, d, Vecblock, Space, fn, n, 2+N-i, nuv)
            n+=1
    return Space, Eblock, np.array(fn), nuv
        
def invariants(N, xQ, yS, Vecblock, Space, fn, n, dA, nuv):
    fnA = []
    fnB = []
    lA_ = [0,0,0,0]
    lB_ = [0,0,0,0]
    dim_lAuv = 0
    dim_lBuv = 0

    m_ = np.cumsum(Space[xQ, yS].gender).astype(int)

    if nuv[n] != 0:

        Vec_m = Vecblock[n]
        Vec_lA = np.conjugate(Vecblock[n+dA])
        Vec_lB = np.conjugate(Vecblock[n+1])

        if yS-1>=0:     #Bra Subspace inside the Space matrix
            lA_ = np.cumsum(Space[xQ+1, yS-1].gender).astype(int)
            dim_lAuv = nuv[n+dA]

        if yS+1<= 2+N:  #Bra Subspace inside the Space matrix
            lB_ = np.cumsum(Space[xQ+1, yS+1].gender).astype(int)
            dim_lBuv = nuv[n+1]


        if dim_lAuv != 0 and lA_[3]!=0:
            fnA = inv.invariant_loop_fna(Vec_m, Vec_lA, m_, lA_, yS, m_[3], nuv[n], lA_[3], dim_lAuv)
        if dim_lBuv !=0 and lB_[3]!=0:
            fnB = inv.invariant_loop_fnb(Vec_m, Vec_lB, m_, lB_, yS, m_[3], nuv[n], lB_[3], dim_lBuv)

            
    fn.append([fnA, fnB])
    return fn

def matrix_elements(N, E, fN):
    a0 = []
    f0 = []
    cd = []
    a0.append([[],[]])
    f0.append([[],[]])
    cd.append([[],[]])

    Qmax = 2 + N
    n = 1

    for i, q in enumerate(range(0,2*Qmax, 2)):
        for d in range(0, 3+N-i):
            a0A = []
            a0B = []
            f0A = []
            f0B = []
            cdA = []
            cdB = []

            if d != 0: #We have NW invariants.
                dE = -np.subtract.outer(np.diag(E[n]), np.diag(E[n+2+N-i])).flatten()
                a0A, f0A, cdA = iterative_fn(N, dE, fN[n, 0]) 

            if d != 2+N-i: #We have NE invariants. 
                dE = -np.subtract.outer(np.diag(E[n]), np.diag(E[n+1])).flatten()
                a0B, f0B, cdB = iterative_fn(N, dE, fN[n, 1])
            n+=1
            a0.append([a0A, a0B])
            f0.append([f0A, f0B])
            cd.append([cdA, cdB])

    return np.array(a0), np.array(f0), np.array(cd)

def iterative_fn(N, dE, fN):
    U, V, Vg, w = u_v_vg_w()
    fn2 = fN
    fn1 = np.sqrt(2)*np.multiply(dE, fN)

    if len(dE) == 0: #One of the subspaces (bra or ket) is empty: there is no matrix element.
        return [], [], []


    for n in range(N-3, -2, -1):
        fn = np.sqrt(2)*2**(n+2-N)*np.multiply(dE, fn1) - fn2/2
        fn2 = fn1
        fn1 = fn

    cd = np.multiply(dE, fn1)/(2**(N-2)*V) - np.sqrt(2)*fn2/V

    return fn1, fn2, cd #RETURNING a0, f0 and cd

#############################################OUTPUT#################################################
def updates_df(df, N, E0):
    Qmax = 2+N
    n = 1

    for i, q in enumerate(range(0,2*Qmax+1, 2)):
        for d in range(0, 3+N-i):
            dim = len(np.diag(E0[n]))
            if dim > 5:
                energ = np.diag(E0[n])[0:10]
                df = df.append({'Energy': energ, 'Subspace': [N, q+d-Qmax,d], 'Number of states': dim}, ignore_index=True)
            if dim < 5 and dim!=0 :
                energ = np.diag(E0[n])
                df = df.append({'Energy': energ, 'Subspace': [N, q+d-Qmax,d], 'Number of states': dim}, ignore_index=True)
            n+=1

    return df
#####################################################################################################
    
def iterative_diagonalization(Nmax):
    Space0, E0, fn0, nuv0 = iteration_0()
    for N in range(1,Nmax+1):
        Space0, E0, fn0, nuv0 = iteration_N(N, Space0, E0, fn0, nuv0)
        print(N)
    f0, f1 = matrix_elements(N, E0, fn0)
    #n=1
    #Qmax = 2+Nmax
    #for i, q in enumerate(range(0,2*(2+Nmax)+1, 2)):
    #   for d in range(0, 3+Nmax-i):
    #        print("_______________")
    #        print(q+d-Qmax, d, f0[n])
    #        print("_______________")

    #        n+=1

    return E0, f0, fn0

#Nmax = 1
#iterative_diagonalization(Nmax)




