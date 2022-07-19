import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from NRG_ import iterative_diagonalization, u_v_vg_w, iteration_0, iteration_N, matrix_elements

def s_(s, sz):
    return np.sqrt((s*(s+1)-sz*(sz-1)))

def sp(s, sz):
    return np.sqrt((s*(s+1)-sz*(sz+1)))

def clebsch_gordan_1(s):
    CG = 1
    coef = 1

    for sz in np.arange(s, -s-1, -1):
        coef = coef*s_(s, sz)/s_(s+1/2, sz+1/2)
        CG += coef**2

    coef = 1
    CG += 1

    for sz in np.arange(-s, s+1, 1):
        coef = coef*sp(s, sz)/sp(s+1/2, sz-1/2)
        CG += coef**2

    return CG

def clebsch_gordan_0(s):

    CG = (np.sqrt(1/(2*s+1)))**2
    coef = -np.sqrt(1/(2*s+1))

    for sz in np.arange(s-1, -s, -1):
        coef = coef*s_(s, sz)/s_(s-1/2, sz+1/2)
        CG += coef**2

    return 2*CG

def L0_ll_(N, B, Ef_, Ep_, f0):
    Gll = 0
    i = 0
    for Ep in Ep_:
        for Ef in Ef_:
            Gll+= B*2**(N-2)*f0[i]**2/(np.exp(B*Ef) + np.exp(B*Ep))
            i += 1
    return Gll

def L0(B, N, E, f0):
    n=1
    G = 0
    Z = 0
    Qmax = 2+N
    #c = 1.790210618183453
    #d = 1.1346407922
    e = 1.5777774168618977

    #U, V, Vg, w = u_v_vg_w()

    for i, q in enumerate(range(0,2*Qmax, 2)):
        for d in range(0, 3+N-i):

            if d != 0:
                G += clebsch_gordan_0(d/2)*L0_ll_(N, B, np.diag(E[n+2+N-i]), np.diag(E[n]), f0[n, 0])

            if d != 2+N-i:
                G += clebsch_gordan_1(d/2)*L0_ll_(N, B, np.diag(E[n+1]), np.diag(E[n]), f0[n, 1])

            Z += (d+1)*np.sum(np.exp(-B*np.diag(E[n])))
            n+=1

    Z += np.sum(np.exp(-B*np.diag(E[n])))

    return e*2*G/Z
    #return np.pi**2*V**2*G/(2*Z)

def L1_ll_(N, B, Ef_, Ep_, f0):
    Gll = 0
    i = 0
    for Ep in Ep_:
        for Ef in Ef_:
            Gll+= B**2*2**(N-2)*f0[i]**2*(Ef-Ep)/(np.exp(B*Ef) + np.exp(B*Ep))
            i += 1
    return Gll

def L1(B, N, E, f0):
    n=1
    G = 0
    Z = 0
    Qmax = 2+N
    #c = 1.790210618183453
    e = 1.5777774168618977


    #U, V, Vg, w = u_v_vg_w()

    for i, q in enumerate(range(0,2*Qmax, 2)):
        for d in range(0, 3+N-i):

            if d != 0:
                G += clebsch_gordan_0(d/2)*L1_ll_(N, B, np.diag(E[n+2+N-i]), np.diag(E[n]), f0[n, 0])

            if d != 2+N-i:
                G += clebsch_gordan_1(d/2)*L1_ll_(N, B, np.diag(E[n+1]), np.diag(E[n]), f0[n, 1])

            Z += (d+1)*np.sum(np.exp(-B*np.diag(E[n])))
            n+=1

    Z += np.sum(np.exp(-B*np.diag(E[n])))

    return e*2*G/Z

def L2_ll_(N, B, Ef_, Ep_, f0):
    Gll = 0
    i = 0
    for Ep in Ep_:
        for Ef in Ef_:
            Gll+= B**3*2**(N-2)*f0[i]**2*(Ef-Ep)**2/(np.exp(B*Ef) + np.exp(B*Ep))
            i += 1
    return Gll

def L2(B, N, E, f0):
    n=1
    G = 0
    Z = 0
    Qmax = 2+N
    #c = 1.790210618183453
    e = 1.5777774168618977

    #U, V, Vg, w = u_v_vg_w()

    for i, q in enumerate(range(0,2*Qmax, 2)):
        for d in range(0, 3+N-i):

            if d != 0:
                G += clebsch_gordan_0(d/2)*L2_ll_(N, B, np.diag(E[n+2+N-i]), np.diag(E[n]), f0[n, 0])

            if d != 2+N-i:
                G += clebsch_gordan_1(d/2)*L2_ll_(N, B, np.diag(E[n+1]), np.diag(E[n]), f0[n, 1])

            Z += (d+1)*np.sum(np.exp(-B*np.diag(E[n])))
            n+=1

    Z += np.sum(np.exp(-B*np.diag(E[n])))

    return e*2*G/Z
    #return np.pi**2*V**2*G/(2*Z)

def L01_ll_(N, B, Ef_, Ep_, f0, f1):
    Gll = 0
    i = 0
    for Ep in Ep_:
        for Ef in Ef_:
            Gll+= B**2*2**(N-2)*(2*f0[i]*f1[i])*(Ef-Ep)/(np.exp(B*Ef) + np.exp(B*Ep))
            i += 1
    return Gll

def L01(B, N, E, f0, f1):
    n=1
    G = 0
    Z = 0
    Qmax = 2+N
    #c = 1.790210618183453
    e = 1.5777774168618977

    #U, V, Vg, w = u_v_vg_w()

    for i, q in enumerate(range(0,2*Qmax, 2)):
        for d in range(0, 3+N-i):

            if d != 0:
                G += clebsch_gordan_0(d/2)*L01_ll_(N, B, np.diag(E[n+2+N-i]), np.diag(E[n]), f0[n, 0], f1[n, 0])

            if d != 2+N-i:
                G += clebsch_gordan_1(d/2)*L01_ll_(N, B, np.diag(E[n+1]), np.diag(E[n]), f0[n, 1], f1[n, 1])

            Z += (d+1)*np.sum(np.exp(-B*np.diag(E[n])))
            n+=1

    Z += np.sum(np.exp(-B*np.diag(E[n])))
    return e*2*G/Z

def print_energies(E0, N, folder):
    Qmax = 2+N
    n = 1

    file = open("resultados/resultados" + str(folder) +"/energies/" + str(N) + ".txt", "w+")


    for i, q in enumerate(range(0,2*Qmax+1, 2)):
        for d in range(0, 3+N-i):

            #if q+d-Qmax == 1 and d == 1:
            dim = len(np.diag(E0[n]))
                
            if dim > 10:
                np.savetxt(file, np.diag(E0[n])[0:10], header='Subspace:' + str([q+d-Qmax,d]))

            if dim <= 10 and dim != 0:
                np.savetxt(file, np.diag(E0[n]), header='Subspace:' + str([q+d-Qmax,d]))

            n+=1

    file.close()


def dynamical_proprieties(Nmax, folder):
    L0N = [] 
    L1N = []  
    L2N = []
    L01v = []


    B_ = 0.025*2.**(np.arange(85, 105)/20)
    print(B_)


    Space0, E0, fn0, nuv0 = iteration_0()
    for N in range(1,Nmax+1):
        auxL0 = []
        auxL1 = []
        auxL2 = []
        auxL01 = []

        Space0, E0, fn0, nuv0 = iteration_N(N, Space0, E0, fn0, nuv0)
        print_energies(E0, N, folder)

        print(N)


        f0, f1, cd = matrix_elements(N, E0, fn0)
        
        for B in B_:

            auxL0.append(L0(B, N, E0, f0))
            auxL1.append(L1(B, N, E0, f0))
            auxL2.append(L2(B, N, E0, f0))
            auxL01.append(L01(B, N, E0, f0, f1))

        L0N.append(auxL0)
        L1N.append(auxL1)
        L2N.append(auxL2)
        L01v.append(auxL01)

    #print_energies(E0, N, folder)

    L0N = np.array(L0N)
    L1N = np.array(L1N)
    L2N = np.array(L2N)
    L01v = np.array(L01v)

    file = open("resultados/resultados"+ str(folder)+"/L0_.txt", "w")
    for i, G in enumerate(L0N):
        N = i+1
        np.savetxt(file, np.c_[1/(B_*2**(N-2)), G, B_])
    file.close()

    file = open("resultados/resultados"+ str(folder)+"/L1_.txt", "w")
    for i, S in enumerate(L1N):
        N = i+1
        np.savetxt(file, np.c_[1/(B_*2**(N-2)), S, B_])
    file.close()

    file = open("resultados/resultados"+ str(folder)+"/L2_.txt", "w")
    for i, K in enumerate(L2N):
        N = i+1
        np.savetxt(file, np.c_[1/(B_*2**(N-2)), K, B_])
    file.close()

    file = open("resultados/resultados"+ str(folder)+"/L01_.txt", "w")
    for i, L in enumerate(L01v):
        N = i+1
        np.savetxt(file, np.c_[1/(B_*2**(N-2)), L, B_])
    file.close()





N= 46
folder = 1
dynamical_proprieties(N, folder)


