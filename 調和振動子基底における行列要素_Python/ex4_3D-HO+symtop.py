"""
Hamiltonian
H = p^2/2mu + B(j - \zeta l)^2 + \mu\omega^2 x^2/2

量子数mはよい量子数であるため，ブロック対角化できる．
 → 各mごとにHamiltonianを作り，対角化して，あとで固有値を合わせる．

回転状態の分布がすべて基底状態にあることはほぼないので回転状態に対してカノニカル分布を定義した．
このプログラムはかなり計算が重い．
多分まだバグがあるが計算に時間がかかりすぎるためデバッグできてない．
"""
#matrix size
j = 3
nvib = 1   #n = 2q + l 

#potential parameter
B_gs = 5.0  #rotational constant of vibrational ground state
B_ex = 5.0  #rotational constant of vibrational exited state
Bzeta = 0.5752  #Coriolis constant
T = 10  #temperature

vibhbaromega = 3000 #vibrational frequency

""""------------------------------------------------------------------
以下計算プログラム
------------------------------------------------------------------"""
import numpy as np
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt
hbar = 1.05457*10**(-34)    #Planck constant
kB = 0.695903     #Boltzmann constant cm-1 K-1

#calculation of matrix dimension
jdimsum = 0
for m in range(0, j + 1):
    for ja in range(m, j + 1):
        jdimsum += 2*ja + 1

ldim = 0;
for nvib_count in range (0, nvib + 1):
    for ll in range (nvib_count, - 1, -2):
        ldim += 2*ll + 1;
dim = (2*j + 1)*ldim

#calculation of matrix element
class MatrixElement:
    def __init__(self, m, jrow, krow, nvibrow, lvibrow, kvibrow, jcolumn, kcolumn, nvibcolumn, lvibcolumn, kvibcolumn):  
        self.m = m
        self.jrow = jrow
        self.krow = krow
        self.nvibrow = nvibrow
        self.lvibrow = lvibrow
        self.kvibrow = kvibrow
        self.jcolumn = jcolumn
        self.kcolumn = kcolumn
        self.nvibcolumn = nvibcolumn
        self.lvibcolumn = lvibcolumn
        self.kvibcolumn = kvibcolumn

    def Tr(self):
        if self.jrow == 0 and self.jcolumn == 0 and self.krow == self.kcolumn and self.nvibrow == self.nvibcolumn and self.lvibrow == self.lvibcolumn and self.kvibrow == self.kvibcolumn: 
            Tr = B_gs*self.jrow*(self.jrow + 1)
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.nvibrow == self.nvibcolumn and self.lvibrow == self.lvibcolumn and self.kvibrow == self.kvibcolumn: 
            Tr = B_ex*self.jrow*(self.jrow + 1)
        else: 
            Tr = 0
        return Tr
    
    def Tr_vib_1(self):
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.nvibrow == self.nvibcolumn and self.lvibrow == self.lvibcolumn and self.kvibrow == self.kvibcolumn: 
            Tr_vib_1 = self.lvibrow*(self.lvibrow + 1)
        else: 
            Tr_vib_1 = 0
        return Tr_vib_1
    
    def Tr_vib_2(self):
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.nvibrow == self.nvibcolumn and self.lvibrow == self.lvibcolumn and self.kvibrow == self.kvibcolumn: 
            Tr_vib_2 = self.krow*self.kvibrow
        else: 
            Tr_vib_2 = 0
        return Tr_vib_2
    
    def Tr_vib_3(self):
        if self.jrow == self.jcolumn  and self.nvibrow == self.nvibcolumn and self.lvibrow == self.lvibcolumn:
            if self.krow - self.kcolumn == + 1 and self.kvibrow - self.kvibcolumn == + 1:
                Tr_vib_3 = np.sqrt(self.jrow*(self.jrow + 1) - self.kcolumn*(self.kcolumn + 1))*np.sqrt(self.lvibrow*(self.lvibrow + 1) - self.kvibcolumn*(self.kvibcolumn + 1))
            elif self.krow - self.kcolumn == - 1 and self.kvibrow - self.kvibcolumn == - 1:
                Tr_vib_3 = np.sqrt(self.jrow*(self.jrow + 1) - self.kcolumn*(self.kcolumn - 1))*np.sqrt(self.lvibrow*(self.lvibrow + 1) - self.kvibcolumn*(self.kvibcolumn - 1))
            else:
                Tr_vib_3 = 0
        else: 
            Tr_vib_3 = 0
        return Tr_vib_3
    
    def Tvib(self):
        if self.jrow == self.jcolumn and self.krow == self.kcolumn and self.nvibrow == self.nvibcolumn and self.lvibrow == self.lvibcolumn and self.kvibrow == self.kvibcolumn:
            Tr_vib = (self.nvibrow + 3/2)
        else:
            Tr_vib = 0
        return Tr_vib

def Hamiltonian(m):
    
    result = np.array([[+ MatrixElement(m, jrow, krow, nvibrow, lvibrow, kvibrow, jcolumn, kcolumn, nvibcolumn, lvibcolumn, kvibcolumn).Tr()
                        + MatrixElement(m, jrow, krow, nvibrow, lvibrow, kvibrow, jcolumn, kcolumn, nvibcolumn, lvibcolumn, kvibcolumn).Tr_vib_1() *Bzeta**2
                        + MatrixElement(m, jrow, krow, nvibrow, lvibrow, kvibrow, jcolumn, kcolumn, nvibcolumn, lvibcolumn, kvibcolumn).Tr_vib_2() *(- 2*Bzeta)
                        + MatrixElement(m, jrow, krow, nvibrow, lvibrow, kvibrow, jcolumn, kcolumn, nvibcolumn, lvibcolumn, kvibcolumn).Tr_vib_3() *(- Bzeta)
                        + MatrixElement(m, jrow, krow, nvibrow, lvibrow, kvibrow, jcolumn, kcolumn, nvibcolumn, lvibcolumn, kvibcolumn).Tvib() *vibhbaromega

            for nvibcolumn in range(0, nvib + 1)               
            for lvibcolumn in range(nvibcolumn, -1, -2)
            for kvibcolumn in range(- lvibcolumn, lvibcolumn + 1)
            for jcolumn in range(abs(m), j + 1)
            for kcolumn in range(- jcolumn, jcolumn + 1)
        ]
        for nvibrow in range(0, nvib + 1)
        for lvibrow in range(nvibrow, -1, -2)
        for kvibrow in range(-lvibrow, lvibrow + 1)
        for jrow in range(abs(m), j + 1)
        for krow in range(-jrow, jrow + 1)
    ])
    
    result = result.astype(np.float64)
    return result

def diagonalization(H, m):
    dim = len(H)
    print("dimension, ", dim)
    eigen_vector = []
    eig_val,eig_vec = np.linalg.eigh(H)
    print('eigen value\n{}\n'.format(eig_val))
    
    ndim = []
    ldim = []
    ndimsum = [0]
    ldimsum = [0]
    allownvib = []
    allowlvib = []
    ndimsum_count = 0
    ldimsum_count = 0
    for nvib_count in range (0, nvib + 1):
        ndim_count = 0
        for ll in range (nvib_count, -1, -2):
            allowlvib.append(ll)
            ldim.append(2*ll + 1)
            ldimsum_count += 2*ll + 1
            ldimsum.append(ldimsum_count)
            ndim_count += 2*ll + 1
        ndim.append(ndim_count)
        allownvib.append(nvib_count)
        ndimsum_count += ndim_count
        ndimsum.append(ndimsum_count)
    
    jcount = []
    jdimsum = []
    jdimsum_count = [0]
    for ja in range(m, j + 1):
        jcount.append(ja)
        jb = (2*ja + 1)
        jdimsum.append(jb)
        jc = sum(jdimsum)
        jdimsum_count.append(jc)
    #eigen vector is columnn one.
    eigen_vector = []
    for r in range(0, len(eig_vec)):
        for c in range(0, len(eig_vec[r])):
            nsurp = c // sum(jdimsum)
            csurp = c % sum(jdimsum)
            for dn in range (0, len(ndimsum)):
                if ndimsum[dn] <= nsurp and nsurp < ndimsum[dn + 1]:
                    quantnvib = allownvib[dn]
            for dl in range (0, len(ldimsum)):
                if ldimsum[dl] <= nsurp and nsurp < ldimsum[dl + 1]:
                    quantlvib = allowlvib[dl]
                    quantkvib = nsurp - ldimsum[dl] - allowlvib[dl]
            for dj in range (0, len(jcount)):
                if jdimsum_count[dj] <= csurp and csurp < jdimsum_count[dj + 1]:
                    quantj = jcount[dj]
                    quantk = csurp - quantj*(quantj + 1) + m**2
                    quantm = m
            eigen_vector.append([float(eig_val[r]), float(eig_vec[r, c]), quantj, quantk, quantm, quantnvib, quantlvib, quantkvib])

    return eig_val, eigen_vector


# coeff A
def A(mB, p):
    if (mB == 0 and p ==0):
        A = 1/2
    else:
        A = 1
    return A

def IR_intensity(eigvec):   #1つの固有値に関する固有ベクトルの二次元配列を渡す関数
    distfunc = 0
    dist = []
    for val in eigvec:
        distfunc += np.exp(-(val[0] - vibhbaromega*1.5)/(kB*T))
        dist.append(np.exp(-(val[0] - vibhbaromega*1.5)/(kB*T)))
    dist = dist/distfunc
    
    print("状態分布\n", dist)

    intensity = []
    print("j dimension, ", jdimsum)
    for ii in range (0, jdimsum): #始状態
        for jj in range (jdimsum, len(eigvec)): #終状態
            cfcg = 0
            for c in range(0, len(eigvec[ii][1])): #始状態のベクトル  
                for cc in range(0, len(eigvec[jj][1])):    #終状態のベクトル
                    #energy, coeffs, j, k, m, nvib, lvib, kvib
                        coeff_init = eigvec[ii][1][c][1]
                        j_init = eigvec[ii][1][c][2]
                        k_init = eigvec[ii][1][c][3]
                        m_init = eigvec[ii][1][c][4]
                        nvib_init = eigvec[ii][1][c][5]
                        lvib_init = eigvec[ii][1][c][6]
                        kvib_init = eigvec[ii][1][c][7]         
                        
                        coeff_fin = eigvec[jj][1][cc][1]
                        j_fin = eigvec[jj][1][cc][2]
                        k_fin = eigvec[jj][1][cc][3]
                        m_fin = eigvec[jj][1][cc][4]
                        nvib_fin = eigvec[jj][1][cc][5]
                        lvib_fin = eigvec[jj][1][cc][6]
                        kvib_fin = eigvec[jj][1][cc][7]  

                        for p in range (-1, 1 + 1):
                            for q in range (-1, 1 + 1):
                                if (kvib_fin == q):
                                    cfcg += (A(m_fin, p)*coeff_fin*coeff_init
                                       *(-1)**(k_fin - m_fin)*((2*j_fin + 1)*(2*j_init + 1))**(1/2)
                                       *wigner_3j( j_fin,  1, j_init,
                                                  -m_fin,  p, m_init)
                                       *wigner_3j( j_fin,  1, j_init,
                                                  -k_fin,  q, k_init))
                        else:
                            cfcg += 0
            intensity.append([eigvec[jj][0] - eigvec[ii][0], abs(cfcg)**2])
    return intensity
    

def makegraph (spectra):
    grid = 1001
    start = 3000 - 50
    end = 3000 + 50
    wn = np.linspace(start, end, grid)
    state = [0]*grid
    for i in range (0, len(spectra)):
        for ii in range (0, grid):
            if round(wn[ii], 1) == round(spectra[i][0], 1):
                state[ii] += spectra[i][1]
    
    plt.figure(dpi = 300, figsize=(4, 3))
    plt.xlim(3000 - 50, 3000 + 50)
    plt.bar(wn, state, align="edge", width= 0.5, color="crimson")
    plt.show()


def main ():
    valvec = []
    #diagonarization for each m
    for m in range (0, j + 1):
        H = Hamiltonian(m)
        (val, vec) = diagonalization(H, m)
        print(m, "\n")
        vec = np.array(vec)
        coeffs = []
        for r in range(0, len(val)):
            num = vec[r*len(val):(r + 1)*len(val), :]
            coeffs.append(num)
        for r in range (0, len(val)):
            valvec.append([round(val[r], 2), coeffs[r]])
    #mごとに計算されていた固有値をエネルギーでソート
    valvec_multim =sorted(valvec, key = lambda x:(x[0]))
    makegraph(IR_intensity(valvec_multim))
    
    return
    
main()
