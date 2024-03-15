"""

"""
#matrix size
j = 13

#potential parameter
B = 5.0  #rotational constant of vibrational ground state
V3 = -30

""""------------------------------------------------------------------
以下計算プログラム
------------------------------------------------------------------"""
import numpy as np
from sympy.physics.wigner import wigner_3j
import matplotlib.pyplot as plt
import time

#calculation of matrix dimension
jdimsum = 0
for m in range(0, j + 1):
    for ja in range(m, j + 1):
        jdimsum += 2*ja + 1

#calculation of matrix element
class MatrixElement:
    def __init__(self, m, jrow, krow, jcolumn, kcolumn):  
        self.m = m
        self.jrow = jrow
        self.krow = krow
        self.jcolumn = jcolumn
        self.kcolumn = kcolumn

    def Tr(self):
        if self.jrow == 0 and self.jcolumn == 0 and self.krow == self.kcolumn: 
            Tr = B*self.jrow*(self.jrow + 1)
        else: 
            Tr = 0
        return Tr
    
    def Wigner3j_product(self, p, q):
        symbol1 = wigner_3j( self.jrow, p, self.jcolumn,
                            -self.krow, q, self.kcolumn)
    
        symbol2 = wigner_3j( self.jrow, p, self.jcolumn,
                            -self.m   , 0, self.m      )

        symbol = symbol1*symbol2
        return symbol
    
    def Vr(self, p, q): 
        Vr = ((-1)**(self.m - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.Wigner3j_product(p, q)
        return Vr
    
    def V_3(self):
        V3 = self.Vr(3, 2) + self.Vr(3, -2)
        return V3

def Hamiltonian(m, tim_gen):
    # 時間計測開始
    time_sta_gen = time.time()
    
    result = np.array([[+ MatrixElement(m, jrow, krow, jcolumn, kcolumn).Tr()
                        + MatrixElement(m, jrow, krow, jcolumn, kcolumn).V_3()*V3
                        

            for jcolumn in range(abs(m), j + 1)
            for kcolumn in range(- jcolumn, jcolumn + 1)
        ]
        for jrow in range(abs(m), j + 1)
        for krow in range(-jrow, jrow + 1)
    ])
    
    dim = len(result)
    print("dimension, ", dim)
    
    # 時間計測終了
    time_end_gen = time.time()
    tim_gen += time_end_gen - time_sta_gen
    print('gen time: ',time_end_gen - time_sta_gen)
    
    result = result.astype(np.float64)
    return result, tim_gen

def diagonalization(H, m, tim_diag):
    eigen_vector = []
    
    # 時間計測開始
    time_sta_diag = time.time()
    
    eig_val,eig_vec = np.linalg.eigh(H)
    
    # 時間計測終了
    time_end_diag = time.time()
    tim_diag += time_end_diag - time_sta_diag
    print('diag time: ',time_end_diag - time_sta_diag)
    
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
            csurp = c % sum(jdimsum)
            for dj in range (0, len(jcount)):
                if jdimsum_count[dj] <= csurp and csurp < jdimsum_count[dj + 1]:
                    quantj = jcount[dj]
                    quantk = csurp - quantj*(quantj + 1) + m**2
                    quantm = m
            eigen_vector.append([float(eig_val[r]), float(eig_vec[r, c]), quantj, quantk, quantm])

    return eig_val, eigen_vector, tim_diag


def main ():
    f = open('run_time_sep.txt', 'a')
    # 時間計測開始
    time_sta_whole = time.time()
    
    valvec = []
    tim_gen = 0
    tim_diag = 0
    #diagonarization for each m
    for m in range (-j, j + 1):
        H, tim_gen = Hamiltonian(m, tim_gen)
        (val, vec, tim_diag) = diagonalization(H, m, tim_diag)
        print(m, "\n")
        vec = np.array(vec)
        coeffs = []
        for r in range(0, len(val)):
            num = vec[r*len(val):(r + 1)*len(val), :]
            coeffs.append(num)
        for r in range (0, len(val)):
                valvec.append([round(val[r], 4), coeffs[r]])
    #mごとに計算されていた固有値をエネルギーでソート
    valvec_multim =sorted(valvec, key = lambda x:(x[0]))
    print("all dimension: ", len(valvec_multim))
    print('all gen time: ',tim_gen)
    print('all diag time: ',tim_diag)
    values = []
    for ii in range(0, len(valvec_multim)):
        values.append(valvec_multim[ii][0])
        #print(valvec_multim[ii][0], end=' , ')
    values = np.array(values)
    print(values)
    
    # 時間計測終了
    time_end_whole = time.time()
    tim_whole = time_end_whole - time_sta_whole
    print('whole time: ',tim_whole)
    f.write(str(j) + ",\t" + str(tim_gen)  + ",\t" + str(tim_diag) + ",\t" + str(tim_whole) + '\n')
    f.close
    

    return
    
main()
