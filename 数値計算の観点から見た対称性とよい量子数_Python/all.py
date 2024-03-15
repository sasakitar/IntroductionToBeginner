"""

"""
#matrix size
j = 3

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
    def __init__(self, jrow, krow, mrow, jcolumn, kcolumn, mcolumn):  
        self.jrow = jrow
        self.krow = krow
        self.mrow = mrow
        self.jcolumn = jcolumn
        self.kcolumn = kcolumn
        self.mcolumn = mcolumn

    def Tr(self):
        if self.jrow == 0 and self.jcolumn == 0 and self.krow == self.kcolumn and self.mrow == self.mcolumn: 
            Tr = B*self.jrow*(self.jrow + 1)
        else: 
            Tr = 0
        return Tr
    
    def Wigner3j_product(self, p, q):
        symbol1 = wigner_3j( self.jrow, p, self.jcolumn,
                            -self.krow, q, self.kcolumn)
    
        symbol2 = wigner_3j( self.jrow, p, self.jcolumn,
                            -self.mrow, 0, self.mcolumn)

        symbol = symbol1*symbol2
        return symbol
    
    def Vr(self, p, q): 
        Vr = ((-1)**(self.mrow - self.kcolumn))*np.sqrt((2*self.jcolumn + 1)*(2*self.jrow + 1))*self.Wigner3j_product(p, q)
        return Vr
    
    def V_3(self):
        V3 = self.Vr(3, 2) + self.Vr(3, -2)
        return V3

def Hamiltonian():
    # 時間計測開始
    time_sta_gen = time.time()
    
    result = np.array([[+ MatrixElement(jrow, krow, mrow, jcolumn, kcolumn, mcolumn).Tr()
                        + MatrixElement(jrow, krow, mrow, jcolumn, kcolumn, mcolumn).V_3()*V3
                        

            for jcolumn in range(0, j + 1)
            for mcolumn in range(- jcolumn, jcolumn + 1)
            for kcolumn in range(- jcolumn, jcolumn + 1)
        ]
        for jrow in range(0, j + 1)
        for mrow in range(-jrow, jrow + 1)
        for krow in range(-jrow, jrow + 1)
    ])
    
    # 時間計測終了
    time_end_gen = time.time()
    tim_gen = time_end_gen - time_sta_gen
    print('gen time: ',tim_gen)
    
    result = result.astype(np.float64)
    return result, tim_gen

def diagonalization(H):
    dim = len(H)
    print("dimension, ", dim)
    eigen_vector = []
    
    # 時間計測開始
    time_sta_diag = time.time()
    
    eig_val,eig_vec = np.linalg.eigh(H)
    print('eigen value\n{}\n'.format(np.round(eig_val, 4)))
    
    # 時間計測終了
    time_end_diag = time.time()
    tim_diag = time_end_diag - time_sta_diag
    print('diag time: ',tim_diag)
    
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
                    quantk = csurp - quantj*(quantj + 1)
                    quantm = m
            eigen_vector.append([float(eig_val[r]), float(eig_vec[r, c]), quantj, quantk, quantm])

    return eig_val, eigen_vector, tim_diag


def main ():
    f = open('run_time_all.txt', 'a')
    # 時間計測開始
    time_sta_whole = time.time()
    
    H, tim_gen = Hamiltonian()
    val, vec, tim_diag = diagonalization(H)
    
    # 時間計測終了
    time_end_whole = time.time()
    tim_whole = time_end_whole - time_sta_whole
    print('whole time: ',tim_whole)
    f.write(str(j) + ",\t" + str(len(H)) + ",\t" + str(tim_gen)  + ",\t" + str(tim_diag) + ",\t" + str(tim_whole) + '\n')
    f.close
    
    return
    
main()
