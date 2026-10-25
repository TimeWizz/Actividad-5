import numpy as numpy
from scipy.linalg import lu_factor, lu_solve

# Matriz A con componentes convertidos a decimales (float) para procesamiento
A = numpy.array(
    [[2, -5, 1],
     [-1, 3, -1],
     [3, -4, 2]],
    dtype = float)

# Vector b con componentes convertidos a decimales (float) para procesamiento
b = numpy.array(
    [12, -8, 16],
    dtype = float)

# Función de factorización: recibe como parámetro una matriz y retorna una matriz
# L y una matriz U
def factorizacion_lu(matrix_A):
    n = len(matrix_A)
    print(n)
    L = numpy.zeros((n, n))
    U = numpy.zeros((n, n))
    
    for i in range(n):
        # La diagonal de L es 1
        L[i, i] = 1.0
        
        # Cálculo de la matriz U
        for j in range(i, n):
            suma = sum(L[i, k] * U[k, j] for k in range(i))
            U[i, j] = A[i][j] - suma
            
        # Cálculo de la matriz L
        for j in range(i + 1, n):
            suma = sum(L[j, k] * U[k, i] for k in range(i))
            L[j, i] = (A[j][i] - suma) / U[i, i]
            
    return L, U

L, U = factorizacion_lu(A)

print("Matriz L:\n", L)
print("Matriz U:\n", U)

# Comprobación de que L * U es igual a A
print("\nComprobación (L x U):\n", numpy.dot(L, U))


# lu_factor devuelve la matriz compacta y el vector de pivotes
lu_piv = lu_factor(A)

# Resolver el sistema para b
solucion_scipy = lu_solve(lu_piv, b)
print("Solución con SciPy:", solucion_scipy)

