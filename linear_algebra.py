# %%
from typing import List
from typing import Tuple
from typing import Callable
import math

# %% [markdown]
# VECTORS

# %%
Vector = List[float]

height_weight_age = [70,  # inches,
                     170, # pounds,
                     40 ] # years

grades = [95,   # exam1
          80,   # exam2
          75,   # exam3
          62 ]  # exam4

# %%
def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

# %%
def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]

# %%
def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    assert vectors, "no vectors provided!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1,2], [2,3], [3,4]]) == [6,9]


# %%
def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    
    return [c * v_i for v_i in v] 

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]

# %%
def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [2, 3], [3, 4]]) == [2, 3]

# %%
def dot(v: Vector, w: Vector) -> Vector: 
    """Computes sum of component-wise products"""
    assert len(v) == len(w), "Vectors must be the same length"
    
    return sum(v_i * w_i for v_i, w_i in zip(v, w)) 

assert dot([1, 2, 3], [4, 5, 6]) == 32

# %%
def sum_of_squares(v: Vector) -> Vector: 
    """Computes sum of the squares of each element of the vector""" 

    return dot(v, v)

assert sum_of_squares([1, 2, 3]) == 14

# %%
def magnitude(v: Vector) -> Vector:
    """Computes magnitude/length of the vector"""

    return math.sqrt(sum_of_squares(v))

assert magnitude([3, 4]) == 5

# %%
def squared_distance(v: Vector, w: Vector) -> float:
    """Computes squared distance of the 2 vectors"""

    return sum_of_squares(subtract(v, w))

assert squared_distance([3, 4, 5], [1, 2, 3]) == 12

# %%
def distance_1(v: Vector, w: Vector) -> float:
    """Computes distance between the 2 vectors: 1st method"""

    return math.sqrt(squared_distance(v, w))

assert distance_1([4, 1], [1, 5]) == 5

# %%
def distance_2(v: Vector, w: Vector) -> float:
    """Computes distance between the 2 vectors: 2nd method"""

    return magnitude(subtract(v, w))

assert distance_2([4, 1], [1, 5]) == 5

# %% [markdown]
# MATRICES
# 

# %%
Matrix = List[List[float]]

A = [[1, 2, 3],
     [4, 5, 6]]

B = [[1, 2],     
     [3, 4],
     [5, 6]]

# %%
def shape(A: Matrix) -> Tuple[int, int]:
    """Returns number of rows and columns of the matrix"""

    num_rows = len(A)
    num_cols = len(A[0]) if A else 0

    return num_rows, num_cols

assert shape([[1, 2, 3],
              [2, 5, 1]]) == (2,3)

# %%
def get_row(A: Matrix, i: int) -> Vector:
    """Returns ith row of the matrix"""

    return A[i]

# %%
def get_column(A: Matrix, j: int) -> Vector:
    """Returns jth column of the matrix"""

    return [A_i[j] for A_i in A]

# %%
def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """Creates matrix"""

    return [[entry_fn(i, j) 
            for j in range(num_cols)]
            for i in range(num_rows)]

# %%
def identity_matrix(n: int) -> Matrix: 
    """Returns identity matrix"""

    return make_matrix(n, n, lambda i, j: 1 if i==j else 0)

assert identity_matrix(3) == [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]


