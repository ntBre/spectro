import numpy as np


def print_mat(mat, rows, cols):
    for row in range(0, rows):
        for col in range(0, cols):
            print(f"{mat[row,col]:14.8f}", end="")
        print()


fxm = np.loadtxt("../testfiles/ph3/fxm_full", dtype=np.double)
want = np.loadtxt("../testfiles/ph3/pre_bdegnl_lxm", dtype=np.double)

w, v = np.linalg.eigh(fxm)

v = np.fliplr(v)

print("got=")
print_mat(v, 12, 6)

print("want=")
print_mat(want, 12, 6)

print("diff=")
print_mat(v - want, 12, 6)

EPS = 1e-6

for col in range(0, 6):
    got = v[:, col]
    iwant = want[:, col]
    if (norm1 := np.linalg.norm(got - iwant)) > EPS:
        got = -got
        if (norm2 := np.linalg.norm(got - iwant)) > EPS:
            print(f"diff of {min(norm1, norm2)} in col {col}")
            assert False
