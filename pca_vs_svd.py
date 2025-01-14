# Adapted from https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

# imports
import pandas as pd
import numpy as np
from numpy import linalg as la

# Utility
def flip_signs(A, B):
    """
    utility function for resolving the sign ambiguity in SVD
    http://stats.stackexchange.com/q/34396/115202
    """
    signs = np.sign(A) * np.sign(B)
    return A, B * signs

# load data, Because this example shows a RNA-seq matrix with rows representing the genes and samples on the columns we transpose this matrix after centering and scaling when doing sample QC
df = pd.read_csv("rna_seq_matrix_selection.tsv", sep="\t")
X = df.values[:, 1:].astype(float)
# log transform, due to count data
X = np.log2(X + 1)

# Assumption: Let the real values data matrix X be of n x p size, where n is the number of samples and p is the number of variables.
n, p = X.shape
# Assumption: Let us assume that it is centered, i.e. column means have been subtracted and are now equal to zero.
X -= np.mean(X, axis=0)

# Optional: consider normalization
X /= X.std(axis=0)

#############
# PCA route #
#############

# covariance matrix
cov = np.cov(X.T)
# eigenvalues, eigenvectors
vals, vecs = np.linalg.eigh(cov)
# sort eigenvectors
idxs = vals.argsort()[::-1]
vals = vals[idxs]
vecs = vecs[:, idxs]
# project (principal components)
proj = np.dot(X, vecs)

#############
# SVD route #
#############

# Perform Single Value Decomposition of X, using full_matrices=False ("economy size"/"thin") such that K = min(n, p).
U, s, Vt = la.svd(X, full_matrices=False)
V = Vt.T
S = np.diag(s)

# Strictly speaking, U is of n x n size and V is of p x p size. However, if n>p then the last n-p columns of u are arbitrary (and corresponding rows of s are constant zero); one should therefore use an economy size (or thin) SVD that returns u of n x p size, dropping the useless columns. For large n>>p the matrix u would otherwise be unnecessarily huge. The same applies for an opposite situation of n<<p.
assert U.shape == (n, p)
assert S.shape == (p, p)
assert V.shape == (p, p)

################
# Theory proof #
################
# if X = USV^T, then the columns of V are principal directions/axes (eigenvectors).
assert np.allclose(*flip_signs(V, vecs))
# Columns of US are principal components ("scores").
assert np.allclose(*flip_signs(U.dot(S), proj))
# Singular values are related to the eigenvalues of covariance matrix. Eigenvalues show variances of the respective PCs.
assert np.allclose((s ** 2) / (n - 1), vals)


####################################
# Relationship between PCA and SVD #
####################################
k = 2
PC_k = proj[:, 0:k]
US_k = U[:, 0:k].dot(S[0:k, 0:k])
assert np.allclose(*flip_signs(PC_k, US_k))

############
# Plotting #
############
import matplotlib.pyplot as plt
plt.close()

fig, ax = plt.subplots(figsize=(6,6))
ax.axis('equal')
ax.scatter(PC_k[:,0], PC_k[:,1], s=1) 
fig.savefig("pca.png")

plt.close()

fig, ax = plt.subplots(figsize=(6,6))
ax.axis('equal')
ax.scatter(US_k[:,0], US_k[:,1], s=1) 
fig.savefig("svd.png")
