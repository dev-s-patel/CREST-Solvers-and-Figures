import numpy as np


def build_positive_definite_covmat(
    input_file="crest_planck.covmat",
    output_file="crest_Alens.covmat",
    alens_sigma=0.03,
    min_eig=1e-5,
):
    with open(input_file, "r") as f:
        header = f.readline().strip().lstrip("#").split()

    mat = np.loadtxt(input_file, comments="#")
    std = np.sqrt(np.diag(mat))

    # Symmetrize correlation matrix and floor small eigenvalues
    corr = mat / np.outer(std, std)
    corr = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(corr)
    corr_pd = eigvecs @ np.diag(np.maximum(eigvals, min_eig)) @ eigvecs.T
    mat_pd = corr_pd * np.outer(std, std)

    # Append Alens variance block
    n = len(header)
    full_mat = np.zeros((n + 1, n + 1))
    full_mat[:n, :n] = mat_pd
    full_mat[n, n] = alens_sigma**2

    new_header = " " + " ".join(header + ["Alens"])
    np.savetxt(output_file, full_mat, header=new_header, comments="#")
    print(f"Saved positive-definite covariance matrix to {output_file}")


if __name__ == "__main__":
    build_positive_definite_covmat()