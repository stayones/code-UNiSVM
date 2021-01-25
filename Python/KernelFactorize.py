import numpy as np
import copy


def CholeskyFactorize(FacorizeData, KerPara, subsetSize, errorBound):
    """
    to obtain the low-rank approximation about K = P * P(t)
    :param FacorizeData: the data to be factorized
    :param KerPara: the kernel parameter
    :param subsetSize: the max length of P
    :param errorBound: the limit for the error
    :return: calculated the P and BS
    """
    BS = []
    tr_2 = np.sum(FacorizeData * FacorizeData, 1)
    r = 0  # select time
    diag = np.ones(FacorizeData.shape[0])
    k = 1 / (KerPara ** 2 * 2)

    limit = errorBound * FacorizeData.shape[0]
    while np.sum(diag) + r * 1e2 > limit and r < subsetSize:
        index = np.argmax(diag)
        diag[index] = -1e2
        k_in = np.exp(-k * (tr_2 + tr_2[index] -
                            np.dot(FacorizeData, 2 * FacorizeData.iloc[index].T)))
        if r == 0:
            p = k_in / k_in[index]
            P = copy.deepcopy(p)
        else:
            nu = np.sqrt(k_in[index] - np.dot(P[index], P[index].T))
            p = (k_in - np.dot(P, P[index])) / nu
            p[BS] = 0
            P = np.column_stack((P, p))

        BS.append(index)
        diag -= p ** 2
        r += 1
    return P, BS


def GaussKernel(X, Z, sigma):
    """
    :param X:
    :param Z:
    :param sigma:
    :return:
    """
    X_2 = np.sum(np.multiply(X, X), axis=1)
    Z_2 = np.sum(np.multiply(Z, Z), axis=1)
    m = np.dot(2 * X, Z.T)
    return np.exp((X_2 + Z_2.T - m) / (-2 * sigma ** 2))
