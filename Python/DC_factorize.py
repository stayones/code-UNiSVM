import numpy as np
import pandas as pd
import copy
import time
from Loss_function import Loss_function


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


class DC_factorize:
    def __init__(self):
        self.data = None  # train or test data
        self.alpha = None
        self.P = None
        self.BS = None
        self.label = None
        self.test_data = None
        self.test_label = None
        self.ker_para = None  # used for kernel function
        self.testAccuracy = 0
        self.iteration = 0

    def dataload(self, str, mode):
        """
        :param str: filename
        :param mode: train or test
        :return: load file_data into the model
        """
        data = pd.read_csv(str, header=None)
        data = data.values
        if mode == 'train':
            self.label = data[:, -1]
            self.data = data[:, :-1]
        elif mode == 'test':
            self.test_data = data[:, :-1]
            self.test_label = data[:, -1]

    def ProvokeChol_factorize(self, subsetSize, errorBound):
        """
        to obtain the low-rank approximation about K = P * P(t)
        :param subsetSize: the max length of P
        :param errorBound: the limit for the error
        :return: calculate the P and save it in model
        """
        self.BS = []
        tr_2 = np.sum(self.data * self.data, 1)
        r = 0  # select time
        diag = np.ones(self.data.shape[0])
        k = 1 / (self.ker_para ** 2 * 2)

        limit = errorBound * self.data.shape[0]
        while np.sum(diag) + r * 1e2 > limit and r < subsetSize:
            index = np.argmax(diag)
            diag[index] = -1e2
            k_in = np.exp(-k * (tr_2 + tr_2[index] -
                                np.dot(self.data, 2 * self.data[index].T)))
            if r == 0:
                p = k_in / k_in[index]
                self.P = copy.deepcopy(p)
            else:
                nu = np.sqrt(k_in[index] - np.dot(self.P[index], self.P[index].T))
                p = (k_in - np.dot(self.P, self.P[index])) / nu
                p[self.BS] = 0
                self.P = np.column_stack((self.P, p))

            self.BS.append(index)
            diag -= p ** 2
            r += 1

    def train(self, lambda_, limit, ker_para, subsetSize, errorBound, str, args):
        """
        :param subsetSize: the max length of P
        :param errorBound: the limit for the error
        :param lambda_: param to be decided
        :param limit: to be decided -- to calculate the alpha
        :param ker_para: kernel param for kernel matrix
        :return:
        """
        self.ker_para = ker_para
        self.ProvokeChol_factorize(subsetSize, errorBound)

        y = self.label
        gama = np.zeros(self.data.shape[0])
        I = self.data.shape[0] * lambda_ * np.eye(self.P.shape[1])
        PBS = self.P[self.BS]

        Q = np.mat(np.dot((I + np.dot(self.P.T, self.P)), PBS.T)).I
        self.alpha = np.dot(np.dot(Q.A, self.P.T), y)
        gama1 = copy.deepcopy(gama)
        lf = Loss_function(str=str)
        Iter = 0
        while True:
            temp = np.dot(self.P, np.dot(PBS.T, self.alpha))

            x = 1 - y * temp  # depend on the loss function
            lf.Function(x=x, args=args)
            gama = - 0.5 / lf.A * y * lf.result

            Iter += 1
            self.alpha = np.dot(np.dot(Q.A, self.P.T), (temp - gama))
            if np.linalg.norm(gama - gama1) < limit:
                break
            gama1 = copy.deepcopy(gama)

        self.iteration = Iter

    def test(self):
        """
        test if the label we generate is correct
        :return: mark the accuracy
        """
        K = GaussKernel(np.mat(self.data[self.BS]), np.mat(self.test_data), self.ker_para)
        y = np.dot(K.A.T, self.alpha)
        t = self.test_label * y > 0
        t = t[t == True]
        self.testAccuracy = len(t) / self.test_label.shape[0] * 100


if __name__ == '__main__':
    a, b, c = 2, 2, 2
    p = 10
    eps = 0.1
    delta = 0.1

    # define all the loss functions to train, for more loss functions, see Loss_function.py
    # define every loss function's args, can be modified by the variant above
    loss_dict = {'Least Squares': [[a]],
                 'Smooth Hinge': [[p]],
                 'Squared Hinge': [[]],
                 'Truncated SH': [[a]],
                 'Truncated LS': [[a]],
                 'Smoothed Ramp1': [[a]],
                 'Smoothed Ramp2': [[a, p]],
                 'nonconvex exp loss': [[a, b, c], [2, 2, 4], [2, 3, 4]]
                 }
    # generate the model
    dc = DC_factorize()
    # load in the train data
    dc.dataload('/Users/wendy/Desktop/Al/SVM_M/data/adult_tr.csv', mode='train')
    # load in the test data
    dc.dataload('/Users/wendy/Desktop/Al/SVM_M/data/adult_ten.csv', mode='test')
    print('Training set: %d, Test set: %d' % (dc.data.shape[0], dc.test_data.shape[0]))

    # To train the model with all the loss functions
    # To test all the model
    for i in range(len(loss_dict.keys())):
        for j in loss_dict[list(loss_dict.keys())[i]]:
            start = time.time()
            dc.train(lambda_=1e-5, limit=1e-3, ker_para=20, subsetSize=1000, errorBound=1e-3,
                     str=list(loss_dict.keys())[i], args=j)
            times = time.time() - start
            dc.test()
            print('UniSVM:%d , Test Accuracy:%.2f %% time: %.2f , Iter: %d, lossName: %s' %
                  (i, dc.testAccuracy, times, dc.iteration, list(loss_dict.keys())[i]))
