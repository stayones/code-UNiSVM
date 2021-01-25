import numpy as np
import pandas as pd
import copy
import scipy
from scipy import sparse
from LossFunction import Loss_function
from KernelFactorize import CholeskyFactorize
from KernelFactorize import GaussKernel
from array import array


class UniSVM:
    def __init__(self):
        self.data = None  # train or test data
        self.alpha = None
        self.P = None
        self.BS = None
        self.label = None
        self.test_data = None
        self.test_label = None
        self.ker_para = None  # used for kernel function
        # mark the result
        self.testAccuracy = 0
        self.iteration = 0

    def dataload(self, str, mode):
        """
        :param str: filename
        :param mode: train/ test
        :return: load file_data into the model
        """
        if mode == 'train':
            self.label, self.data = self.svm_read_problem(str, True)
        elif mode == 'test':
            self.test_label, self.test_data = self.svm_read_problem(str, True)

    def svm_read_problem(self, data_file_name, return_scipy=False):
        """
        svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
        svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

        Read LIBSVM-format data from data_file_name and return labels y
        and data instances x.
        """
        if scipy != None and return_scipy:
            prob_y = array('d')
            prob_x = array('d')
            row_ptr = array('l', [0])
            col_idx = array('l')
        else:
            prob_y = []
            prob_x = []
            row_ptr = [0]
            col_idx = []
        indx_start = 1
        for i, line in enumerate(open(data_file_name)):
            line = line.split(None, 1)
            # In case an instance with all zero features
            if len(line) == 1: line += ['']
            label, features = line
            prob_y.append(float(label))
            if scipy != None and return_scipy:
                nz = 0
                for e in features.split():
                    ind, val = e.split(":")
                    if ind == '0':
                        indx_start = 0
                    val = float(val)
                    if val != 0:
                        col_idx.append(int(ind) - indx_start)
                        prob_x.append(val)
                        nz += 1
                row_ptr.append(row_ptr[-1] + nz)
            else:
                xi = {}
                for e in features.split():
                    ind, val = e.split(":")
                    xi[int(ind)] = float(val)
                prob_x += [xi]
        if scipy != None and return_scipy:
            prob_y = pd.DataFrame(scipy.frombuffer(prob_y, dtype='d'))
            prob_x = scipy.frombuffer(prob_x, dtype='d')
            col_idx = scipy.frombuffer(col_idx, dtype='l')
            row_ptr = scipy.frombuffer(row_ptr, dtype='l')
            prob_x = pd.DataFrame(sparse.csr_matrix((prob_x, col_idx, row_ptr)).todense())
        return (prob_y, prob_x)

    def train(self, lambda_, limit, ker_para, subsetSize, errorBound, str, args):
        """
        :param subsetSize: the max length of P
        :param errorBound: the limit for the error
        :param lambda_: param to be decided
        :param limit: to be decided -- to calculate the alpha
        :param ker_para: kernel param for kernel matrix
        :return:
        """
        self.P, self.BS = CholeskyFactorize(self.data, ker_para, subsetSize, errorBound)
        self.ker_para = ker_para
        y = self.label
        gama = pd.DataFrame(np.zeros(self.data.shape[0]))
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
        if self.data.shape[1] != self.test_data.shape[1]:
            self.test_data = pd.concat([self.test_data, pd.DataFrame(np.zeros((self.test_data.shape[0], np.abs(self.test_data.shape[1] - self.data.shape[1]))))], axis=1)
        K = GaussKernel(np.mat(self.data.iloc[self.BS]), np.mat(self.test_data), self.ker_para)
        y = np.dot(K.A.T, self.alpha)
        t = (self.test_label * y > 0).value_counts()[True]

        self.testAccuracy = t / self.test_label.shape[0] * 100



