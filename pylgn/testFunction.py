import numpy as np
import pylgn
from pylgn import kernels as kernel
import quantities as pq
import scipy.io as sio
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def find_max(matrix):
    '''
    find maximum value in a matrix
    '''
    new_data = []
    for i in range(len(matrix)):
        new_data.append(max(matrix[i]))
    return max(new_data)

def find_min(matrix):
    '''
    find maximum value in a matrix
    '''
    new_data = []
    for i in range(len(matrix)):
        new_data.append(min(matrix[i]))
    return min(new_data)

def normalizeMatrix(input_matrix):
    '''
    :param input_matrix: a numpy matrix
    :return: a numpy matrix
    '''
    return (input_matrix-find_min(input_matrix))/(find_max(input_matrix)-find_min(input_matrix))

def RSquare(matrix1, matrix2):
    R_square = 0
    for i in range(0,13):
        slope, intercept, r_value, p_value, std_err = st.linregress(matrix1[i], matrix2[i])
        R_square = R_square + r_value ** 2
    R_Square = R_square / 13
    return R_Square

def correlation(matrix1,matrix2):
    correlation = 0
    for i in range(0, 13):
        s1 = pd.Series(matrix1[i])
        s2 = pd.Series(matrix2[i])
        correlation = correlation + s2.corr(s1)
    correlation = correlation / 13
    return correlation

def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
        '''
        计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
        注意有展平操作。
        :param arr1:矩阵1
        :param arr2:矩阵2
        :return:实际是夹角的余弦值，ret = (cos+1)/2
        '''
        farr1 = arr1.ravel()
        farr2 = arr2.ravel()
        len1 = len(farr1)
        len2 = len(farr2)
        if len1 > len2:
            farr1 = farr1[:len2]
        else:
            farr2 = farr2[:len1]

        numer = np.sum(farr1 * farr2)
        denom = np.sqrt(np.sum(farr1 ** 2) * np.sum(farr2 ** 2))
        similar = numer / denom
        return (similar + 1) / 2

def mtx_similar2(arr1: np.ndarray, arr2: np.ndarray) -> float:
        '''
        计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
        如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
        :param arr1:矩阵1
        :param arr2:矩阵2
        :return:相似度（0~1之间）
        '''
        if arr1.shape != arr2.shape:
            minx = min(arr1.shape[0], arr2.shape[0])
            miny = min(arr1.shape[1], arr2.shape[1])
            differ = arr1[:minx, :miny] - arr2[:minx, :miny]
        else:
            differ = arr1 - arr2
        numera = np.sum(differ ** 2)
        denom = np.sum(arr1 ** 2)
        similar = 1 - (numera / denom)
        return float(similar)

def mtx_similar3(arr1: np.ndarray, arr2: np.ndarray) -> float:
        '''
        There are many ways to decide whether
        two matrices are similar; one of the simplest is the Frobenius norm. In case
        you haven't seen it before, the Frobenius norm of two matrices is the square
        root of the squared sum of differences of all elements; in other words, reshape
        the matrices into vectors and compute the Euclidean distance between them.
        difference = np.linalg.norm(dists - dists_one, ord='fro')
        :param arr1:矩阵1
        :param arr2:矩阵2
        :return:相似度（0~1之间）
        '''
        if arr1.shape != arr2.shape:
            minx = min(arr1.shape[0], arr2.shape[0])
            miny = min(arr1.shape[1], arr2.shape[1])
            differ = arr1[:minx, :miny] - arr2[:minx, :miny]
        else:
            differ = arr1 - arr2
        dist = np.linalg.norm(differ, ord='fro')
        len1 = np.linalg.norm(arr1)
        len2 = np.linalg.norm(arr2)
        denom = (len1 + len2) / 2
        similar = 1 - (dist / denom)
        return similar

def preprocess(best):
    paraDict_tp = {'excPhase':best['excPhase'],
                   'excPhaseDelay':best['excPhaseDelay'],
                   'inhPhase': best['inhPhase'],
                   'inhPhaseDelay': best['inhPhaseDelay'],
                   'excDamping': best['excDamping'],
                   'inhDamping': best['inhDamping']
                   }
    paraDict_sp = {'a': best['a'],
                   'B': best['B'],
                   'b': best['b']}
    paraDict_weight = {'offWeight': best['offWeight']}
    return paraDict_tp, paraDict_sp, paraDict_weight

class operation:
    def __init__(self, cellIndex=1, cellType='offsta', fieldSize=7):
        self.fieldSize = fieldSize
        self.data = sio.loadmat('allFinal2.mat')
        self.cellIndex = cellIndex
        self.cellType = cellType
        if self.cellType == 'offsta' or self.cellType == 'onsta':
            self.contrast = 10
            self.time = 9  # approximately 500 s
            self.duration = 500
        else:
            self.contrast = -10
            self.time = 9.24  # approximately 600 s
            self.duration = 300
        self.cell = self.make_cell()
        self.patchNumber = self.cell.shape[0]
        if self.patchNumber == 13:
            self.patchSize = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2., 3, 4.01, 6., 8., 10.]
        elif self.patchNumber == 12:
            self.patchSize = [0.2, 0.4, 0.6, 0.8, 1., 1.5, 2., 3, 4.01, 6., 8., 10.]
        elif self.patchNumber == 10:
            self.patchSize = [0.2, 0.4, 0.6, 0.8, 1., 2., 4.01, 6., 8., 10.]
        else:
            raise Exception('Something wrong with the patch size!')
        self.para = None

    def _get_para(self, *args):
        if args:
            self.para = args
        else:
            return self.para

    def make_array(self, size):
        args = self._get_para()
        paraDict_tp = args[0]
        paraDict_sp = args[1]
        paraDict_weight = args[2]
        network = pylgn.Network()
        integrator = network.create_integrator(nt=self.time, nr=self.fieldSize, dt=1 * pq.ms, dr=0.1 * pq.deg)
        stimulus = pylgn.stimulus.create_flashing_spot(contrast=self.contrast, patch_diameter=size * pq.deg,
                                                       delay=0 * pq.ms, duration=self.duration * pq.ms)
        network.set_stimulus(stimulus, True)
        tp_bi_exc = kernel.temporal.create_biphasic_ft(phase=paraDict_tp['excPhase'] * (1.0 - 0.05 * size) * pq.ms,
                                                       damping=paraDict_tp['excDamping'],
                                                       delay=paraDict_tp['excPhaseDelay'] * (1.0 - 0.02 * size) * pq.ms)
        tp_bi_inh = kernel.temporal.create_biphasic_ft(phase=paraDict_tp['inhPhase'] * (1.0 - 0.05 * size) * pq.ms,
                                                       damping=paraDict_tp['inhDamping'],
                                                       delay=paraDict_tp['inhPhaseDelay'] * (1.0 - 0.02 * size) * pq.ms)

        Wg_r_on = kernel.spatial.create_dog_ft(A=1, a=paraDict_sp['a'] * pq.deg, B=paraDict_sp['B'],
                                           b=paraDict_sp['b'] * pq.deg)
        Wg_r_off = kernel.spatial.create_dog_ft(A=-1, a=paraDict_sp['a'] * pq.deg, B=-paraDict_sp['B'],
                                              b=paraDict_sp['b'] * pq.deg)

        receptNeuron = network.create_ganglion_cell()
        ganglionOn = network.create_relay_cell(background_response=6 / pq.s)

        network.connect(receptNeuron, ganglionOn, (Wg_r_on, tp_bi_exc), weight=1 * size ** 0.8)
        network.connect(receptNeuron, ganglionOn, (Wg_r_off, tp_bi_inh), weight=paraDict_weight['offWeight'] * size ** 0.8)

        receptNeuron.set_kernel((kernel.spatial.create_delta_ft(), kernel.temporal.create_delta_ft()))
        network.compute_response(ganglionOn)

        return ganglionOn.center_response

    def make_matrix(self, *args):
        self._get_para(*args)
        pool = ProcessPoolExecutor(max_workers=8)
        matrix = np.array(list(pool.map(self.make_array, self.patchSize)))
        new_matrix = normalizeMatrix(matrix)

        return new_matrix

    def make_cell(self):
        if self.cellType == 'onsta' or self.cellType == 'offsta':
            cell = self.data['allcell'][self.cellType][0][0][self.cellIndex][0][:, 0:int(2**self.time)]
        else:
            cell = self.data['allcell'][self.cellType][0][0][self.cellIndex][0][:, 500:200+int(2 ** self.time)]

        new_cell = normalizeMatrix(cell)
        return new_cell

    def correlation(self,matrix1,matrix2):
        correlation = 0
        for i in range(0, self.patchNumber):
            s1 = pd.Series(matrix1[i])
            s2 = pd.Series(matrix2[i])
            correlation = correlation + s2.corr(s1)
        correlation = correlation / self.patchNumber
        return correlation

    def make_correlation(self, space):
        tp = space[0]
        sp = space[1]
        w = space[2]
        matrix1 = self.cell
        matrix2 = self.make_matrix(tp, sp, w)
        return -self.correlation(matrix1,matrix2)

    def make_similar1(self, space):
        tp = space[0]
        sp = space[1]
        wp = space[2]
        matrix1 = self.cell
        matrix2 = self.make_matrix(tp, sp, wp)
        if self.cellType == 'onbla' or self.cellType == 'offbla':
            matrix2 = matrix2[:, 300: int(2**self.time)]
        return -mtx_similar1(matrix1, matrix2)

    def make_similar2(self, space):
        tp = space[0]
        sp = space[1]
        wp = space[2]
        matrix1 = self.cell
        matrix2 = self.make_matrix(tp, sp, wp)
        return -mtx_similar2(matrix1, matrix2)

    def make_similar3(self, space):
        tp = space[0]
        sp = space[1]
        wp = space[2]
        matrix1 = self.cell
        matrix2 = self.make_matrix(tp, sp, wp)
        return -mtx_similar3(matrix1, matrix2)

    def plotting(self, best_condition):
        '''
        :param matrix: model result
        :param cell: vivo result
        :return: plotting two graphs
        '''

        tp,sp,w = preprocess(best_condition)
        since =tm.time()
        matrix = self.make_matrix(tp, sp, w)

        if self.cellType == 'onbla' or self.cellType == 'offbla':
            matrix = matrix[:,300: int(2**self.time)]
        now = tm.time() - since
        print('time consume')
        print(now)
        x1, y1 = matrix.shape
        X1, Y1 = np.meshgrid(range(0, y1), range(0, x1))
        x2, y2 = self.cell.shape
        X2, Y2 = np.meshgrid(range(0, y2), range(0, x2))

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.contourf(X2, Y2, self.cell)
        ax2 = fig.add_subplot(122)
        ax2.contourf(X1, Y1, matrix)
        plt.show()

    def parameter_visualization(self,trials):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,5,1)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['excPhase'] for t in trials.trials]
        ax1.set_xlim(xs[0] - 5, xs[-1] + 5)
        ax1.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax1.set_title('$excPhase$ $vs$ $t$ ', fontsize=12)

        ax2 = fig.add_subplot(2,5,2)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['excPhaseDelay'] for t in trials.trials]
        ax2.set_xlim(xs[0] - 5, xs[-1] + 5)
        ax2.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax2.set_title('$excPhaseDelay$ $vs$ $t$ ', fontsize=12)

        ax3 = fig.add_subplot(2,5,3)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['inhPhase'] for t in trials.trials]
        ax3.set_xlim(xs[0] - 5, xs[-1] + 5)
        ax3.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax3.set_title('$inhPhase$ $vs$ $t$ ', fontsize=12)

        ax4 = fig.add_subplot(2,5,4)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['inhPhaseDelay'] for t in trials.trials]
        ax4.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax4.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax4.set_title('$inhPhaseDelay$ $vs$ $t$ ', fontsize=12)

        ax5 = fig.add_subplot(2,5,5)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['excDamping'] for t in trials.trials]
        ax5.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax5.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax5.set_title('$excDamping$ $vs$ $t$ ', fontsize=12)

        ax6 = fig.add_subplot(2,5,6)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['inhDamping'] for t in trials.trials]
        ax6.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax6.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax6.set_title('$inhDamping$ $vs$ $t$ ', fontsize=12)

        ax7 = fig.add_subplot(2,5,7)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['a'] for t in trials.trials]
        ax7.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax7.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax7.set_title('$a$ $vs$ $t$ ', fontsize=12)

        ax8 = fig.add_subplot(2,5,8)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['B'] for t in trials.trials]
        ax8.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax8.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax8.set_title('$B$ $vs$ $t$ ', fontsize=12)

        ax9 = fig.add_subplot(2,5,9)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['b'] for t in trials.trials]
        ax9.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax9.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax9.set_title('$b$ $vs$ $t$ ', fontsize=12)

        ax10 = fig.add_subplot(2,5,10)
        xs = [t['tid'] for t in trials.trials]
        ys = [t['misc']['vals']['offWeight'] for t in trials.trials]
        ax10.set_xlim(xs[0] - 10, xs[-1] + 10)
        ax10.scatter(xs, ys, s=5, linewidth=0.01, alpha=0.75)
        ax10.set_title('$offWeight$ $vs$ $t$ ', fontsize=12)

        plt.show()

    def plot_cell(self):
        x,y = self.cell.shape
        X, Y = np.meshgrid(range(0, y), range(0, x))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(X, Y, self.cell)
        plt.show()

   # def loss_visualizaion