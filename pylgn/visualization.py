import pickle
import sys
sys.path.append('/home/aistation/Documents/pycharm/pylgn/data/')
import testFunction as tf
import time as tm

cellType = 'onbla'
cellIndex = 11

f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_method_mtx1_trial_'+cellType+'_index_'+str(cellIndex)+'_loop_1000.pk','rb')
trials = pickle.load(f)

g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_method_mtx1_best_'+cellType+'_index_'+str(cellIndex)+'_loop_1000.pk','rb')
best = pickle.load(g)


op = tf.operation(cellIndex=cellIndex, cellType='onbla')

#op.plot_cell()

op.plotting(best)
op.parameter_visualization(trials)

