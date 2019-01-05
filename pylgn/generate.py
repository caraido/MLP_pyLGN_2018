from hyperopt import hp, fmin, rand, tpe, Trials
import testFunction as tf
import pickle
import time as tm
from twilio.rest import Client

since = tm.time()
paraDict_tp = {'excPhase': hp.uniform('excPhase', 0, 200),
               'excPhaseDelay': hp.uniform('excPhaseDelay', 0, 200),
               'inhPhase': hp.uniform('inhPhase', 0, 200),
               'inhPhaseDelay': hp.uniform('inhPhaseDelay', 0, 200),
               'excDamping': hp.uniform('excDamping', -0.5, 2),
               'inhDamping': hp.uniform('inhDamping', -0.5, 2)}
paraDict_sp = {'a': hp.uniform('a', 0, 6),
               'B': hp.uniform('B', 0, 10),
               'b': hp.uniform('b', 0, 10)}
paraDict_weight = {'offWeight': hp.uniform('offWeight', 0, 1.5)}

space = [paraDict_tp,
         paraDict_sp,
         paraDict_weight
         ]

trialnum = 3
cellType = 'offsta'
max_evals = 1000
for index in range(72, 80):
    op = tf.operation(cellType=cellType, cellIndex=index)
    print('generated object number'+str(index))
    mtx_1 = op.make_similar1

    trials = Trials()
    best = fmin(mtx_1, space, tpe.suggest, max_evals=max_evals, trials=trials)
    file1 = open('/home/aistation/Documents/pycharm/pylgn/data/trial' + str(trialnum) + '_method_mtx1_trial_' + str(
        cellType) + '_index_' + str(index) + '_loop_' + str(max_evals) + '.pk', 'wb')
    file2 = open('/home/aistation/Documents/pycharm/pylgn/data/trial' + str(trialnum) + '_method_mtx1_best_' + str(
        cellType) + '_index_' + str(index) + '_loop_' + str(max_evals) + '.pk', 'wb')
    print('finished running')
    pickle.dump(trials, file1)
    pickle.dump(best, file2)
    print('dumped cell number' + str(index))

now = tm.time()-since
print('time consume')
print(now)

