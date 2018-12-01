from hyperopt import hp, fmin, rand, tpe, Trials
import testFunction as tf
import pickle
import time as tm
from twilio.rest import Client

since = tm.time()
paraDict_tp = {'excPhase': hp.uniform('excPhase', 0, 150),
               'excPhaseDelay': hp.uniform('excPhaseDelay', 0, 150),
               'inhPhase': hp.uniform('inhPhase', 0, 150),
               'inhPhaseDelay': hp.uniform('inhPhaseDelay', 0, 150),
               'excDamping': hp.uniform('excDamping', -0.5, 1.5),
               'inhDamping': hp.uniform('inhDamping', -0.5, 1.5)}
paraDict_sp = {'a': hp.uniform('a', 0, 6),
               'B': hp.uniform('B', 0, 10),
               'b': hp.uniform('b', 0, 10)}
paraDict_weight = {'offWeight': hp.uniform('offWeight', 0, 1.5)}

space = [paraDict_tp,
         paraDict_sp,
         paraDict_weight
         ]

trialnum = 3
cellType = 'onsta'
max_evals = 1500
for index in range(62,75):
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




account_sid = 'AC54ddb2edef6333de4a9852e448d334ec'
auth_token = '5e0119419b64214fe5a0afbd39e9a3b1'
client = Client(account_sid, auth_token)

message = client.messages.create(
    from_='+17575449618',
    to='+8616621022480',
    body='your simulation is finished! And it took '+str(now/3600)+' hours.'
)

print(message.sid)

