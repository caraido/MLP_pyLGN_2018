import pickle
import matplotlib.pyplot as plt
import numpy as np

best = []
trials=[]
case = 'parameter'
for cellIndex in range(0, 25):
    g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_offsta/trial3_method_mtx1_best_offsta_index_'+str(cellIndex)+'_loop_1000.pk','rb')
    f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_offsta/trial3_method_mtx1_trial_offsta_index_' + str(cellIndex) + '_loop_1000.pk', 'rb')
    trials.append(pickle.load(f).trials)
    best.append(pickle.load(g))

for cellIndex in range(25, 75):
    g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_offsta/trial3_method_mtx1_best_offsta_index_' + str(cellIndex) + '_loop_1500.pk', 'rb')
    f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_offsta/trial3_method_mtx1_trial_offsta_index_' + str(cellIndex) + '_loop_1500.pk', 'rb')
    trials.append(pickle.load(f).trials)
    best.append(pickle.load(g))

excPhase = [best[i]['excPhase'] for i in range(0, 75)]
excPhaseDelay = [best[i]['excPhaseDelay'] for i in range(0, 75)]
inhPhase = [best[i]['inhPhase'] for i in range(0, 75)]
inhPhaseDelay = [best[i]['inhPhaseDelay'] for i in range(0, 75)]
excDamping = [best[i]['excDamping'] for i in range(0, 75)]
inhDamping = [best[i]['inhDamping'] for i in range(0, 75)]
a = [best[i]['a'] for i in range(0, 75)]
B = [best[i]['B'] for i in range(0, 75)]
b = [best[i]['b'] for i in range(0, 75)]
offWeight = [best[i]['offWeight'] for i in range(0, 75)]

#result = [-trials[i]['result']['loss'] for i in range(0, 75)]
trial = [i for i in range(0, 75)]


#print(result)

if case == 'parameter':
    fig = plt.figure()
    ax1 = fig.add_subplot(2,5,1)
    xs = trial
    ys = excPhase
    ax1.set_xlim(xs[0] -5, xs[-1] + 5)
    ax1.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax1.set_title('$excPhase$ $vs$ $t$ ', fontsize=12)

    ax2= fig.add_subplot(2,5,2)
    xs = trial
    ys = excPhaseDelay
    ax2.set_xlim(xs[0] -5, xs[-1] + 5)
    ax2.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax2.set_title('$excPhaseDelay$ $vs$ $t$ ', fontsize=12)

    ax3 = fig.add_subplot(2,5,3)
    xs = trial
    ys = inhPhase
    ax3.set_xlim(xs[0] -5, xs[-1] + 5)
    ax3.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax3.set_title('$inhPhase$ $vs$ $t$ ', fontsize=12)

    ax4 = fig.add_subplot(2,5,4)
    xs = trial
    ys = inhPhaseDelay
    ax4.set_xlim(xs[0] -5, xs[-1] + 5)
    ax4.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax4.set_title('$inhPhaseDelay$ $vs$ $t$ ', fontsize=12)

    ax5 = fig.add_subplot(2,5,5)
    xs = trial
    ys = excDamping
    ax5.set_xlim(xs[0] -5, xs[-1] + 5)
    ax5.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax5.set_title('$excDamping$ $vs$ $t$ ', fontsize=12)

    ax6 = fig.add_subplot(2,5,6)
    xs = trial
    ys = inhDamping
    ax6.set_xlim(xs[0] -5, xs[-1] + 5)
    ax6.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax6.set_title('$inhDamping$ $vs$ $t$ ', fontsize=12)

    ax7 = fig.add_subplot(2,5,7)
    xs = trial
    ys = a
    ax7.set_xlim(xs[0] -5, xs[-1] + 5)
    ax7.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax7.set_title('$a$ $vs$ $t$ ', fontsize=12)

    ax8 = fig.add_subplot(2,5,8)
    xs = trial
    ys = B
    ax8.set_xlim(xs[0] -5, xs[-1] + 5)
    ax8.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax8.set_title('$B$ $vs$ $t$ ', fontsize=12)

    ax9 = fig.add_subplot(2,5,9)
    xs = trial
    ys = b
    ax9.set_xlim(xs[0] -5, xs[-1] + 5)
    ax9.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax9.set_title('$b$ $vs$ $t$ ', fontsize=12)

    ax10 = fig.add_subplot(2,5,10)
    xs = trial
    ys = offWeight
    ax10.set_xlim(xs[0] -5, xs[-1] + 5)
    ax10.scatter(xs, ys, s=10, linewidth=0.01, alpha=0.75)
    ax10.set_title('$offWeight$ $vs$ $t$ ', fontsize=12)

    plt.show()