import pickle
import matplotlib.pyplot as plt
import pandas as pd


#case = 'offsta'
paraSpace = ('excPhase', 'excPhaseDelay', 'inhPhase', 'inhPhaseDelay', 'excDamping', 'inhDamping', 'a', 'B', 'b',
			 'offWeight')

'''
data preprocess
'''

def preprocess(case, parameter = paraSpace):
	'''
	case should be a string
	input: a case
	output: 1, a dictionary of parameter indices and its list of trials ( {"parameter": trials } )
			2, a list of trial indices (1~1000 or 1~1500)
	'''

	scope = {}
	best = []
	trials = []
	if case == 'offsta':

		index = 75
		scope.update(index=75)
		for cellIndex in range(0,25):
			g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_best_' + case + '_index_' + str(cellIndex) + '_loop_1000.pk' , 'rb')
			f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_trial_' + case +'_index_' + str(cellIndex) + '_loop_1000.pk' , 'rb')
			trials.append(pickle.load(f).trials)
			best.append(pickle.load(g))

		for cellIndex in range(25,index):
			g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_best_' + case + '_index_' + str(cellIndex) + '_loop_1500.pk' , 'rb')
			f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_trial_' + case +'_index_' + str(cellIndex) + '_loop_1500.pk' , 'rb')
			trials.append(pickle.load(f).trials)
			best.append(pickle.load(g))

	elif case == 'offbla':

		index = 75
		scope.update(index=75)
		for cellIndex in range(0,index):
			g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_best_' + case + '_index_' + str(cellIndex) + '_loop_1000.pk' , 'rb')
			f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_trial_' + case +'_index_' + str(cellIndex) + '_loop_1000.pk' , 'rb')
			trials.append(pickle.load(f).trials)
			best.append(pickle.load(g))

	elif case == 'onbla':

		index = 93
		scope.update(index=93)
		for cellIndex in range(0, index):
			g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_best_' + case + '_index_' + str(cellIndex) + '_loop_1000.pk' , 'rb')
			f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_trial_' + case +'_index_' + str(cellIndex) + '_loop_1000.pk' , 'rb')
			trials.append(pickle.load(f).trials)
			best.append(pickle.load(g))

	elif case == 'onsta':

		index = 93
		scope.update(index=93)
		for cellIndex in range(0, index):
			g = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_best_' + case + '_index_' + str(cellIndex) + '_loop_1500.pk' , 'rb')
			f = open('/home/aistation/Documents/pycharm/pylgn/data/trial3_'+ case +'/trial3_method_mtx1_trial_' + case +'_index_' + str(cellIndex) + '_loop_1500.pk' , 'rb')
			trials.append(pickle.load(f).trials)
			best.append(pickle.load(g))

	else:
		raise Exception("invalid case!")

	scope.update(best = best)
	scope.update(paraBook = {})
	for item in parameter:
		exec(item + " = [best[i]['" + item + "'] for i in range(0,index)]", scope)
		# exec("print ("+item+")",scope)
		exec("paraBook.update("+item+" = " + item + ")", scope)

	trial = [i for i in range(0, index)]
	return scope['paraBook'], trial

offsta, trial_offsta = preprocess(case= 'offsta')
onsta,trial_onsta = preprocess(case= 'onsta')
offbla, trial_offbla = preprocess(case= 'offbla')
onbla,trial_onbla = preprocess(case= 'onbla')



'''
data analysis
'''
offsta = pd.DataFrame(offsta)
onsta = pd.DataFrame(onsta)
offbla = pd.DataFrame(offbla)
onbla = pd.DataFrame(onbla)

d1 = offsta.mean()
d2 = onsta.mean()
d3 = offbla.mean()
d4 = onbla.mean()



'''
data plotting
'''
'''
i = 1
fig = plt.figure()
for item in scope['paraBook']:
	ax = fig.add_subplot(2,5,i)
	xs = trial
	exec('ys = ' +item, scope)
	ax.set_xlim(xs[0] -5, xs[-1]+ 5)
	ax.scatter(xs, scope['ys'], s=5, linewidth=0.01, alpha=0.75)
	ax.set_title('$'+item+'$ $vs$ $t$', fontsize=10)
	i = i+1

plt.show()	
'''

