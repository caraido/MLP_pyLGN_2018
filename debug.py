import sys
sys.path.append('C:/Users/pc/Desktop/pylgn/pylgn')
import testFunction as tf

paraDict_tp = {'excPhase': 50., 'excPhaseDelay': 20., 'inhPhase': 80., 'inhPhaseDelay': 40., 'excDamping': 0.4, 'inhDamping': 0.8}
paraDict_sp = {'a': 0.62, 'B': 0.85, 'b': 1.26}
paraDict_weight = {'offWeight': 0.5}

paraSpace = {}
for paraDict_tp['excPhase'] in [40,50,60]:
    for paraDict_tp['inhPhase'] in [30,40,50]:
        for paraDict_tp['excDamping'] in [0.3,0.4,0.5]:
            for paraDict_tp['inhDamping'] in [0.7,0.8,0.9]:
                t = tf.operation()
                cell = t.make_cell(1)
                matrix = t.make_matrix(paraDict_tp,paraDict_sp,paraDict_weight)
                R2 = tf.RSquare(cell, matrix)
                corr = tf.correlation(cell, matrix)
                paraSpace