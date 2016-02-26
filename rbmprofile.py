"""
Get line profile of RBM module
"""

import line_profiler as p
# import loose_fun as lf
import RBM_m as Rm
# import fig_gen as fg
import pickle

profiler = p.LineProfiler()                 # Create LineProfiler object

# profiler.add_module(lf)                     # Add modules used for RBM
profiler.add_module(Rm)
# profiler.add_module(fg)

# profiler.run('ibc = Rm.RBM("ibc")')         # Run RBM through profiler
# profiler.run('ibc.training1(epochs=10)')
ibc = pickle.load(open('ibc_tr1_complete_added_classif', 'rb'))
# profiler.run('ibc.reproductions()')
# profiler.run('ibc.addclassifiers()')
profiler.run('ibc.training2(epochs=10)')
profiler.run('ibc.testing()')

# profiler.run("fg.generatefigures('training2')")
# profiler.run("fg.generatefigures('testing')")

profiler.print_stats()                      # Print results in terminal
