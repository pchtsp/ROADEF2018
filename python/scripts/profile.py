import pstats
p = pstats.Stats('scripts/restats')
p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats('cumulative').print_stats(10)


# python -m cProfile -o scripts/restats scripts/exec.py