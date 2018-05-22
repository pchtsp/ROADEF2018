import pstats
p = pstats.Stats('scripts/restats')
p.strip_dirs().sort_stats(-1)
p.sort_stats('cumulative').print_stats(20)  # tottime



# python -m cProfile -o scripts/restats scripts/exec.py