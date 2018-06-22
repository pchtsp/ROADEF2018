import pstats
p = pstats.Stats('scripts/restats')
p.strip_dirs().sort_stats(-1)
p.sort_stats('tottime').print_stats(20)  # tottime, cumulative



# python -m cProfile -o scripts/restats scripts/exec.py