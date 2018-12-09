python3 Documents/projects/ROADEF2018/python/scripts/exec.py --all-cases --path-root /home/pchtsp/Documents/projects/ROADEF2018/ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20180916_venv --no-graph --extra-jumbos 0 --time-limit 600 -s 500&

python3 Documents/projects/ROADEF2018/python/scripts/exec.py --case-name A14 --path-root /home/pchtsp/Documents/projects/ROADEF2018/ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20180916_venv --no-graph --extra-jumbos 0 --time-limit 600 --num-process 1 --heur-remake '{"iterations_initial": 1000, "iterations_remake": 100}'

python3 Documents/projects/ROADEF2018/python/scripts/exec.py --all-cases --path-root /home/disc/f.peschiera/Documents/projects/ROADEF2018/ --path-results /home/disc/f.peschiera/Documents/projects/ROADEF2018/results/ --results-dir prise_20180918_venv --no-graph --extra-jumbos 0 --time-limit 3600 -s 500 --num-process 12 --heur-remake '{"iterations_initial": 1000, "iterations_remake": 100}' '{"cooling_rate": 0.005}' & 

python scripts/exec.py --all-cases --path-root ./../ --path-results ./../results/ --results-dir test -t 180 & 

python scripts/exec.py -p ./../resources/data/dataset_A/A3 -o ./../resources/checker/instances_checker/A3_solution.csv --time-limit 60 -s 500 --num-process 12 --heur-remake "{\"iterations_initial\": 10, \"iterations_remake\": 10, \"probability\": 0}" --heur-params "{\"cooling_rate\": 0.005}"

python scripts/exec.py --all-cases --path-root ./../ --path-results ./../results/ --results-dir hp_20181123 --time-limit 3600 & 

python python/scripts/exec.py --all-cases --path-root ./ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20181126 --time-limit 3600 & > log.txt