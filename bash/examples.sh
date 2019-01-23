python3 Documents/projects/ROADEF2018/python/scripts/exec.py --all-cases --path-root /home/pchtsp/Documents/projects/ROADEF2018/ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20180916_venv --no-graph --extra-jumbos 0 --time-limit 600 -s 500&

python3 Documents/projects/ROADEF2018/python/scripts/exec.py --case-name A14 --path-root /home/pchtsp/Documents/projects/ROADEF2018/ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20180916_venv --no-graph --extra-jumbos 0 --time-limit 600 --num-process 1 --heur-remake '{"iterations_initial": 1000, "iterations_remake": 100}'

python3 Documents/projects/ROADEF2018/python/scripts/exec.py --all-cases --path-root /home/disc/f.peschiera/Documents/projects/ROADEF2018/ --path-results /home/disc/f.peschiera/Documents/projects/ROADEF2018/results/ --results-dir prise_20180918_venv --no-graph --extra-jumbos 0 --time-limit 3600 -s 500 --num-process 12 --heur-remake '{"iterations_initial": 1000, "iterations_remake": 100}' --heur-params '{"cooling_rate": 0.005}' & 

python scripts/exec.py --all-cases --path-root ./../ --path-results ./../results/ --results-dir test -t 180 & 

python scripts/exec.py -p ./../resources/data/dataset_A/A3 -o ./../resources/checker/instances_checker/A3_solution.csv --time-limit 60 -s 500 --num-process 12 --heur-remake "{\"iterations_initial\": 10, \"iterations_remake\": 10, \"probability\": 0}" --heur-params "{\"cooling_rate\": 0.005}"

python scripts/exec.py --all-cases --path-root ./../ --path-results ./../results/ --results-dir hp_20181123 --time-limit 3600 & 

python python/scripts/exec.py --all-cases --path-root ./ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20181126 --time-limit 3600 > log.txt &

 python python/scripts/exec.py --all-cases --path-root ./ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20181210 --time-limit 3600 --data-set B > log.txt &

python python/scripts/exec.py --all-cases --path-root ./ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20181212 --time-limit 3600 --data-set A --heur-remake "{\"iterations_initial\": 1000, \"iterations_remake\": 100, \"num_trees\": [0.20, 0.20, 0.20, 0.20, 0.20]}" --heur-params  "{\"rotation_probs\": [0.5, 0.5, 0, 0]}" > log.txt & 

nohup python python/scripts/exec.py --all-cases --path-root ./ --path-results /home/pchtsp/Dropbox/ROADEF2018/ --results-dir hp_20190117 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > log.txt &

nohup python python/scripts/exec.py --all-cases --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190122 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

nohup python python/scripts/exec.py --all-cases --data-set B --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

# tune
cd R/ROADEF-R/tuning
/home/pchtsp/R/x86_64-pc-linux-gnu-library/3.4/irace/bin/irace

# compile:
cd python
source venv/bin/activate
python setup.py build_ext --inplace

# bundle:
cd python
source venv/bin/activate
pyinstaller -y challengeSG.spec

# zip it
cd python
7za a challenge.7z dist

# move things
scp pchtsp@port-peschiera:/home/pchtsp/Documents/projects/ROADEF2018/python/challenge_20190122.7z ./
scp f.peschiera@serv-cluster1:/tmp/f.peschiera/roadef2018/clust_20190122 ./