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

# remake case A results
nohup python python/scripts/exec.py --case-name A10 A11 A13 A14 A15 A18 A2 A7 A8 --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123_2 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

# reremake case A results
nohup python python/scripts/exec.py --case-name A11 A13 A14 A15 A18 A2 A7 A8 --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123_2 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

# rereremake case A results
nohup python python/scripts/exec.py --case-name A11 --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123_2 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

nohup python python/scripts/exec.py --case-name A2 A7 A8 --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123_2 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

nohup python python/scripts/exec.py --case-name A8 --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123_2 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log_A8.txt &

nohup python python/scripts/exec.py --case-name A7 --data-set A --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190123_2 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

nohup python python/scripts/exec.py --all-cases --data-set B --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190124 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log.txt &

nohup python python/scripts/exec.py --case-name B1 B2 B3 --data-set B --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190124 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log_1.txt &

nohup python python/scripts/exec.py --case-name B4 B5 B6 B7 --data-set B --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190124 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log_2.txt &

nohup python python/scripts/exec.py --case-name B8 B9 B10 B11 --data-set B --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190124 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\", \"num_processors\": 24}" > /tmp/f.peschiera/roadef2018/log_3.txt &

nohup python python/scripts/exec.py --case-name B12 B13 B14 B15 --data-set B --path-root ./ --path-results /tmp/f.peschiera/roadef2018/ --results-dir clust_20190124 --main-param "{\"timeLimit\": 3600, \"multiprocess\": \"True\"}" > /tmp/f.peschiera/roadef2018/log_4.txt &

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
scp -r f.peschiera@serv-cluster1:/tmp/f.peschiera/roadef2018/clust_20190123_2 ./

scp -r f.peschiera@serv-cluster1:/tmp/f.peschiera/roadef2018/clust_20190123_2/A2 ./clust_20190123_2
scp -r f.peschiera@prise-srv3:/tmp/f.peschiera/roadef2018/clust_20190123_2/A11 ./clust_20190123_2
