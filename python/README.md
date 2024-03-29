# Python

Here all the code related to python technology will be included.

For example:

* models done in `pulp`.
* data wrangling scripts done with `pandas`.

Code is organized as follows:

* **scripts**: python scripts done for executing model, heuristics, etc.
* **package**: core code with models, data processing and main logic. Objects for data input and output.
* **tests**: unit tests for specific parts of the heuristic.
* **examples**: data for unit tests or to produce example images.

## Install dependencies

### Ubuntu:

    sudo apt-get install python3 pip git python3-venv python3-dev

### Windows

This requires to have `choco` installed.

    choco install python3 git pip -y

Alternatively, one can manually download the latest python version and git. `pip` usually comes with python already.

## Get the software and build it

Steps to set up development environment:

Windows: Same as Linux but just change the forth line for: `venv\Scripts\activate`

Linux:

    git clone git@github.com:pchtsp/ROADEF2018.git
    cd ROADEF2018/python
    python -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    python setup.py build_ext --inplace

## Additional instructions for installing requirements in Windows:

(This is only necessary if not using anaconda and having problems installing some packages).

Check: https://stackoverflow.com/a/32064281

* Build Tools 2017: http://landinghub.visualstudio.com/visual-cpp-build-tools
* numpy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
* Scipy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
* cx_freeze in github version, not pip.
* specific configuration for windows?

## Test it's working

I Ubuntu you'd do something like the following:

    cd ROADEF2018/python
    mkdir result
    source venv/bin/activate
    python scripts/exec.py --case-name A1 --path-root /PATH/TO/PROJECT/ROADEF2018/ --path-results /PATH/TO/PROJECT/ROADEF2018/python/results/ --results-dir test_experiment1 --time-limit 10

An alternative way to use it, the challenge standard, is the following:

    cd ROADEF2018/python
    mkdir results
    source venv/bin/activate
    python scripts/exec.py -p INSTANCES_LOCATION/A1 -o PATH_TO_SOLUTIONS/A1_solution.csv -t 60

This will solve the case A1 for 10 seconds and store the result and the log in the following directory: `/PATH/TO/PROJECT/ROADEF2018/python/results/test_experiment1/`. It's important to add the `/` at the end of the arguments!

## Configuring the application

There is a file called `python/package/params.py` that stores the default configuration. It's possible to change this directly. Another option is to create another file similar to this one and edit it. Then this file needs to be given as an argument to the `exec.py` script so it can import it instead of the default one. The argument is called `--config-file`

Another option is to give the options as arguments. This is not possible for all options but the most used ones. For a list of all options, run the following:

    cd ROADEF2018/python
    source venv/bin/activate
    python scripts/exec.py -h 

## Creating executable

I'm using PyInstaller, from the following url: https://docs.python-guide.org/shipping/freezing/

In order to create a bundle, the instructions (work in windows as well as linux):

    cd ROADEF2018/python
    source venv/bin/activate
    pyinstaller -y challengeSG.spec

In windows just change the second line for: `venv\Scripts\activate`

## Parameter optimization

There was an unsuccessful attempt to use `irace` package in R to do the parameter optimization. The steps to run it are the following (only work in Linux). Some libraries need to be installed first, (irace is one). The path to irace needs to be checked using the file `getPath_irace.R`

In R console:

    install.packages('irace')
    library("irace")
    PATH_TO_IRACE_PACKAGE <- system.file(package = "irace")

In bash:

    cd R/ROADEF-R/tuning
    PATH_TO_IRACE_PACKAGE/bin/irace
