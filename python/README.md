# Python

Here all the code related to python technology will be included.

For example:

* models done in `pulp`.
* data wrangling scripts done with `pandas`.

Code is organized as follows:

* **scripts**: python scripts done for executing model, heuristics, etc.
* **package**: core code with models, data processing and main logic. Objects for data input and output.

## Install dependencies

### Ubuntu:

    sudo apt-get install python3 pip git python3-venv python3-devel

### Windows

This requires to have `choco` installed.

    choco install python3 git pip -y

Alternatively, one can manually download the latest python version and git. `pip` usually comes with python already.

## Get the software

Steps to set up development environment:

Windows:

    git clone git@github.com:pchtsp/ROADEF2018.git
    cd ROADEF2018\python
    python3 -m venv venv
    venv\Scripts\activate
    pip3 install -r requirements.txt

Ubuntu:

    git clone git@github.com:pchtsp/ROADEF2018.git
    cd ROADEF2018/python
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

## Test it's working

I Ubuntu you'd do something like the following:

    cd ROADEF2018/python
    mkdir results
    source venv/bin/activate
    python3 scripts/exec.py --case-name A1 --path-root /PATH/TO/PROJECT/ROADEF2018/ --path-results /PATH/TO/PROJECT/ROADEF2018/python/results/ --results-dir test_experiment1 --time-limit 10

This will solve the case A1 for 10 seconds and store the result and the log in the following directory: `/PATH/TO/PROJECT/ROADEF2018/python/results/test_experiment1/`. It's important to add the `/` at the end of the arguments!

## Configuring the application

There is a file called `python/package/params.py` that stores the default configuration. It's possible to change this directly. Another option is to create another file similar to this one and edit it. Then this file needs to be given as an argument to the `exec.py` script so it can import it instead of the default one. The argument is called `--config-file`

Another option is to give the options as arguments. This is not possible for all options but the most used ones. For a list of all options, run the following:

    cd ROADEF2018/python
    source venv/bin/activate
    python scripts/exec.py -h 

## Additional instructions for installing requirements in Windows:

(This is only necessary if not using anaconda and having problems installing some packages).

Check: https://stackoverflow.com/a/32064281

* Build Tools 2017: http://landinghub.visualstudio.com/visual-cpp-build-tools
* numpy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
* Scipy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
* cx_freeze in github version, not pip.
* specific configuration for windows?

## Configuration

It's important to edit the params.py file to get the absolute path correctly. We could potentially solve this in the future by using relative paths correctly...

## Use pypy in Ubuntu

An alternative to cpython is to use pypy which is a ported version of python for the JIT compiler, used for java and other languages. It's usually faster. What I had to do to get the virtual environment of this distribution is the following.

I followed http://doc.pypy.org/en/latest/install.html and https://askubuntu.com/questions/441981/how-to-install-pypy3-on-ubuntu-for-nebies.

For distributions that are not Ubuntu: https://github.com/squeaky-pl/portable-pypy#portable-pypy-distribution-for-linux

<!-- pip install git+https://bitbucket.org/pypy/numpy.git -->

    virtualenv -p /path/to/pypy/tarball/pypy3-v6.0.0-linux64/bin/pypy3 venv_pypy
    source venv_pypy/bin/activate
    pip install -r requirements.txt
