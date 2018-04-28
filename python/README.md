# Python

Here all the code related to python technology will be included.

For example:

* models done in `pulp`.
* data wrangling scripts done with `pandas`.

Code is organized as follows:

* **scripts**: python scripts done for executing model, heuristics, etc.
* **package**: core code with models, data processing and main logic. Objects for data input and output.

Requirements:

* python >= 3.5
* git

## Install dependencies

### Ubuntu:

    sudo apt-get install python3 pip git python3-venv

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
    venv\Scripts\bin\activate
    pip3 install -r requirements

Ubuntu:

    git clone git@github.com:pchtsp/ROADEF2018.git
    cd ROADEF2018/python
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements


## For installing requirements in Windows:

(This is only necessary if not using anaconda and having problems installing some packages).

Check: https://stackoverflow.com/a/32064281

* Build Tools 2017: http://landinghub.visualstudio.com/visual-cpp-build-tools
* numpy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
* Scipy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
* cx_freeze in github version, not pip.
* specific configuration for windows?

## Configuration

It's important to edit the params.py file to get the absolute path correctly. We could potentially solve this in the future by using relative paths correctly...