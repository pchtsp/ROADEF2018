# Python

Here all the code related to python technology will be included.

For example:

* models done in `pulp`.
* data wrangling scripts done with `pandas`.
* applications done with `flask`.

Code is organized as follows:

* **scripts**: python scripts done for data wrangling.
* **package**: core code with models, data proessing and main logic. Objects for data input and output.
* **desktop_app**: PyQt gui for app.

Steps to set up development environment:

* git clone https://github.com/pchtsp/ROADEF2018
* cd ROADEF2018
* virtualenv venv
* `venv\Scripts\bin\activate` or `source venv/bin/activate`
* pip install -r requirements

Requirements:

* python >= 3.5
* pip install virtualenv
* git
* R
* gurobipy => install manually.

## Ubuntu:

(not tested)

    sudo apt-get install python3 r-core pip git r-base
    pip install virtualenv --user

## Windows

(not tested)

    choco install python3 git r.project pip -y
    pip install virtualenv --user

For installing requirements in Windows:

Check: https://stackoverflow.com/a/32064281

* Build Tools 2017: http://landinghub.visualstudio.com/visual-cpp-build-tools
* numpy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
* Scipy from wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
* cx_freeze in github version, not pip.
* specific configuration for windows?

## Configuration

It's important to edit the params.py file to get the absolute path correctly. We could potentially solve this in the future by using relative paths correctly...