# -*- mode: python -*-

block_cipher = None

def get_pandas_path():
    import pandas
    pandas_path = pandas.__path__[0]
    return pandas_path

def get_palettable_path():
    import palettable
    return palettable.__path__[0]


a = Analysis(['scripts/exec.py'],
             pathex=['C:/Users/pchtsp/Documents/projects/ROADEF2018/python'],
             binaries=[],
             datas=[('../resources/data/dataset_A', 'resources/data/dataset_A')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['pandas.tests'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

dict_tree = Tree(get_pandas_path(), prefix='pandas', excludes=["*.pyc", "tests"])
a.datas += dict_tree
a.datas += Tree(get_palettable_path(), prefix='palettable', excludes=["*.pyc"])
a.binaries = filter(lambda x: 'pandas' not in x[0], a.binaries)
a.binaries = [x for x in a.binaries if not x[0].startswith("IPython")]
a.binaries = [x for x in a.binaries if not x[0].startswith("zmq")]
a.binaries = [x for x in a.binaries if not x[0].startswith("PyQt5")]
a.binaries = [x for x in a.binaries if not x[0].startswith("Qt5")]
a.binaries = a.binaries - TOC([
 #('sqlite3.dll', None, None),
 ('tcl85.dll', None, None),
 ('tk85.dll', None, None),
 #('_sqlite3', None, None),
 ('_ssl', None, None),
 #('_tkinter', None, None)
 ])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# for several files, keep the following two and comment the other one.
exe = EXE(pyz, a.scripts, exclude_binaries=True, name='challengeSG', debug=False, strip=False, upx=True, console=True)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=True, name='challengeSG')

# for one exe, replace the two above for.
# exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, name='challengeSG', debug=False, strip=False, upx=True, runtime_tmpdir=None, console=True)

# pyinstaller -y challengeSG.spec