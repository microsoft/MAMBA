# MAMBA #

These codes accompany the paper "Policy Improvement from Multiple Experts" by Ching-An Cheng, Andrey Kolobov, and Alekh Agarwal.

### Run experiments ###
Update `PYTHONPATH` to include the repo, if not already.
```
source start.sh
```
Run a single script.
```
python scripts/mamba.py
```
The configs of the experiments and the hyperparameters can be set by the dict named `CONFIG` in scripts/mamba.py. The default setting runs MAMBA with ADAM and two weak experts (policy_15). Increasing the index number of `CONFIG['expert_info']['name']` uses better experts.

Batch run scripts/mamba.py with a set of different configurations given as a dict named `range_common` in `scripts/mamba_ranges.py`
```
python batch_run.py mamba -r common
```
The experimental results are saved in folders starting with `log_` in the top folder of the repo.


### Installation ###
Tested in buntu 18.04 with python 3.7.

#### Install mamba and most of the requirements ####
Install this repo and requirements:
```
git clone https://github.com/chinganc/mamba.git
pip install --upgrade -r requirements.txt
```
You may need to run
```
export PYTHONPATH="{PYTHONPATH}:[the parent folder of mamba repo]"
```
The current version requires also tensorflow2.
```
pip install --upgrade tensorflow
```

Below we install DartENV that was used in producing the experiments in the paper. Other gym environments can also be used.


#### Install DART ####
The Ubuntu package is too new for PyDart2, so we install it manually.

First install the requirements following the instructions of Install DART from source at https://dartsim.github.io/install_dart_on_ubuntu.html.
Next we compile and install DART manually, because PyDart2 only supports DART before 6.8.
```
git clone git://github.com/dartsim/dart.git
cd dart
git checkout tags/v6.7.2
mkdir build
cd build
cmake ..
make -j4
sudo make install
```
Someitmes the library may need to be linked manually.
```
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib" >> ~/.bashrc
```

#### Install PyDart2 ####
Installing PyDart2 through pip does not work, so we install it manually.
```
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python setup.py build build_ext
python setup.py develop
```


#### Install DartEnv ####
This is a slightly modified version of [DartEnv](https://github.com/DartEnv/dart-env). The changes include:

* Make nodisplay as default.
* Add a state property for convenience.

To install it,
```
git clone https://github.com/gtrll/dartenv.git
cd dartenv
git checkout nodisplay
pip install -e .[dart]
```


# Contributing 

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

