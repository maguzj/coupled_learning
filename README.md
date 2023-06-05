# coupled_learning
## Setup

1. Clone a local copy of the repository:

```bash
git clone git@github.com:maguzj/coupled_learning.git
```

2. The main directory contains an environment.yml file for easily setting up a conda environment, named cl, with all the package dependencies:
(For M1/M2 computers see next section)

```bash
conda env create --file=environment.yml
```

To activate environment, run

```bash
conda activate cl
```

2.1.  For M1/M2 computers only. We have to build numpy with the accelerator.

```bash
conda env create --file=environment-M1-M2.yml
```

Activate the environment, install numpy using pip and set the pip to be recognized by further package installations:

```bash
conda activate cl
pip install --no-binary :all: -no-use-pep157 numpy
conda config --set pip_interop_enabled true
```

check that numpy is using vecLib:

```bash
>>> import numpy
>>> numpy.show_config()
```

If everything is right, you should see info like ```/System/Library/Frameworks/vecLib.framework/Headers``` printed

Then install the higher level dependencies.

```bash
conda activate cl
pip install -r requirements.txt
```


3. Several jupyter notebooks are provided for getting started.