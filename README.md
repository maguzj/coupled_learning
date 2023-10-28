# coupled_learning
## Setup

1. Clone a local copy of the repository:

```bash
git clone git@github.com:maguzj/coupled_learning.git
```

2. Create the environment.

**For Intel chips:** The main directory contains an environment.yml file for easily setting up a conda environment, named cl, with all the package dependencies:

```bash
conda env create --file=environment-intel.yml
```

To activate the environment, run

```bash
conda activate cl
```

**For M1/M2 chips:** We have to build numpy with the accelerator.

```bash
conda env create --file=environment-M1-M2.yml
```

Activate the environment, install numpy using pip and set the pip to be recognized by further package installations:

```bash
conda activate cl
```
for numpy version 1.25.2 or older, then run

```bash
pip install --no-binary :all: --no-use-pep517 numpy
```
It seems that for newer versions --no-use-pep517 is deprecated. Instead, run
```bash
pip install --no-binary :all: numpy
```

Once numpy installed, run

```bash
conda config --set pip_interop_enabled true
```

check that numpy is using vecLib:

```bash
>>> import numpy
>>> numpy.show_config()
```

If everything is right, you should see info like ```/System/Library/Frameworks/vecLib.framework/Headers``` printed.

Then install the higher level dependencies.

```bash
pip install -r requirements.txt
```

(for more information see: https://gist.github.com/MarkDana/a9481b8134cf38a556cf23e1e815dafb)


3. Several jupyter notebooks are provided for getting started.



## TO DO LIST:

- Implement conversion toJAX
- Implement efficient prunning, back compatible with labeling and indices.
- Implement plotting routines, using Ax as arguments.
- Implement Read the Docs
- Implement triangular networks
