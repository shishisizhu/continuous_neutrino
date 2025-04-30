# Artifact of Neutrino [OSDI'25]

## Abstract

The artifact of NEUTRINO is hosted at [GitHub](https://github.com/neutrino-gpu/neutrino/tree/artifact) (in branch `artifact`), containing the source code, 
installation/collection/analysis scripts, collected traces that reproduce all the evaluation results in our paper. 
We also package the artifact evaluation as Jupyter Notebooks hosted on Google Colab, offering one-click results reproduction without local runtime setup. 
In addition, we also maintain an [online documentation](https://neutrino-gpu.github.io/) of Neutrino containing project highlights, user guides, roadmaps, and references for evaluating the functionality and reusability. 

**Artifact Claim**: 
The collected traces and the codes are identical to our paper's corresponding description. 
You can replicate all the major results using the traces and analysis codes we provided (details in the _Expected Results_ section below). 
We also provide the trace collection code for you to collect your own traces on your own devices. 
It's worth noticing that customized traces, particularly DMAT, can only yield different but similar results due to hardware and runtime dynamics. 

## Scope (Meta-Information)

* **Design**: Neutrino is a GPU assembly probing tool designed to _attach small code snippets (probes)_ to _GPU kernels at runtime_ to expose runtime execution details (profiling). 
* **System**: Neutrino system consists of two parts, the probe engine to attach code snippets, and the hook driver to capture GPU kernels launched at runtime. The source code is available at [GitHub](https://github.com/neutrino-gpu/neutrino/tree/artifact) and is installable as a Python package.
* **Probes**: Neutrino probes are small TOML files that define the profiling task via snippet, datamodel, position, and callback. All probes used in the paper are also available at [GitHub](https://github.com/neutrino-gpu/neutrino/tree/artifact/neutrino/tools).
* **Output**: Fig. 1, Fig. 10 (A/B/C), Fig. 11, Fig. 12, Fig. 13, and Table. 2 in the paper.
* **Evaluations**: We arrange evaluations in notebooks structured linearly, allowing simple click `Runtime -> Run All` execution. Please refer to the [README](https://github.com/neutrino-gpu/neutrino/blob/artifact/README.md), and instructions in each Jupyter Notebook (Colab). 
* **Special Requirements**: No special requirements for static trace analysis, For dynamic trace collection, a NVIDIA GPU, e.g., A100, and a PTX-included build of PyTorch v2.5.0  and CUTLASS v3.5.0 are required. 
* **Disk Space Requirements**: Evaluating on Google Colab doesn't require any disk space. Regarding local evaluation, please arrange 3GB for static traces and at least 10GB for collecting dynamic traces.  
* **Experiment Time**: Less than 30 minutes for static evaluations analyzing collected traces on CPU, and ≈ 7 hours for dynamic evaluation collecting traces on GPU. 
* **Environment Setup Time**: For **_static_** evaluation, it takes ≈ 2 minutes to download traces. For **_dynamic_** evaluation, it takes ≈ 15 seconds to build NEUTRINO. Setting up PyTorch and CUTLASS might take ≈ 3 minutes.
* **Publicly Available**: Yes. 
* **Code License**: We use the Apache License, Version 2.0 for the system source code. 
* **Probe Licenses**: We use the CC BY 4.0 license for probes used in the paper. 

## Contents

NEUTRINO’s artifact evaluation is arranged in 6 parts, corresponding to different figures or tables in the paper:

* **block_sched**: Sec. 4.5
* **dmat**: Fig. 1, Fig. 10A, Fig. 10B, Fig. 10C.
* **kernel_overhead**: Table. 2.
* **max_mem**: Fig. 11.
* **exposed_latency**: Fig. 12.
* **warp_sched**: Fig. 13.

We arrange each part to correspond to a section in the Jupyter Notebook.
Moreover, each evaluation is provided in two modes: 
the **static** that parses collected traces, suitable for _Getting Started_ on local CPU-only devices without special hardware/software requirements, 
and the **dynamic** that collects the traces on the real GPU-enabled environment, suitable for _Full Evaluation_.

## Hosting and Requirements

### How to Access

Please choose one of the following to access the artifact:

* **GitHub**:
  1. **Static** evaluation notebook: [`artifact/static.ipynb`](https://github.com/neutrino-gpu/neutrino/tree//artifact/static.ipynb)
  2. **Dynamic** evaluation notebook: [`artifact/dynamic.ipynb`](https://github.com/neutrino-gpu/neutrino/tree//artifact/dynamic.ipynb)
* **Google Colab**:
  1. **Static** evaluation notebook (Use CPU as Runtime) is https://colab.research.google.com/drive/1w2vvjXlOIy00KNwStmSy-rVi2Y0CXfQx?usp=sharing 
  2. **Dynamic** evaluation notebook (Use GPU as Runtime) is https://colab.research.google.com/drive/1Ffg5zWZzvsXxb9vuquBvK0cwSf_SReVt?usp=sharing

### Hardware Requirement

For **static** evaluation, only a CPU machine with Python 3 runtime is needed. You _don't need to install Neutrino_ for static evaluation.

For **dynamic** evaluation, you will need a NVIDIA GPU with the CUDA driver installed. 
Please note:
1. The choice of hardware will significantly affect results. Please consider using A100, the same hardware we used in the paper. 
2. Please make sure no other workload is executing on the same GPU.
3. Please arrange enough disk space, at least 10GB, for dynamic traces collection. 

### Software Requirements

NEUTRINO system only depends on GNU toolchain (`gcc`, `file`, `git`, `nm`), CUDA toolchain (`cuobjdump`, `ptxas`) and Python 3.12 (`pip`, `toml`). 
But evaluation workload needs a PTX-included build of PyTorch and CUTLASS.
We package the dependency checking and installation in [`prepare_env.py`](https://github.com/neutrino-gpu/neutrino/tree/) for one-click installation.

### Installation

It's recommended to use virtual environments, e.g., `conda`, for installation when not using Colab.

**Automatic Installation**: We provide a helper script [`prepare_env.py`](https://github.com/neutrino-gpu/neutrino/tree/) that one can `python prepare_env.py` to install all dependencies. Jupyter Notebooks (also Google Colab) also use this way.

**Manual Installation**:

    1. Clone the repo: `git clone -b  https://github.com/neutrino-gpu/neutrino.git`
    2. Create a virtualenv: `conda create -y -n  python=3.12 && conda activate `
    3. Build and install neutrino: `cd neutrino && python setup.py install && cd ..`
    4. Test installation with `neutrino --help`
    Please refer to the README file on GitHub for more detailed descriptions in installing the PTX-included build of PyTorch and CUTLASS.

## Evaluation Workflow

### Getting Started Instructions

The Getting Started instructions, taking 30 minutes or less, consist of two parts:
1. All **static** evaluation that reproduces all figures and tables in the paper based on collected traces. 
2. The `block_sched` section (1st section) of the **dynamic** evaluation that collects and analyzes the block scheduling traces. This section takes <1 minute and helps justify the correct environment setup for detailed instructions.

You can use Colab to execute the evaluation scripts. 
To do this, first select the correct Runtime (CPU or GPU as stated above), 
then click the Runtime button at the top of the Colab web page, 
and click the Run All button in the dropdown menu to execute the scripts. 
Each section (of several blocks) can be executed independently.
Statistics or figures will be displayed below each cell when execution finishes.

If you choose to evaluate locally, please download the Jupyter Notebooks and follow the same steps as the Colab execution instructions above. 

### Detailed Instructions

The detailed instructions cover the rest five sections of the **dynamic** evaluation. 
They are also packaged in a Jupyter Notebook (also available on Colab), allowing one-click (`Runtime -> Run All`) execution and evaluation. 
Each section can also be executed independently. 
So you can clear up traces after each section to save disk space.

### Expected Results

_Static evaluation_ on collected traces are expected to closely fit the figures and tables presented in the paper, 
except for some statistics in the first five rows of Table. 2 and Sec. 4.5. 
To save disk space, we mistakenly deleted the original traces for these results. 
And because these results capture the finest runtime dynamics of the GPU, exact reproduction will be impossible. 
Our later experiments can only reproduce similar results. 
Please accept our apologies for the inconvenience, and we will update the revised paper to include the latest results. 

_Dynamic evaluation_ on customized traces is expected to produce similar results, i.e., similar numbers or figure shapes. 
And if you encountered any problems, please contact us through HotCRP.

### Further Evaluation

After completing the above evaluation and reading the documentation, we recommend several ways for further evaluating Neutrino: 
* **Test your workloads**: Neutrino supports most GPU workloads. You can import your GPU kernels (CUDA C++, Triton, etc) and test them via `neutrino <your workload>`. 
* **Customize probes**: First, read the Programmable Probe guide, write and save your probe in `.toml` locally, and apply it using `neutrino -p <path>`. 
* **Investigate Implementation**: Neutrino's implementation is small and well organized, and it's a good entry to understand how GPU code dispatches from OS. You can find the implementation of hook driver in [neutrino/src/](https://github.com/neutrino-gpu/neutrino/tree/artifact/neutrino/src) and the probe engine in [neutrino/probe](https://github.com/neutrino-gpu/neutrino/tree/artifact/neutrino/probe).

## Badges Checklists

* **Artifacts Available**: The source code of Neutrino is available at https://github.com/neutrino-gpu/neutrino/
* **Artifacts Functional**:
  * _Documentation_: We maintain an online documentation at https://neutrino-gpu.github.io/
  * _Completeness_: Our artifacts cover all system components described in the paper (a  complete list in our [GitHub README](https://github.com/neutrino-gpu/neutrino/blob/artifact/README.md)):
    * Hook Driver: https://github.com/neutrino-gpu/neutrino/tree/artifact/neutrino/src
    * Probe Engine: https://github.com/neutrino-gpu/neutrino/tree/artifact/neutrino/probe
  * _Exercisability_: We package Neutrino as a Python package for easy execution and include all scripts for conducting experiments in the paper. 
* **Results Reproduced**: To reproduce the main results presented in the paper, we provide Jupyter Notebooks (also on Colab) containing all environment setup, trace collection, and results analysis. We also provide detailed guidelines to help understand results reproduction. 