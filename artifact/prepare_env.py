import os
import sys
import subprocess

# Prepare Neutrino
try:
    print(subprocess.check_output(["neutrino", "--help"]).decode())
except:
    # TODO verify if this is included in the clone
    # clone the code from Github
    branch = os.getenv("NEUTRINO_BRANCH", "artifact")
    subprocess.check_output(["git", "clone", "--branch", branch, "https://github.com/neutrino-gpu/neutrino.git"])
    subprocess.check_output([sys.executable, "setup.py", "install"], cwd="neutrino")
    print(subprocess.check_output(["neutrino", "--help"]).decode())

# Prepare PyTorch
# first check the torch installation, noted that we shall not import torch in Jupyter Notebook
# or the import will be locked by Jupyter Notebook
torch_version = subprocess.check_output([sys.executable, "-c", "import torch; print(torch.__version__)"]).decode()
# our building is from v2.5.0 branch, and unavoidably has git in version tag
need_reinstall = len(torch_version) == 0 or not "git" in torch_version
if need_reinstall:
    subprocess.check_output([sys.executable, "-c", "'import torch; print(torch.__version__)'"])
    # now check the installation taget
    sm_version = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
    # sm_version like `8.9`
    sm_version = sm_version.decode().split("\n")[0].strip()
    major, minor = sm_version.split(".")
    # now download the pre-built wheel from anonymous link of object storage
    subprocess.check_output(["wget", "--quiet", f"https://pub-eef24bf0aa5b4950860ea28dfbe39d8c.r2.dev/sm_{major}{minor}/torch/torch-2.5.0-cp311-cp311-linux_x86_64.whl"])
    # now install the wheel
    subprocess.check_output([sys.executable, "-m", "pip", "install", "torch-2.5.0-cp311-cp311-linux_x86_64.whl"])
    # validate the installation
    torch_version = subprocess.check_output([sys.executable, "-c", "'import torch; print(torch.__version__)'"])
    # fix a bug in mkl version
    subprocess.check_output([sys.executable, "-m", "pip", "install", "mkl==2024.0"])
    # assert len(torch_version) > 0 and "git" in torch_version.decode()

# Prepare Triton
triton_version = subprocess.check_output([sys.executable, "-c", "'import triton; print(triton.__version__)'"])
if len(triton_version) == 0: # No installation so let's install 3.0.0, but 3.+ is fine
    subprocess.check_output([sys.executable, "-m", "pip", "install", "triton==3.1.0"])

# Prepare CUTLASS
cutlass_exists = "cutlass" in os.listdir(".")
if not cutlass_exists:
    subprocess.check_output(["git", "clone", "https://github.com/neutrino-gpu/cutlass/"])
    os.makedirs("cutlass/build", exist_ok=True)
    sm_version = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
    # sm_version like `8.9`
    sm_version = sm_version.decode().split("\n")[0].strip()
    major, minor = sm_version.split(".")
    subprocess.check_output(["cmake", "..", f'-DCUTLASS_NVCC_ARCHS={major}{minor}', '-DCUTLASS_NVCC_EMBED_PTX=1'], cwd="cutlass/build")
    subprocess.check_output(["make"], cwd="cutlass/build/examples/benchmark/")
    subprocess.check_output(["ls"], cwd="cutlass/build/examples/benchmark/")
