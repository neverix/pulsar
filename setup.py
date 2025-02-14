from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from glob import glob

setup(
    name='pulsar',
    ext_modules=[
        CUDAExtension(
            'pulsar_cuda',
            list(f for ext in ("cu", "cpp") for f in glob(f"csrc/**/*.{ext}", recursive=True)))
    ],
    packages=["pulsar"],
    cmdclass={
        'build_ext': BuildExtension
    })
