from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension("cuda.nvfp4_cuda", ["cuda/nvfp4.cu"]),
        CUDAExtension("cuda.mxfp8_cuda", ["cuda/mxfp8.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension}
)