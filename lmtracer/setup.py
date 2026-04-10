from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
import site
import os

parallel_jobs = os.cpu_count()

class build_py_with_pth_file(build_py):
     """Include the .pth file for this project, in the generated wheel."""

     def run(self):
         super().run()

         destination_in_wheel = "lmtracer_bootstrap.pth"
         location_in_source_tree = "lmtracer_bootstrap.pth"
 
         outfile = os.path.join(self.build_lib, destination_in_wheel)
         self.copy_file(location_in_source_tree, outfile, preserve_mode=0)

setup(
    name="lmtracer",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="lmtracer.cuda.timer_kernel",
            sources=[str(Path("csrc/timer_kernel.cu"))],
            include_dirs=["csrc"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            name="lmtracer.cuda.timer_mem",
            sources=[str(Path("csrc/timer_mem.cu"))],
            include_dirs=["csrc"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            name="lmtracer.cuda.globaltimer",
            sources=[str(Path("csrc/globaltimer.cu"))],
            include_dirs=["csrc"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            name="lmtracer.cuda.timer_op",
            sources=[str(Path("csrc/timer_op.cu"))],
            include_dirs=["csrc"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(parallel=parallel_jobs, use_ninja=False), "build_py": build_py_with_pth_file},
    
    # cmdclass={"build_ext": BuildExtension},
)