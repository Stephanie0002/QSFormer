#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension

module = CppExtension(
    name='cpp_cores',
    sources=[
        'src/export.cpp',
        'src/temporal_neighbor_sampler.cpp'
    ],
    include_dirs=['src'],
    extra_compile_args={'cxx': ['-g', '-O3', '-fopenmp', '-std=c++17']}
)

# Now proceed to setup
setup(
    name='cpp_cores',
    version='1.0',
    description='some cpp implementations for Dygformer',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    ext_modules=[module],
    cmdclass={
        'build_ext': BuildExtension
    }
)
