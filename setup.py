import platform
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

SYSTEM = platform.system()
CPU = platform.processor()

C_COMPILE_ARGS = ["-std=c99", "-O3", "-ffast-math", "-DREP"]
CXX_COMPILE_ARGS = ["-std=c++11", "-O3", "-ffast-math"]
CXX_LINK_ARGS = ["-std=c++11"]

if not CPU:
    CPU = platform.machine()

if (SYSTEM != "Darwin") and (CPU not in "arm64"):
    C_COMPILE_ARGS.append("-march=native")
    CXX_COMPILE_ARGS.append("-march=native")
    CXX_LINK_ARGS.append("-fopenmp")

extensions = [
    Extension(
        "radius_clustering.utils._emos",
        ["radius_clustering/utils/emos.pyx", "radius_clustering/utils/main-emos.c"],
        include_dirs=[np.get_include(), "radius_clustering/utils"],
        extra_compile_args=C_COMPILE_ARGS,
    ),
    Extension(
        "radius_clustering.utils._mds_approx",
        [
            "radius_clustering/utils/mds.pyx",
            "radius_clustering/utils/mds_core.cpp",
            "radius_clustering/utils/random_manager.cpp",
        ],
        include_dirs=[np.get_include(), "radius_clustering/utils"],
        language="c++",
        extra_compile_args=CXX_COMPILE_ARGS,
        extra_link_args=CXX_LINK_ARGS,
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    package_data={"radius_clustering": ["utils/*.pyx", "utils/*.h"]},
)
