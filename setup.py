from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "radius_clustering.utils._emos",
        ["radius_clustering/utils/emos.pyx", "radius_clustering/utils/main-emos.c"],
        include_dirs=[np.get_include(), "radius_clustering/utils"],
        extra_compile_args=["-std=c99", "-O3", "-march=native", "-ffast-math", "-DREP"],
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
        extra_compile_args=["-std=c++11", "-O3", "-march=native", "-ffast-math"],
        extra_link_args=["-std=c++11", "-fopenmp"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()],
    package_data={"radius_clustering": ["utils/*.pyx", "utils/*.h"]},
)
