from pathlib import Path

from setuptools import Extension, setup


def get_numpy_include() -> str:
    """Helper function to determine the numpy include path.

    The purpose of this function is to postpone importing includeigen
    until it is actually installed.
    """
    import numpy as np

    return np.get_include()


ext_modules = [
    Extension(
        "compositionaldust.core.dust",
        sources=[
            "src/compositionaldust/core/dust.pyx",
        ],
    ),
    Extension(
        "compositionaldust.core.cost_symbolic",
        sources=[
            "src/compositionaldust/core/cost_symbolic.pyx",
        ],
    ),
    Extension(
        "compositionaldust.core.random",
        sources=[
            "src/compositionaldust/core/random.pyx",
        ],
    ),
    Extension(
        "compositionaldust.core.utils",
        sources=[
            "src/compositionaldust/core/utils.pyx",
        ],
    ),
]

if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(
        ext_modules=cythonize(ext_modules, language_level="3"),
        include_dirs=[get_numpy_include()],
    )
