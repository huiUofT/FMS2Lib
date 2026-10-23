# Optional MassFormer extension

The inherited MassFormer support includes these files under `src/foriceberg/ms_pred/massformer_pred/massformer_code/`:

- `algos2.pyx`: Cython source, retaining its Microsoft MIT header.
- `algos2.c`: generated C source.
- `algos2.cpython-311-x86_64-linux-gnu.so`: an existing binary whose filename identifies CPython 3.11 on x86-64 GNU/Linux.

The filename does not guarantee compatibility with a particular NumPy ABI, Linux runtime, or compiler environment. The original binary build environment was not recorded.

## Build from source for another environment

This extension is imported by the optional MassFormer modules. It is separate from the documented ICEBERG DAG and MSBuddy workflows. To build it, use the intended Python environment, a C compiler, Python development headers, NumPy headers, setuptools, and Cython.

The following source-build recipe has not been verified. It compiles the existing `.pyx` source and writes a platform-specific extension beside it. Run it only when you intend to rebuild the binary; it can replace an existing binary for the same Python/platform.

```bash
# From the repository root, in your selected Python environment:
python -m pip install setuptools Cython 'numpy>=1.24,<2'
cd src/foriceberg
python - <<'PY'
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extension = Extension(
    'ms_pred.massformer_pred.massformer_code.algos2',
    ['ms_pred/massformer_pred/massformer_code/algos2.pyx'],
    include_dirs=[numpy.get_include()],
)
setup(
    name='fms2lib-massformer-extension',
    ext_modules=cythonize(
        [extension],
        build_dir='build/cython',
        compiler_directives={'language_level': 3},
    ),
    script_args=['build_ext', '--inplace'],
)
PY
```

From the same `src/foriceberg` directory, check loading in that environment:

```bash
python -c 'from ms_pred.massformer_pred.massformer_code import algos2; print(algos2.__file__)'
```

The import also executes package initialization, so the dependencies of those packages must be installed. Successful loading verifies the extension can be imported; it does not validate MassFormer predictions or model compatibility. See [third-party code and licenses](../README.md#third-party-code-and-licenses) for the retained license material.
