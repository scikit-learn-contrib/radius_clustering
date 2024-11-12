#!/bin/bash
set -e -u -x

PLAT=manylinux_2_24_x86_64

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Compile wheels
for PYBIN in /opt/python/*/bin; do
  if [[ "$PYBIN" == *cp39* || "$PYBIN" == *cp310* || $PYBIN == *cp311* ]] ; then
    "${PYBIN}/pip" wheel --no-deps /io/ -w wheelhouse/
  fi
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done