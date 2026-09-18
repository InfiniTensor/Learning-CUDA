#!/usr/bin/env bash

COREX_HOME=${COREX_HOME:-/usr/local/corex-4.4.0}
export PATH="$COREX_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="${COREX_LD_LIBRARY_PATH:-$COREX_HOME/lib64}:${LD_LIBRARY_PATH:-}"
