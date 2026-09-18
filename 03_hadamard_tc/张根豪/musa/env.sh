#!/usr/bin/env bash
MUSA_HOME=${MUSA_HOME:-/usr/local/musa-5.1.0}
export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:${LD_LIBRARY_PATH:-}"