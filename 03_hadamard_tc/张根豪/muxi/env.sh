#!/usr/bin/env bash
MUXI_HOME=${MUXI_HOME:-/opt/maca-3.5.3}
export PATH="$MUXI_HOME/mxgpu_llvm/bin:$MUXI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUXI_HOME/lib:$MUXI_HOME/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"
