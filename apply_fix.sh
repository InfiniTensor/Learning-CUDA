#!/bin/bash
# Script to apply double precision fix to flashAttentionFallback kernel

cd /data1/kppppp/Learning-CUDA

# Backup original file
cp src/kernels.cu src/kernels.cu.backup

# Apply the fix using sed
sed -i '
/flashAttentionFallback/,/^}/ {
    s/float maxVal = -INFINITY;/double maxVal = -INFINITY;/
    s/float sumExp = 0\.0f;/double sumExp = 0.0;/
    s/float result = 0\.0f;/double result = 0.0;/
    s/float dot = 0\.0f;/double dot = 0.0;/
    s/float prevMax = maxVal;/double prevMax = maxVal;/
    s/float correction =/double correction =/
    s/float weight =/double weight =/
    s/fmaxf(maxVal, dot)/fmax(maxVal, dot)/
    s/expf(/exp(/g
    s/0\.0f/0.0/g
}
' src/kernels.cu

# Also update the comment
sed -i 's/\/\/ Online softmax approach$/\/\/ Online softmax approach - use double precision for accumulation/' src/kernels.cu

echo "Fix applied! Verifying changes..."
grep -A 30 "// Online softmax approach" src/kernels.cu | head -35

echo ""
echo "Now compile and test:"
echo "  make PLATFORM=iluvatar build"
echo "  ./test_kernels"
