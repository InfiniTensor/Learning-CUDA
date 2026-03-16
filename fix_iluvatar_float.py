#!/usr/bin/env python3
"""
Fix script for Iluvatar BI-V100 Flash Attention float precision issue
Changes float accumulation to double precision in flashAttentionFallback kernel
"""

import os
import sys

def apply_fix():
    filepath = 'src/kernels.cu'
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found!")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Read the file
    print(f"Reading {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Backup
    backup_path = filepath + '.before_double_fix'
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Backup created: {backup_path}")
    
    # Apply fixes - be very specific to avoid changing other parts
    replacements = [
        # In flashAttentionFallback only
        ('    // Online softmax approach\n    float maxVal = -INFINITY;',
         '    // Online softmax approach - use double precision for accumulation\n    double maxVal = -INFINITY;'),
        ('    float sumExp = 0.0f;', '    double sumExp = 0.0;'),
        ('    float result = 0.0f;', '    double result = 0.0;'),
        ('        float dot = 0.0f;', '        double dot = 0.0;'),
        ('        float prevMax = maxVal;', '        double prevMax = maxVal;'),
        ('        maxVal = fmaxf(maxVal, dot);', '        maxVal = fmax(maxVal, dot);'),
        ('        float correction = (prevMax == -INFINITY) ? 0.0f : expf(prevMax - maxVal);',
         '        double correction = (prevMax == -INFINITY) ? 0.0 : exp(prevMax - maxVal);'),
        ('        float weight = expf(dot - maxVal);',
         '        double weight = exp(dot - maxVal);'),
        ('    O[oIdx] = TypeConverter<T>::fromFloat((sumExp > 0.0f) ? (result / sumExp) : 0.0f);',
         '    O[oIdx] = TypeConverter<T>::fromFloat((sumExp > 0.0) ? (result / sumExp) : 0.0);'),
    ]
    
    print("\nApplying fixes...")
    for i, (old, new) in enumerate(replacements, 1):
        if old in content:
            content = content.replace(old, new, 1)  # Replace only first occurrence
            print(f"  ✓ Fix {i}/9 applied")
        else:
            print(f"  ✗ Fix {i}/9 FAILED - pattern not found:")
            print(f"    Looking for: {old[:60]}...")
            # Don't exit, continue to see all failures
    
    # Write the modified content
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"\n✓ Changes written to {filepath}")
    
    # Show the modified section
    print("\n" + "="*70)
    print("Modified flashAttentionFallback kernel (lines with double):")
    print("="*70)
    
    lines = content.split('\n')
    in_section = False
    line_count = 0
    for i, line in enumerate(lines, 1):
        if '// Online softmax approach' in line:
            in_section = True
        if in_section:
            print(f"{i:4d}: {line}")
            line_count += 1
            if 'O[oIdx] = TypeConverter' in line:
                break
    
    print("\n" + "="*70)
    print("Verification:")
    print("="*70)
    
    # Count occurrences to verify
    double_count = content.count('double maxVal')
    double_sumexp = content.count('double sumExp')
    double_result = content.count('double result')
    
    print(f"  double maxVal occurrences: {double_count} (expected: 1)")
    print(f"  double sumExp occurrences: {double_sumexp} (expected: 1)")
    print(f"  double result occurrences: {double_result} (expected: 1)")
    
    if double_count >= 1 and double_sumexp >= 1 and double_result >= 1:
        print("\n✓ Fix appears successful!")
        print("\nNext steps:")
        print("  1. Compile: make PLATFORM=iluvatar build")
        print("  2. Test: ./test_kernels")
        print("  3. Check if all 90 tests pass")
    else:
        print("\n✗ Fix may not have been fully applied. Check the output above.")
        return False
    
    return True

if __name__ == '__main__':
    print("="*70)
    print("Iluvatar Flash Attention Float Fix - Double Precision Patch")
    print("="*70)
    
    # Change to the right directory if needed
    if not os.path.exists('src/kernels.cu'):
        expected_dir = '/data1/kppppp/Learning-CUDA'
        if os.path.exists(expected_dir):
            os.chdir(expected_dir)
            print(f"Changed directory to: {expected_dir}")
        else:
            print(f"ERROR: Cannot find kernels.cu")
            sys.exit(1)
    
    success = apply_fix()
    sys.exit(0 if success else 1)
