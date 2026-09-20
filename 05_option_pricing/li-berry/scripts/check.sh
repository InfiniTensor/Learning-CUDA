#!/usr/bin/env bash
# check.sh - e2e 测试辅助：结果文件数值提取与容差比较（source 使用）
set -u

# 提取结果文件中的数值字段
extract() { # $1=文件 $2=键
    awk -v k="$2" '$1 == k { print $NF }' "$1"
}

# 数值比较：|a - b| <= tol（awk 浮点运算）
num_ok() { # $1=a $2=b $3=tol
    awk -v a="$1" -v b="$2" -v tol="$3" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= tol) }'
}

# 数值范围检查：lo <= x <= hi
range_ok() { # $1=x $2=lo $3=hi
    awk -v x="$1" -v lo="$2" -v hi="$3" 'BEGIN { exit !(x >= lo && x <= hi) }'
}

# 断言（打印 PASS/FAIL 并累计计数）
CHECK_PASS=0
CHECK_FAIL=0
check() { # $1=描述 $2=条件命令（退出码 0 为通过）
    if eval "$2"; then
        CHECK_PASS=$((CHECK_PASS + 1))
        echo "PASS: $1"
    else
        CHECK_FAIL=$((CHECK_FAIL + 1))
        echo "FAIL: $1"
    fi
}

summary() {
    echo "=============================="
    echo "e2e: $CHECK_PASS passed, $CHECK_FAIL failed"
    [ "$CHECK_FAIL" -eq 0 ]
}
