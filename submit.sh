#!/bin/bash

mkdir -p /proj/checkpoints/zwang619/results/$2/$3

bsub -q normal -G grp_ai_compiler_design -M 2000G -hl -n $1 -J $2/$3 \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo /proj/checkpoints/zwang619/results/$2/$3/output.log \
  -eo /proj/checkpoints/zwang619/results/$2/$3/err.log \
  blaunch bash $3.sh \
    
watch -n 1 bjobs -prs

# -R "select[hname != 'p2-r05-n1' && hname != 'p5-r02-n4' && hname != 'p3-r20-n2' && hname != 'p6-r01-n2' && hname != 'p3-r18-n2' && hname != 'p1-r31-n3' && hname != 'p5-r24-n4']" \