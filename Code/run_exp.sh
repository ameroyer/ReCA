#!/bin/bash

[[ "$#" > 0 ]] && PLEVEL="$1" || PLEVEL="4"
[[ "$#" > 1 ]] && HIST="$2" || HIST="2"
[[ "$#" > 2 ]] && PRECISION="$3" || PRECISION="0"


# GLOBAL VARIABLES
PROFILES=6
ALPHA=1.10
STEPS=1000000
PRECISION=0 #PRECISION=5 for 3 actions
DISCOUNT=0.95

echo "Foodmart"
echo
echo "MDP"
echo
./run_mdp.sh "fm" $PLEVEL $HIST $DISCOUNT $PRECISION
echo
echo "IP"
echo "too slow"
#./run_memdp.sh "fm" "IP"  $PLEVEL $HIST 1 0.95 $PRECISION
echo
echo "POMCP"
echo
./run_memdp.sh "fm" "POMCP"  $PLEVEL $HIST 1 0.95 $PRECISION
echo
echo "MEMCP"
echo
./run_memdp.sh "fm" "MEMCP"  $PLEVEL $HIST 1 0.95 $PRECISION

echo "Random"
echo
echo "MDP"
echo
./run_mdp.sh "rd" $PLEVEL $HIST $DISCOUNT $PRECISION
echo
echo "IP"
echo "too slow"
#./run_memdp.sh "rd" "IP"  $PLEVEL $HIST 1 0.95 $PRECISION
echo
echo "POMCP"
echo
./run_memdp.sh "rd" "POMCP"  $PLEVEL $HIST 1 0.95 $PRECISION
echo
echo "MEMCP"
echo
./run_memdp.sh "rd" "MEMCP"  $PLEVEL $HIST 1 0.95 $PRECISION