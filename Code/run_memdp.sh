#!/bin/bash

AIBUILD="/home/amelie/Libs/AI-Toolbox/build"
AIINCLUDE="/home/amelie/Libs/AI-Toolbox/include"
EIGEN="/usr/local/include/eigen3/"
LPSOLVE="/usr/local/lib/"
GCC="/usr/local/bin/gcc-4.9.0/bin/g++"
STDLIB="/usr/local/bin/gcc-4.9.0/lib64"

# PARAMETERS
[[ "$#" > 0 ]] && DATA="$1" || DATA="fm"
[[ "$#" > 1 ]] && ALGO="$2" || ALGO="IP"
[[ "$#" > 2 ]] && PLEVEL="$3" || PLEVEL="4"
[[ "$#" > 3 ]] && HIST="$4" || HIST="2"
[[ "$#" > 4 ]] && HORIZON="$5" || HORIZON="90"
[[ "$#" > 5 ]] && DISCOUNT="$6" || DISCOUNT="0.95"
[[ "$#" > 6 ]] && PRECISION="$7" || PRECISION="0"

# GLOBAL VARIABLES
PROFILES=6
ALPHA=1.10
STEPS=1200
PRECISION=0

# DATA PATH
if [ $DATA = "fm" ]; then 
    printf -v BASE '/home/amelie/Rotations/ChatterjeeRotation/Code/Models/Foodmart%d%d%d/foodmart_u%d_k%d_pl%d_a%.2f' "$PROFILES" "$HIST" "$PLEVEL" "$PROFILES" "$HIST" "$PLEVEL" "$ALPHA"
elif [ $DATA = "rd" ]; then
    if [ $PLEVEL = "4" ]; then
	PROFILES=3
    elif [ $PLEVEL = "3" ]; then
	PROFILES=22
    elif [ $PLEVEL = "2" ]; then
	PROFILES=45
    fi
    printf -v BASE '/home/amelie/Rotations/ChatterjeeRotation/Code/Models/Random%d%d%d/random_u%d_k%d_pl%d' "$PROFILES" "$HIST" "$PLEVEL" "$PROFILES" "$HIST" "$PLEVEL"
else
    echo "Unkown data mode $DATA"
    echo "exit"
    exit 1
fi
if [ ! -f "$BASE.items" ]; then
    echo "File $BASE.items not found"
    echo "exit"
    exit 1
fi
ITEMS=$(($(wc -l < "$BASE.items") + 1))

# COMPILE
echo
echo "Compiling MEMDP model in mainMEMDP"
$GCC -O3 -Wl,-rpath,$STDLIB -DNITEMSPRM=$ITEMS -DHISTPRM=$HIST -DNPROFILESPRM=$PROFILES -std=c++11 utils.cpp main_MEMDP.cpp -o mainMEMDP -I $AIINCLUDE -I $EIGEN -L $LPSOLVE -L $AIBUILD -l AIToolboxMDP -l AIToolboxPOMDP -l lpsolve55
if [ $? -ne 0 ]
then
    echo "Compilation failed!"
    echo "exit"
    exit 1
fi

# RUN
echo
echo "Running mainMEMDP on $BASE with $ALGO solver"
./mainMEMDP $BASE $ALGO $DISCOUNT $STEPS $PRECISION $HORIZON
echo
