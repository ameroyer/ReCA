#!/bin/bash

AIBUILD="/home/amelie/Libs/AI-Toolbox/build"
AIINCLUDE="/home/amelie/Libs/AI-Toolbox/include"
EIGEN="/usr/local/include/eigen3/"
GCC="/usr/local/bin/gcc-4.9.0/bin/g++"
STDLIB="/usr/local/bin/gcc-4.9.0/lib64"


# PARAMETERS
[[ "$#" > 0 ]] && DATA="$1" || DATA="fm"
[[ "$#" > 1 ]] && PLEVEL="$2" || PLEVEL="4"
[[ "$#" > 2 ]] && HIST="$3" || HIST="2"
[[ "$#" > 3 ]] && DISCOUNT="$4" || DISCOUNT="0.95"
[[ "$#" > 4 ]] && PRECISION="$5" || PRECISION="0"

# GLOBAL VARIABLES
PROFILES=6
ALPHA=1.10
STEPS=1000000
#PRECISION=5 for 3 actions

# DATA PATH
if [ $DATA = "fm" ]; then 
    printf -v BASE '/home/amelie/Rotations/ChatterjeeRotation/Code/Models/Foodmart%d%d%d/foodmart_u%d_k%d_pl%d_a%.2f' "$PROFILES" "$HIST" "$PLEVEL" "$PROFILES" "$HIST" "$PLEVEL" "$ALPHA"
elif [ $DATA = "rd" ]; then
    if [ $PLEVEL == "4" ]; then
	PROFILES=3
    elif [ $PLEVEL == "3" ]; then
	PROFILES=22
    elif [ $PLEVEL == "2" ]; then
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
echo "Compiling MDP model in mainMDP"
$GCC -O3 -Wl,-rpath,$STDLIB -DNITEMSPRM=$ITEMS -DHISTPRM=$HIST -DNPROFILESPRM=$PROFILES -std=c++11 utils.cpp main_MDP.cpp -o mainMDP -I $AIINCLUDE -I $EIGEN -L $AIBUILD -l AIToolboxMDP -l AIToolboxPOMDP -l lpsolve55
if [ $? -ne 0 ]
then
    echo "Compilation failed!"
    echo "exit"
    exit 1
fi

# RUN
echo
echo "Running mainMDP on $BASE"
./mainMDP $BASE $DISCOUNT $STEPS $PRECISION
echo
