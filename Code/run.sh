#!/bin/bash

# PATHES (if local installation)
AIBUILD="/home/amelie/Libs/AI-Toolbox/build"
AIINCLUDE="/home/amelie/Libs/AI-Toolbox/include"
EIGEN="/usr/local/include/eigen3/"
LPSOLVE="/usr/local/lib/"
GCC="/usr/local/bin/gcc-4.9.0/bin/g++"
STDLIB="/usr/local/bin/gcc-4.9.0/lib64"

# DEFAULT ARGUMENTS
MODE="mdp"
DATA="fm"
PLEVEL="4"
HIST="2"
DISCOUNT="0.95"
STEPS="1500"
EPSILON="0.01"
PRECISION="0"
VERBOSE="0"
BELIEFSIZE="100"
EXPLORATION="10000"
HORIZON="2"
COMPILE=false

# SET  ARGUMENTS FROM CMD LINE
while getopts "m:d:n:k:g:s:h:e:x:b:cpv" opt; do
  case $opt in
    m)
      MODE=$OPTARG
      ;;
    d)
      DATA=$OPTARG
      ;;
    n)
      PLEVEL=$OPTARG
      ;;
    k)
      HIST=$OPTARG
      ;;
    g)
      DISCOUNT=$OPTARG
      ;;
    s)
      STEPS=$OPTARG
      ;;
    p)
      PRECISION=1
      ;;
    v)
      VERBOSE=1
      ;;
    h)
      HORIZON=$OPTARG
      ;;
    e)
      EPSILON=$OPTARG
      ;;
    b)
      BELIEFSIZE=$OPTARG
      ;;
    x)
      EXPLORATION=$OPTARG
      ;;
    c)
      COMPILE=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# SET CORRECT DATA PATHS
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ $DATA = "fm" ]; then 
    PROFILES=6
    ALPHA=1.10
    printf -v BASE "$DIR/Models/Foodmart%d%d%d/foodmart_u%d_k%d_pl%d_a%.2f" "$PROFILES" "$HIST" "$PLEVEL" "$PROFILES" "$HIST" "$PLEVEL" "$ALPHA"
    if [ ! -f "$BASE.items" ]; then
	echo "File $BASE.items not found"
	echo "exit"
	exit 1
    fi
    DATA="reco"
    NITEMS=$(($(wc -l < "$BASE.items") + 1))
elif [ $DATA = "rd" ]; then
    PROFILES=$PLEVEL
    NITEMS=$PLEVEL
    printf -v BASE "$DIR/Models/Synth%d%d%d/synth_u%d_k%d_pl%d" "$PLEVEL" "$HIST" "$PLEVEL" "$PLEVEL" "$HIST" "$PLEVEL"
    DATA="reco"
elif [ $DATA = "mz" ]; then
    NAME=$PLEVEL
    printf -v BASE "$DIR/Models/%s/%s" "$PLEVEL" "$PLEVEL"
    DATA="maze"
else
    echo "Unkown data mode $DATA"
    echo "exit"
    exit 1
fi
# MDP
if [ $MODE = "mdp" ]; then
# COMPILE
    if [ "$COMPILE" = true ]; then
	echo
	echo "Compiling MDP model in mainMDP"
	$GCC -O3 -Wl,-rpath,$STDLIB -DNITEMSPRM=$NITEMS -DHISTPRM=$HIST -DNPROFILESPRM=$PROFILES -std=c++11 mazemodel.cpp recomodel.cpp utils.cpp main_MDP.cpp -o mainMDP -I $AIINCLUDE -I $EIGEN -L $AIBUILD -l AIToolboxMDP -l AIToolboxPOMDP -l lpsolve55
	if [ $? -ne 0 ]; then
	    echo "Compilation failed!"
	    echo "exit"
	    exit 1
	fi
    fi

# RUN
    echo
    echo "Running mainMDP on $BASE"
    ./mainMDP $BASE $DATA $DISCOUNT $STEPS $EPSILON $PRECISION $VERBOSE
    echo
# POMDPs
else
# COMPILE
    if [ "$COMPILE" = true ]; then
	echo
	echo "Compiling MEMDP model in mainMEMDP"
	$GCC -O3 -Wl,-rpath,$STDLIB -DNITEMSPRM=$NITEMS -DHISTPRM=$HIST -DNPROFILESPRM=$PROFILES -std=c++11 mazemodel.cpp recomodel.cpp utils.cpp main_MEMDP.cpp -o mainMEMDP -I $AIINCLUDE -I $EIGEN -L $LPSOLVE -L $AIBUILD -l AIToolboxMDP -l AIToolboxPOMDP -l lpsolve55
	if [ $? -ne 0 ]
	then
	    echo "Compilation failed!"
	    echo "exit"
	    exit 1
	fi
    fi   

# RUN
    echo
    echo "Running mainMEMDP on $BASE with $MODE solver"
    ./mainMEMDP $BASE $DATA $MODE $DISCOUNT $STEPS $HORIZON $EPSILON $EXPLORATION $BELIEFSIZE $PRECISION $VERBOSE
    echo
fi