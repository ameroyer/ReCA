# [ R e C A ]  r e a d   m e 

# installation

#### requirements
   * python (the scripts were tested with versions 2.7 and 3.2)
   * GCC 4.9 +
   * cmake [``cmake`` package]
   * Boost version 1.53+ [``libboost-dev`` package]
   * [Eigen 3.2+](http://eigen.tuxfamily.org/index.php?title=Main_Page) library
   * [lp_solve](http://lpsolve.sourceforge.net/5.5/) library [``lp-solve`` package]

#### installing AIToolbox
Clone the [AITB repository](https://github.com/Svalorzen/AI-Toolbox), then build and test the installation with the following commands

```bash
cd AIToolbox_root
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
ctest -V
```

# dataset generation

#### synthetic recommandation dataset
Generate synthetic POMDP parameters to highlight the impact of using multiple environments. The model comprises as many environments as possible recommandations. The *i*-th environment corresponds to users choosing item *i* with a high probability (``p=0.8``) and uniform preference towards other recommandations.  The reward is 0 if the recommendation does not match the user's choice, and 1 otherwise.

  ```bash
  cd Data/
  ./prepare_synth.py -n [1] -k [2] -a [3] -t [4] -o [5] --norm --help
  ```

  * ``[1]`` Number of items (Defaults to 3).
  * ``[2]`` History length (Defaults to 2). Must be strictly greater than 1.
  * ``[3]`` Positive scaling parameter for correct recommandation. Must be greater than 1. Defaults to 1.1.
  * ``[4]`` Number of test sessions to generate following the generated distribution. Defaults to 2000.
  * ``[5]`` Path to the output directory (Defaults to ``../Code/Models``).
  * ``[--norm]`` If present, normalize the output transition probabilities.
  * ``[--zip]`` If present, transitions are stored in an archive. Recommended for large state spaces.
  * ``[--help]`` displays help about the script.

#### Foodmart dataset
 Estimate a POMDP model parameters and test sequences from the [Foodmart](https://github.com/neo4j-examples/neo4j-foodmart-dataset) dataset (.csv dataset files are included in the ``Data/`` directory).

 ```bash
  cd Data/
  ./prepare_foodmart.py -pl [1] -k [2] -ul [3] -a [4] -t [5] -d [6] -o [7] --norm --help
```

  * ``[1]`` Items discretization level. Must be between 0 (*1561 fine-grained products*)  and 4 (*3 high-level categories*). Defaults to 4.
  * ``[2]`` History length, > 1. Defaults to 2.
  * ``[3]`` User profiles discretization level (*0*: 10 profiles, *1*: 24 profiles, *2*: 48 profiles). Defaults to 0.
  * ``[4]`` Positive scaling parameter for correct recommandation. Must be greater than 1. Defaults to 1.4.
  * ``[5]`` Proportion of the dataset to keep for parameter inference. Defaults to 0.8.
  * ``[6]`` Path to the Foodmart dataset. Defaults to ``Data/Foodmart.gz``.
  * ``[7]`` Path to the output directory. Defaults to ``../Code/Models``.
  * ``[--norm]`` If present, output transition probabilities are normalized.
  * ``[--zip]`` If present, transitions are stored in an archive. Recommended for large state spaces.
  * ``[--help]`` displays help about the script.
  
#### maze dataset
Generating POMDP parameters for a typical maze/path finding problem with multiple environments.

  ```bash
  cd Data/
  ./prepare_maze.py -i [1] -n [2] -s [3] -t [4] -w [5] -g [6] -e [7] -wf [8] -o [9] --rdf --help
  ```

  * ``[1]`` If given, load the maze structure from a file (see toy examples in the ``Mazes`` subdirectory). if not, the mazes are generated randomly with the following parameters.
  * ``[2]`` Maze width and height.
  * ``[3]`` Number of initial states in each maze. Defaults to 1.
  * ``[4]`` Number of trap states in each maze (non-rewarding absorbing states). Defaults to 0.
  * ``[5]`` Number of obstacles in each maze. Defaults to 0.
  * ``[6]`` Number of goal states in each maze. Defaults to 1.
  * ``[7]`` Number of mazes (environments) to generate. Defaults to 1.
  * ``[8]`` Failure rate (equivalent to falling in a trap state) when going forward in the direction of an obstacle. Defaults to 0.05.
  * ``[9]`` Path to the output directory (Defaults to ``../Code/Models``).
  * ``[--norm]`` If present, normalize the output transition probabilities.
  * ``[--rdf]`` If present, the failure rates (probability of staying put instead of realizing the intended action) for each environment are sampled uniformly over [0; 0.5[
  * ``[--help]`` displays help about the script.

# building and evaluating the MEMDP-based models

#### set-up
The following variables can be configured at the beginning of the ``run.sh`` script (e.g. if some libaries are installed locally and not globally)
  * ``AIROOT``: path to the AIToolbox installation directory.
  * ``EIGEN``: path to the Eigen library installation directory.
  * ``LPSOLVE``: path to the lpsolve library installation directory.
  * ``GCC``: path to the g++ binary.
  * ``STDLIB``: path to the stdlib matching the given gcc compiler.

#### run
```bash
  cd Code/
./run.sh -m [1] -d [2] -n [3] -k [4] -u [5] -g [6] -s [7] -h [8] -e [9] -x [10] -b [11] -c -p -v
```

   * ``[1]`` Model to use. Defaults to mdp. Available options are 
      * *mdp*. MDP model obtained by a weighted average of all the environments' transition probabilities and solved by Value iteration. The solver can be configured with 
        * ``[7]`` Number of iterations. Defaults to 1500.
      * *pbvi*. point-based value iteration optimized for the MEMDP structure with options
        * ``[8]`` Horizon parameter. Must be greater than 1. Defaults to 2.
        * ``[11]`` Belief size. Defaults to  100.
      * *pomcp* and *pamcp*. Monte-carlo solver, respectively without and with optimization for the MEMDP structure with options
        * ``[7]`` Number of simulation steps. Defaults to 1500.
        * ``[8]`` Horizon parameter. Must be greater than 1. Defaults to 2.
        * ``[10]`` Exploration parameter. Defaults to 10000 (high exploration).
        * ``[11]`` Number of particles for the belief approximation. Defaults to  100.
   * ``[2]`` Dataset to use. Defaults to rd. Available options are
     * *fm* (foodmart recommandations) with following options
       * ``[3]`` Product discretization level. Defaults to 4.
       * ``[4]`` History length. Must be strictly greater than 1. Defaults to 2.
     * *mz* recommandations.
       * ``[3]`` Base name for the directory containing the corresponding MEMDP model parameters.
     * *rd* (synthetic data recommandations) 
       * ``[3]`` Number of actions. Defaults to 4.
       * ``[4]`` History length. Must be strictly greater than 1. Defaults to 2.
   * ``[6]`` Discount Parameter. Must be strictly between 0 and 1. Defaults to 0.95.
   * ``[9]`` Convergence criterion. Defaults to 0.01.
   * ``[-c]`` If present, recompile the code before running (*Note*: this should be used whenever using a dataset with different parameters as the number of items, environments etc are determined at compilation time).
   * ``[-p]`` If present, normalize the transition and use Kahan summation for more precision while handling small probabilities. Use this option if AIToolbox throws an ``Input transition table does not contain valid probabilities`` error.
   * ``[-v]`` If present, enables verbose output. In verbose mode, evaluation results per environments are displayed, and the std::cerr stream is eanbled during evaluation.

#### examples
**Example** *(synthetic recommandations, 10 environments, 10 actions, ~100 states)* :
```bash
cd Data/
python prepare_synth.py --norm --zip -n 10 -k 2 
cd ../Code/
./run.sh -m mdp -d rd -n 10 -k 2 -c
./run.sh -m pamcp -d rd -n 10 -k 2 -c
```

# known issues
  * When using the ``--zip`` option for data generation, it might be necessary to run the script with ``python3`` due to an [issue](https://bugs.python.org/issue23306) with the gzip library in python < 3.
