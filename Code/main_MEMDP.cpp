/* ---------------------------------------------------------------------------
** main_MEMDP.cpp
** This file contains the routine to load a POMDP parameters, solve and evaluate it.
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include <iostream>
#include <tuple>
#include <math.h>
#include <chrono>
#include "utils.hpp"

#include <AIToolbox/POMDP/IO.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>


/*
 * Global variables
 */
// Number of environments
extern const size_t n_environments;
// Number of actions available
extern const size_t n_actions;
// Number of observations
extern const size_t n_observations;
// Number of states
extern const size_t n_states;
// Discount factor
double discount;
// Random generator
static std::default_random_engine generator(time(NULL));


//T(s1, a, s2) = T(s1, a, connected[s1][s2]) if linked else 0
static double transition_matrix [n_environments][n_observations][n_actions][n_actions];
//R(s1, a, s2) = R(s1, connected[s1][s2]) if a == connected[s1][s2] else 0
static double rewards [n_observations][n_actions];


/*! \brief Loads the Model parameters from the precomputed data files.
 *
 * \param tfile Full path to the base_name.transitions file.
 * \param rfile Full path to the base_name.rewards file.
 * \param sfile Full path to the base_name.summary file.
 * \param precision Maximum precision while reading stored probabilities.
 */
void load_model_parameters(std::string tfile, std::string rfile,
			   std::string sfile, double precision) {

  // Variables
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  size_t s1, a, s2, link, p;
  double v;
  int links_found = 0, transitions_found = 0, profiles_found = 0;

  // Check summary file
  check_summary_file(sfile, true);

  
  // Load rewards
  infile.open(rfile, std::ios::in);
  assert((".rewards file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> s1 >> link >> s2 >> v)) { break; }
    assert(("Unvalid reward tuple", link > 0 && link <= n_actions));
    rewards[s1][link - 1] = v;
    links_found++;
  }
  assert(("Missing links while parsing .rewards file",
	  links_found == n_observations * n_actions));
  infile.close();
  
  // Load transitions
  infile.open(tfile, std::ios::in);
  assert((".transitions file not found", infile.is_open()));
  //double normalization [n_environments][n_observations][n_actions] = {0};
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    // Change profile
    if (!(iss >> s1 >> a >> s2 >> v)) {
      profiles_found += 1;
      assert(("Incomplete transition function in current profile in .transitions",
	      transitions_found == links_found * n_actions));
      assert(("Too many profiles found in .transitions file",
	      profiles_found <= n_environments));
      transitions_found = 0;
      continue;
    }
    // Set transition
    if (precision > 1) { v = std::trunc(v * precision); }
    link = is_connected(s1, s2);
    assert(("Unfeasible transition with >0 probability", link < n_actions));
    transition_matrix[profiles_found][s1][a - 1][link] = v;
    //normalization[profiles_found][s1][a - 1] += v;
    transitions_found++;
  }
  assert(("Missing profiles in .transitions file", profiles_found == n_environments));
  infile.close();
  
  // Normalize transition matrix [sparing memory]
  double nrm;
  for (p = 0; p < n_environments; p++) {
    for (s1 = 0; s1 < n_observations; s1++) {
      for (a = 0; a < n_actions; a++) {
	nrm = std::accumulate(transition_matrix[p][s1][a],
			      transition_matrix[p][s1][a] + n_actions, 0);
	//nrm = normalization[p][s1][a];
	std::transform(transition_matrix[p][s1][a],
		       transition_matrix[p][s1][a] + n_actions,
		       transition_matrix[p][s1][a],
		       [nrm](const double t){ return t / nrm ; }
		       );
      }
    }
  }
}


/*! \brief Class representing a recommender system as a MEMDP/POMDP and
 * implementing the AIToolbox::POMDP is_model and is_generative_model
 * interfaces.
 */
class RecoMEMDP {
public:
  /*! \brief Returns the number of states in the POMDP model.
   *
   * \return number of states in the MEMDP.
   */
  size_t getS() const { return n_states; }


  /*! \brief Returns the number of actions in the  POMDP model.
   *
   * \return number of actions in the POMDP.
   */
  size_t getA() const { return n_actions; }


  /*! \brief Returns the number of observations in the  POMDP model.
   *
   * \return number of actions in the POMDP.
   */
  size_t getO() const { return n_observations; }


  /*! \brief Returns the discount factor in the  MDP model.
   *
   * \return Discount factor in the MDP.
   */
  double getDiscount() const { return discount; }


  /*! \brief Returns a given transition probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return P( s2 | s1 -a-> ).
   */
  double getTransitionProbability( size_t s1, size_t a, size_t s2 ) const {
    size_t link = is_connected(get_rep(s1), get_rep(s2));
    if (get_env(s1) != get_env(s2) || link >= n_actions) {
      return 0.;
    } else {
      return transition_matrix[get_env(s1)][get_rep(s1)][a][link];
    }
  }


  /*! \brief Returns a given observation probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param o observation.
   *
   * \return P( o | s1 -a-> ).
   */
  double getObservationProbability(size_t s1, size_t a, size_t o) const {
    if (get_rep(s1) == o) {
      return 1.;
    } else {
      return 0.;
    }
  }

  /*! \brief Returns a given reward.
   *
   * \param s1 origin state.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return R(s1, a, s2).
   */
  double getExpectedReward( size_t s1, size_t a, size_t s2 ) const {
    size_t link = is_connected(get_rep(s1), get_rep(s2));
    if (get_env(s1) != get_env(s2) || link >= n_actions) {
      return 0.;
    } else {
      return rewards[get_rep(s1)][link];
    }
  }


  /*! \brief Sample a state and reward given an origin state and chosen acion.
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated reward R(s, a, s2).
   */
  std::tuple<size_t, double> sampleSR(size_t s,size_t a) const {
    // Sample random transition
    std::discrete_distribution<int> distribution (transition_matrix[get_env(s)][get_rep(s)][a], transition_matrix[get_env(s)][get_rep(s)][a] + n_actions);
    size_t link = distribution(generator);
    // Return values
    size_t s2 = get_env(s) * n_observations + next_state(get_rep(s), link);
    if (a == link) {
      return std::make_tuple(s2, rewards[get_rep(s)][link]);
    } else {
      return std::make_tuple(s2, 0);
    }
  }


  /*! \brief Sample a state and reward given an origin state and chosen acion.
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated reward R(s, a, s2).
   */
  std::tuple<size_t, size_t, double> sampleSOR(size_t s, size_t a) const {
    // Sample random transition
    std::discrete_distribution<int> distribution (transition_matrix[get_env(s)][get_rep(s)][a], transition_matrix[get_env(s)][get_rep(s)][a] + n_actions);
    size_t link = distribution(generator);
    // Return values
    size_t o2 = next_state(get_rep(s), link);
    size_t s2 = get_env(s) * n_observations + o2;
    if (a == link) {
      return std::make_tuple(s2, o2, rewards[get_rep(s)][link]);
    } else {
      return std::make_tuple(s2, o2, 0);
    }
  }


  /*! \brief Rwturns whether a state is terminal or not.
   *
   * \param s state
   *
   * \return whether the state s is terminal or not.
   */
  bool isTerminal(size_t) const {return false;}
};


/**
 * MAIN ROUTINE
 */
int main(int argc, char* argv[]) {

  // Parse input arguments
  assert(("Usage: ./main files_basename [solver] [discount] [nsteps] [precision]", argc >= 2));
  std::string algo = ((argc > 2) ? argv[2] : "pbvi");
  std::transform(algo.begin(), algo.end(), algo.begin(), ::tolower);
  assert(("Unvalid POMDP solver parameter", !(algo.compare("ip") && algo.compare("pomcp") && algo.compare("memcp"))));
  discount = ((argc > 3) ? std::atof(argv[3]) : 0.95);
  assert(("Unvalid discount parameter", discount > 0 && discount < 1));
  int steps = ((argc > 4) ? std::atoi(argv[4]) : 1000000);
  assert(("Unvalid steps parameter", steps > 0));
  unsigned int horizon = ((argc > 5) ? std::atoi(argv[5]) : 1);
  assert(("Unvalid horizon parameter", horizon > 0));
  double epsilon = ((argc > 6) ? std::atof(argv[6]) : 0.01);
  assert(("Unvalid convergence criterion", epsilon >= 0));
  double exp = ((argc > 7) ? std::atof(argv[7]) : 10000);
  assert(("Unvalid exploration parameter", exp >= 0));
  unsigned int beliefSize = ((argc > 8) ? std::atoi(argv[8]) : 100);
  assert(("Unvalid belief size", beliefSize >= 0));
  int precision = ((argc > 9) ? std::atoi(argv[9]) : 10);
  assert(("Unvalid precision parameter", precision >= 0));


  // Load model parameters
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Loading model parameters\n";
  assert(("Usage: ./main Param_basename [Discount] [solver steps]", argc >= 2));
  std::string datafile_base = std::string(argv[1]);
  init_pows();
  load_model_parameters(datafile_base + ".transitions",
			datafile_base + ".rewards",
			datafile_base + ".summary",
			std::pow(10, precision));
  double loading_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  //return 0;
  // Assert correct sizes
  assert(("Error in TRANSITION_MATRIX initialization",
	  sizeof(transition_matrix)/sizeof(****transition_matrix) ==
	  n_states * n_actions * n_actions));
  assert(("Error in REWARDS initialization",
	  sizeof(rewards) / sizeof(**rewards) == n_observations * n_actions));
  assert(("Out of range discount parameter", discount > 0 && discount <= 1));

  // Init Sparse Model in AIToolbox
  start = std::chrono::high_resolution_clock::now();
  RecoMEMDP world;
  std::cout << "\n" << current_time_str() << " - Copying model [sparse]...!\n";
  AIToolbox::POMDP::SparseModel<decltype(world)> model(world);

  // Training
  double training_time, testing_time;
  start = std::chrono::high_resolution_clock::now();
  std::cout << current_time_str() << " - Init " << algo << " solver...!\n";

  // Evaluation
  // POMCP
  if (!algo.compare("pomcp")) {
    AIToolbox::POMDP::POMCP<decltype(model)> solver(model, beliefSize, steps, exp);
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    start = std::chrono::high_resolution_clock::now();
    std::cout << current_time_str() << " - Starting evaluation!\n";
    evaluate_pomcp(datafile_base + ".test", solver, discount, horizon, rewards);
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }
  // MEMCP
  else if (!algo.compare("memcp")) {
    AIToolbox::POMDP::MEMCP<decltype(model)> solver(model, n_environments, beliefSize, steps, exp);
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    start = std::chrono::high_resolution_clock::now();
    std::cout << current_time_str() << " - Starting evaluation!\n";
    evaluate_memcp(datafile_base + ".test", solver, discount, horizon, rewards);
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }
  // Incremental Pruning
  else if (!algo.compare("ip")) {
    AIToolbox::POMDP::IncrementalPruning solver(horizon, epsilon);
    auto solution = solver(model);
    std::cout << current_time_str() << " - Convergence criterion reached: " << std::boolalpha << std::get<0>(solution) << "\n";
    std::chrono::high_resolution_clock::now() - start;
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;

    // Build and Evaluate Policy
    start = std::chrono::high_resolution_clock::now();
    std::cout << "\n" << current_time_str() << " - Evaluation results\n";
    AIToolbox::POMDP::Policy policy(world.getS(), world.getA(), world.getO(), std::get<1>(solution));
    evaluate_policyMEMDP(datafile_base + ".test", policy, discount, horizon, rewards);
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }

  // Output Times
  std::cout << current_time_str() << " - Timings\n";
  std::cout << "   > Loading : " << loading_time << "s\n";
  std::cout << "   > Training : " << training_time << "s\n";
  std::cout << "   > Testing : " << testing_time << "s\n";

  // Save policy in file
  /*
   * TODO
   */

  // End
  return 0;

}
