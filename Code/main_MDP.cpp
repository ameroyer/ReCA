/* ---------------------------------------------------------------------------
** main_MDP.cpp
** This file contains the routine to load a MDP parameters, solve and evaluate it.
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include <iostream>
#include <tuple>
#include <math.h>
#include <chrono>
#include "utils.hpp"

#include <AIToolbox/MDP/IO.hpp>
#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

/*
 * Global variables
 */

// Number of actions available
extern const size_t n_actions;
// Number of observations (= states in the MDP)
extern const size_t n_observations;
// Discount factor
double discount;
// Random generator
static std::default_random_engine generator(time(NULL));


//T(s1, a, s2) = T(s1, a, connected[s1][s2]) if linked else 0
double transition_matrix [n_observations][n_actions][n_actions] = {0};
//R(s1, a, s2) = R(s1, connected[s1][s2]) if a == connected[s1][s2] else 0
double rewards [n_observations][n_actions];


/*! \brief Loads the Model parameters from the precomputed data files.
 *
 * \param tfile Full path to the base_name.transitions file.
 * \param rfile Full path to the base_name.rewards file.
 * \param sfile Full path to the base_name.summary file.
 * \param precision Maximum precision while reading stored probabilities.
 */
void load_model_parameters(std::string tfile, std::string rfile,
			   std::string pfile, std::string sfile, bool precision) {

  // Variables
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  size_t s1, s2, a, link;
  double v;
  int links_found = 0, transitions_found = 0;

  // Check summary file
  check_summary_file(sfile, false);

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

  // Load profiles proportions
  infile.open(pfile, std::ios::in);
  assert((".profiles file not found", infile.is_open()));
  std::vector<double> profiles_prop;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> s1 >> s2 >> v)) { break; }
    profiles_prop.push_back(v);
  }
  infile.close();

  // Load transitions
  infile.open(tfile, std::ios::in);
  assert((".transitions file not found", infile.is_open()));
  int n_profile = 0;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    // Profile change
    if (!(iss >> s1 >> a >> s2 >> v)) {
      n_profile += 1;
      assert(("Incomplete transition function in current profile in .transitions",
	      transitions_found == links_found * n_actions));
      transitions_found = 0;
      if (n_profile >= profiles_prop.size()) {
	break;
      } else {
	continue;
      }
    }
    // Accumulate
    link = is_connected(s1, s2);
    assert(("Unfeasible transition with >0 probability", link < n_actions));
    transition_matrix[s1][a - 1][link] += v;
    transitions_found++;
  }
  infile.close();

  // Normalize transition matrix
  double nrm;
  for (s1 = 0; s1 < n_observations; s1++) {
    for (a = 0; a < n_actions; a++) {
      // If asking for precision, use kahan summation [slightly slower]
      if (precision) {
	double kahan_correction = 0.0;
	nrm = 0.0;
	for (s2 = 0; s2 < n_actions; s2++) {
	  double val = transition_matrix[s1][a][s2] - kahan_correction;
	  double aux = nrm + val;
	  kahan_correction = (aux - nrm) - val;
	  nrm = aux;
	}
      }
      // Else basic sum
      else {
	nrm = std::accumulate(transition_matrix[s1][a],
			      transition_matrix[s1][a] + n_actions, 0.);
      }
      // Normalize
      std::transform(transition_matrix[s1][a],
		     transition_matrix[s1][a] + n_actions,
		     transition_matrix[s1][a],
		     [nrm](const double t){ return t / nrm; }
		     );
    }
  }

}



/*! \brief Class representing a recommender system as a MDP and
 * implementing the AIToolbox::MDP is_model and is_generative_model
 * interfaces.
 */
class RecoMDP {
public:
  /*! \brief Returns the number of states in the  MDP model.
   *
   * \return number of states in the MDP.
   */
  size_t getS() const { return n_observations; }


  /*! \brief Returns the number of actions in the  MDP model.
   *
   * \return number of actions in the MDP.
   */
  size_t getA() const { return n_actions; }


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
    size_t link = is_connected(s1, s2);
    if (link >= n_actions) {
      return 0.;
    } else {
      return transition_matrix[s1][a][link];
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
    size_t link = is_connected(s1, s2);
    if (link >= n_actions) {
      return 0.;
    } else {
      return rewards[s1][link];
    }
  }


  /*! \brief Sample a state and reward given an origin state and chosen action.
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated reward R(s, a, s2).
   */
  std::tuple<size_t, double> sampleSR(size_t s,size_t a) const {
    // Sample transition
    std::discrete_distribution<int> distribution (transition_matrix[s][a], transition_matrix[s][a] + n_actions);
    size_t link = distribution(generator);
    // Return values
    size_t s2 = next_state(s, link);
    if (a == link) {
      return std::make_tuple(s2, rewards[s][link]);
    } else {
      return std::make_tuple(s2, 0);
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
  assert(("Usage: ./main files_basename [Discount] [nsteps] [precision]", argc >= 2));
  discount = ((argc > 2) ? std::atof(argv[2]) : 0.95);
  assert(("Unvalid discount parameter", discount > 0 && discount < 1));
  int steps = ((argc > 3) ? std::atoi(argv[3]) : 1000000);
  assert(("Unvalid steps parameter", steps > 0));
  float epsilon = ((argc > 4) ? std::atof(argv[4]) : 0.01);
  assert(("Unvalid epsilon parameter", epsilon >= 0));
  bool precision = ((argc > 5) ? (atoi(argv[5]) == 1) : false);

  // Load model parameters
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Loading model parameters\n";
  std::string datafile_base = std::string(argv[1]);
  init_pows();
  load_model_parameters(datafile_base + ".transitions",
			datafile_base + ".rewards",
			datafile_base + ".profiles",
			datafile_base + ".summary", std::pow(10, precision));
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double loading_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.;

  // Assert correct sizes
  assert(("Error in TRANSITION_MATRIX initialization",
	  sizeof(transition_matrix)/sizeof(***transition_matrix) ==
	  n_observations * n_actions * n_actions));
  assert(("Error in REWARDS initialization",
	  sizeof(rewards) / sizeof(**rewards) == n_observations * n_actions));
  assert(("Out of range discount parameter", discount > 0 && discount <= 1));


  // Solve
  start = std::chrono::high_resolution_clock::now();
  std::cout << current_time_str() << " - Init solver...!\n";
  RecoMDP model;
  AIToolbox::MDP::ValueIteration<decltype(model)> solver(steps, epsilon);
  std::cout << current_time_str() << " - Starting solver!\n";
  auto solution = solver(model);
  std::cout << current_time_str() << " - Convergence criterion e = " << epsilon << " reached ? " << std::boolalpha << std::get<0>(solution) << "\n";
  elapsed = std::chrono::high_resolution_clock::now() - start;
  double training_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.;

  // Build and Evaluate Policy
  start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Evaluation results\n";
  AIToolbox::MDP::Policy policy(n_observations, n_actions, std::get<1>(solution));
  evaluate_policyMDP(datafile_base + ".test", policy, discount, rewards);
  elapsed = std::chrono::high_resolution_clock::now() - start;
  double testing_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.;

  // Output Times
  std::cout << current_time_str() << " - Timings\n";
  std::cout << "   > Loading : " << loading_time << "s\n";
  std::cout << "   > Training : " << training_time << "s\n";
  std::cout << "   > Testing : " << testing_time << "s\n";

  // Save policy in file
  std::cout << "\n" << current_time_str() << " - Saving policy\n";
  std::ofstream output(datafile_base + ".mdppolicy");
  output << policy;

  // End
  return 0;
}
