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
#include "model.hpp"
#include "recomodel.hpp"
#include "mazemodel.hpp"

/**
 * MAIN ROUTINE
 */
int main(int argc, char* argv[]) {
  // Parse input arguments
  assert(("Usage: ./main file_basename data_mode [Discount] [nsteps] [precision]", argc >= 3));
  std::string data = argv[2];
  assert(("Unvalid data mode", !(data.compare("reco") && data.compare("maze"))));
  double discount = ((argc > 3) ? std::atof(argv[3]) : 0.95);
  assert(("Unvalid discount parameter", discount > 0 && discount < 1));
  int steps = ((argc > 4) ? std::atoi(argv[4]) : 1000000);
  assert(("Unvalid steps parameter", steps > 0));
  float epsilon = ((argc > 5) ? std::atof(argv[5]) : 0.01);
  assert(("Unvalid epsilon parameter", epsilon >= 0));
  bool precision = ((argc > 6) ? (atoi(argv[6]) == 1) : false);
  bool verbose = ((argc > 7) ? (atoi(argv[7]) == 1) : false);
  Model model;

  // Create model
  auto start = std::chrono::high_resolution_clock::now();
  std::string datafile_base = std::string(argv[1]);
  if (!data.compare("reco")) {
    model = Recomodel(datafile_base + ".summary", discount, true);
    model.load_rewards(datafile_base + ".rewards");
    model.load_transitions(datafile_base + ".transitions", precision, datafile_base + ".profiles");
  } else if (!data.compare("maze")) {
    model = Mazemodel(datafile_base + ".summary", discount);
    model.load_rewards(datafile_base + ".rewards");
    // TODO
  }
  assert(("Model does not enable MDP mode", model.mdp_enabled));
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double loading_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.;

  // Solve
  start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Starting MDP ValueIteration solver\n";
  AIToolbox::MDP::ValueIteration<decltype(model)> solver(steps, epsilon);
  auto solution = solver(model);
  std::cout << current_time_str() << " - Convergence criterion e = " << epsilon << " reached ? " << std::boolalpha << std::get<0>(solution) << "\n";
  elapsed = std::chrono::high_resolution_clock::now() - start;
  double training_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.;

  // Build and Evaluate Policy
  start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Evaluation results\n";
  AIToolbox::MDP::Policy policy(model.getO(), model.getA(), std::get<1>(solution));
  evaluate_policyMDP(datafile_base + ".test", model, policy, verbose);
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
