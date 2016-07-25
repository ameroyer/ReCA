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
#include "mazemodel.hpp"
#include "recomodel.hpp"

#include <AIToolbox/POMDP/IO.hpp>
#include "AIToolBox/PBVI.hpp"


template <typename M>
void mainMEMDP(M model, std::string datafile_base, std::string algo, int horizon, int steps, float epsilon, int beliefSize, float exp, bool precision, bool verbose, bool has_test) {
  // Training
  double training_time, testing_time;
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Starting " << algo << " solver...!\n";

  // Evaluation
  // POMCP
  if (!algo.compare("pomcp")) {
    AIToolbox::POMDP::POMCP<decltype(model)> solver( model, beliefSize, steps, exp);
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    start = std::chrono::high_resolution_clock::now();
    std::cout << current_time_str() << " - Starting evaluation!\n";
    if (has_test) {
      evaluate_from_file(datafile_base + ".test", model, solver, horizon, verbose);
    } else {
      evaluate_interactive(2000, model, solver, horizon, verbose);
    }
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }
  // MEMCP
  else if (!algo.compare("memcp")) {
    AIToolbox::POMDP::MEMCP<decltype(model)> solver( model, beliefSize, steps, exp);
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    start = std::chrono::high_resolution_clock::now();
    std::cout << current_time_str() << " - Starting evaluation!\n";
    std::cout << std::flush;
    std::cerr << std::flush;
    if (has_test) {
      evaluate_from_file(datafile_base + ".test", model, solver, horizon, verbose);
    } else {
      evaluate_interactive(2000, model, solver, horizon, verbose);
    }
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }
  // PBVI
  else if (!algo.compare("pbvi")) {
    AIToolbox::POMDP::PBVI solver(beliefSize, horizon, epsilon);
    if (!verbose) {std::cerr.setstate(std::ios_base::failbit);}
    auto solution = solver(model);
    if (!verbose) {std::cerr.clear();}
    std::cout << "\n" << current_time_str() << " - Convergence criterion reached: " << std::boolalpha << std::get<0>(solution) << "\n";
    int horizon_reached = std::get<2>(solution);
    std::cout << "Horizon " << horizon_reached << " reached\n";
    std::chrono::high_resolution_clock::now() - start;
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;

    // Build and Evaluate Policy
    start = std::chrono::high_resolution_clock::now();
    std::cout << "\n" << current_time_str() << " - Starting evaluation!\n";
    AIToolbox::POMDP::Policy policy(model.getS(), model.getA(), model.getO(), std::get<1>(solution));
    std::cout << std::flush;
    std::cerr << std::flush;
    if (has_test) {
      evaluate_from_file(datafile_base + ".test", model, policy, horizon_reached, verbose);
    } else {
      evaluate_interactive(2000, model, policy, horizon_reached, verbose);
    }
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }

  // Output Times
  std::cout << current_time_str() << " - Timings\n";
  std::cout << "   > Training : " << training_time << "s\n";
  std::cout << "   > Testing : " << testing_time << "s\n";
}


/**
 * MAIN ROUTINE
 */
int main(int argc, char* argv[]) {

  // Parse input arguments
  assert(("Usage: ./main file_basename data_mode [solver] [discount] [nsteps] [precision]", argc >= 3));
  std::string data = argv[2];
  assert(("Unvalid data mode", !(data.compare("reco") && data.compare("maze"))));
  std::string algo = ((argc > 3) ? argv[3] : "pbvi");
  std::transform(algo.begin(), algo.end(), algo.begin(), ::tolower);
  assert(("Unvalid POMDP solver parameter", !(algo.compare("pbvi") && algo.compare("pomcp") && algo.compare("memcp"))));
  double discount = ((argc > 4) ? std::atof(argv[4]) : 0.95);
  assert(("Unvalid discount parameter", discount > 0 && discount <= 1));
  int steps = ((argc > 5) ? std::atoi(argv[5]) : 1000000);
  assert(("Unvalid steps parameter", steps > 0));
  unsigned int horizon = ((argc > 6) ? std::atoi(argv[6]) : 1);
  assert(("Unvalid horizon parameter", ( !algo.compare("pbvi") && horizon > 1 ) || (algo.compare("pbvi") && horizon > 0)));
  double epsilon = ((argc > 7) ? std::atof(argv[7]) : 0.01);
  assert(("Unvalid convergence criterion", epsilon >= 0));
  double exp = ((argc > 8) ? std::atof(argv[8]) : 10000);
  assert(("Unvalid exploration parameter", exp >= 0));
  unsigned int beliefSize = ((argc > 9) ? std::atoi(argv[9]) : 100);
  assert(("Unvalid belief size", beliefSize >= 0));
  bool precision = ((argc > 10) ? (atoi(argv[10]) == 1) : false);
  bool verbose = ((argc > 11) ? (atoi(argv[11]) == 1) : false);

  // Create model
  std::string datafile_base = std::string(argv[1]);
  std::cout << "\n" << current_time_str() << " - Loading appropriate model\n";
  if (!data.compare("reco")) {
    Recomodel model (datafile_base + ".summary", discount, false);
    model.load_rewards(datafile_base + ".rewards");
    model.load_transitions(datafile_base + ".transitions", precision, datafile_base + ".profiles");
    mainMEMDP(model, datafile_base, algo, horizon, steps, epsilon, beliefSize, exp, precision, verbose, true);
  } else if (!data.compare("maze")) {
    if (discount < 1) {
      std::cout << "Setting undiscounted model";
      discount = 1.;
    }
    Mazemodel model(datafile_base + ".summary", discount);
    model.load_rewards(datafile_base + ".rewards");
    model.load_transitions(datafile_base + ".transitions", precision, verbose);
    mainMEMDP(model, datafile_base, algo, horizon, steps, epsilon, beliefSize, exp, precision, verbose, false);
  }
  return 0;

}
