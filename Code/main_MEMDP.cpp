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
  assert(("Unvalid discount parameter", discount > 0 && discount < 1));
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
  auto start = std::chrono::high_resolution_clock::now();
  std::string datafile_base = std::string(argv[1]);
  //Recomodel model = NULL;
  //if (!data.compare("reco")) {
    Recomodel model(datafile_base + ".summary", discount, false);
    model.load_rewards(datafile_base + ".rewards");
    model.load_transitions(datafile_base + ".transitions", precision, datafile_base + ".profiles");
    //model = m;
    //} else if (!data.compare("maze")) {
    //Mazemodel m(datafile_base + ".summary", discount);
    //m.load_rewards(datafile_base + ".rewards");
    //m.load_transitions(datafile_base + ".transitions", precision);
    //model = &m;
    // TODO
    //}
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  double loading_time = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000000.;
  //std::cout << model->mdp_enabled() << "\n";
  // return 0;

  // Training
  double training_time, testing_time;
  start = std::chrono::high_resolution_clock::now();
  std::cout << "\n" << current_time_str() << " - Starting " << algo << " solver...!\n";

  // Evaluation
  // POMCP
  if (!algo.compare("pomcp")) {
    AIToolbox::POMDP::POMCP<Model> solver( model, beliefSize, steps, exp);
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    start = std::chrono::high_resolution_clock::now();
    std::cout << current_time_str() << " - Starting evaluation!\n";
    evaluate_pomcp(datafile_base + ".test", model, solver, horizon, verbose);
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }
  // MEMCP
  // TODO MEMCP remove model.getE pqrqmeter
  else if (!algo.compare("memcp")) {
    AIToolbox::POMDP::MEMCP<Recomodel> solver( model, beliefSize, steps, exp);
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
    start = std::chrono::high_resolution_clock::now();
    std::cout << current_time_str() << " - Starting evaluation!\n";
    evaluate_memcp(datafile_base + ".test", model, solver, horizon, verbose);
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }
  // Incremental Pruning
  else if (!algo.compare("pbvi")) {
    // TODO DEBUG PBVI //nBelef = n observations ?
    AIToolbox::POMDP::PBVI solver(beliefSize, horizon, epsilon);    
    if (!verbose) {std::cerr.setstate(std::ios_base::failbit);}
    auto solution = solver(model);
    if (!verbose) {std::cerr.clear();}
    std::cout << current_time_str() << " - Convergence criterion reached: " << std::boolalpha << std::get<0>(solution) << "\n";
    std::chrono::high_resolution_clock::now() - start;
    training_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;

    // Build and Evaluate Policy
    start = std::chrono::high_resolution_clock::now();
    std::cout << "\n" << current_time_str() << " - Evaluation results\n";
    AIToolbox::POMDP::Policy policy(model.getS(), model.getA(), model.getO(), std::get<1>(solution));
    evaluate_policyMEMDP(datafile_base + ".test", model, policy, horizon, verbose, true);
    testing_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000000.;
  }

  // Output Times
  std::cout << current_time_str() << " - Timings\n";
  std::cout << "   > Loading : " << loading_time << "s\n";
  std::cout << "   > Training : " << training_time << "s\n";
  std::cout << "   > Testing : " << testing_time << "s\n";

  // Save policy or pomcp seach tree in file
  /*
   * TODO
   */

  // End
  return 0;

}
