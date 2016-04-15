#ifndef UTILS_H_
#define UTILS_H_
/* ---------------------------------------------------------------------------
** utils.hpp
** This files contains functions related to the conversion from states
** (item sequences) to and from indices in the MDP models, as well as
** functions for the evaluation procedure.
**
** NOTE: Call function init_pows once before using the state-index conversions.
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <AIToolbox/MDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include "AIToolBox/MEMCP.hpp"
#include "model.hpp"



/*! \brief Returns a string representation of the current system time.
 *
 * \return current time in a readable string format.
 */
std::string current_time_str();


/*! \brief Returns a sequence of sessions and corresponding user
 * profile for evaluation. Sessions are loaded from the corresponding
 * base_name.test file.
 *
 * \param sfile full path to the base_name.test file.
 *
 * \return vector of test sessions where a session is of the
 * form (environment_id, vector of (state, action pairs).
 */
std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > load_test_sessions(std::string sfile);


/*! \brief Pretty-printer for the results returned by one of the
 * evaluation routines.
 *
 * \param set_lengths contains the number of test sessions per cluster
 * \param n_environments length of n_environments
 * \param results contains the various evaluation measures per cluster
 * \param titles contains the name of each evaluation measures
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
void print_evaluation_result(int* set_lengths,
			     int n_environments,
			     std::vector<double*> results,
			     std::vector<std::string> titles,
			     bool verbose /* = false*/);


/*! \brief Returns a 0-1 accuracy score given a prediction and ground-truth.
 *
 * \param predicted the predicted action.
 * \param action the ground-truth action.
 *
 * \return 1 if the prediction matches the ground-truth, otherwise 0.
 */
double accuracy_score(size_t predicted, size_t action);


/*! \brief Returns an average precision for a retrieval list of actions.
 *
 * \param action_scores vector mapping an action to its score.
 * \param action the ground-truth action.
 *
 * \return average precision for the retrieved list.
 */
double avprecision_score(std::vector<double> action_scores, size_t action);


/*! \brief Returns accuracy and precision for the identification ability for a set of belief particles.
 *
 * \param sampleBelief current belief particles of the model.
 * \param cluster ground-truth identity of the current user.
 * \param model MEMDP model.
 *
 * \return accuracy and average precision for the retrieved list.
 */
std::pair<double, double> identification_score_particles(std::vector<size_t> sampleBelief, int cluster, const Model& model);


/*! \brief Evaluates a given policy (MDP) on a sequence of test user sessions
 * and prints the score for each user profiles (environments).
 *
 * \param sfile full path to the base_name.test file.
 * \param policy AIToolbox policy.
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
void evaluate_policyMDP(std::string sfile,
			const Model& model,
			AIToolbox::MDP::Policy policy,
			bool verbose=false);


/*! \brief Builds a belief over environments corresponding to the
 * given observation.
 *
 * \param o observation.
 * \param n_states total number of states.
 * \param n_observations total number of observations.
 * \param n_environments total number of environments.
 */
AIToolbox::POMDP::Belief build_belief(size_t o, size_t n_states, size_t n_observations, size_t n_environments);


/*! \brief Evaluates a given policy (POMDP) on a sequence of test user sessions
 * and prints the score for each user profiles (environments).
 *
 * \param sfile full path to the base_name.test file.
 * \param policy AIToolbox POMDP::policy.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for action sampling.
 * \param rewards stored reward values.
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
void evaluate_policyMEMDP(std::string sfile,
			  const Model& model,
			  AIToolbox::POMDP::Policy  policy,
			  unsigned int horizon,
			  bool verbose /* =false */,
			  bool supervised /* =true */);

void evaluate_policy_interactiveMEMDP(int n_sessions,
				      const Model& model,
				      AIToolbox::POMDP::Policy policy,
				      unsigned int horizon,
				      bool verbose /* =false */,
				      bool supervised /* =true */);


/*! \brief Evaluates the sequence of actions recommended by POMCP.
 *
 * \param sfile full path to the base_name.test file.
 * \param pomcp AIToolbox pomcp instantiation.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for POMCP.
 * \param rewards stored reward values.
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
template<typename M>
void evaluate_pomcp(std::string sfile,
		    const Model& model,
		    AIToolbox::POMDP::POMCP<M> pomcp,
		    unsigned int horizon,
		    bool verbose=false,
		    bool supervised=true)
{
  // Aux variables
  size_t observation, action, prediction;
  int cluster, session_length, chorizon, user = 0;
  double cdiscount, accuracy, precision, total_reward, discounted_reward, identity, identity_precision;

  // Initialize arrays
  int set_lengths [model.getE()] = {0};
  double mean_accuracy [model.getE()] = {0};
  double mean_precision [model.getE()] = {0};
  double mean_total_reward [model.getE()] = {0};
  double mean_discounted_reward [model.getE()] = {0};
  double mean_identification [model.getE()] = {0};
  double mean_identification_precision [model.getE()] = {0};

  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  for (auto it = begin(aux); it != end(aux); ++it) {
    // Identity
    user++;
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;

    // Reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0, identity = 0, identity_precision = 0;
    cdiscount = 1.;
    chorizon = horizon;
    std::vector< double > action_scores(model.getA(), 0);

    // Init belief
    size_t init_state = 0;
    AIToolbox::POMDP::Belief init_belief = build_belief(init_state, model.getS(), model.getO(), model.getE());
    prediction = pomcp.sampleAction(init_belief, chorizon);

    if (!verbose) {std::cerr.setstate(std::ios_base::failbit);}
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      // Update reward
      if (!model.isInitial(std::get<0>(*it2)) && (prediction == action)) {
	double r = model.getExpectedReward(cluster * model.getO() + observation, prediction, cluster * model.getO() + std::get<0>(*it2));
	total_reward += r;
	discounted_reward += cdiscount * r;
      }
      cdiscount *= model.getDiscount();
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1 );

      // Predict
      observation  = std::get<0>(*it2);
      if (!model.isInitial(observation)) {
       	prediction = (supervised ? pomcp.sampleAction(action, observation, chorizon) : pomcp.sampleAction(prediction, observation, chorizon));
      }

      // Get graph and action scores
      auto & graph_ = pomcp.getGraph();
      for (size_t a = 0; a < model.getA(); a++) {
	action_scores.at(a) = graph_.children[a].V;
      }

      // Evaluate
      action = std::get<1>(*it2);
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      auto aux = identification_score_particles(pomcp.getGraph().belief, cluster, model);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
    }

    if (!verbose) {std::cerr.clear();}
    mean_accuracy[cluster] += accuracy / session_length;
    mean_precision[cluster] += precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
    mean_identification[cluster] += identity / session_length;
    mean_identification_precision[cluster] += identity_precision / session_length;
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"acc", "avgpr", "avgrw", "discrw", "idac", "idpr"};
  std::vector<double*> results {mean_accuracy, mean_precision, mean_total_reward, mean_discounted_reward, mean_identification, mean_identification_precision};
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
}


/*! \brief Evaluates the sequence of actions recommended by POMCP.
 *
 * \param sfile full path to the base_name.test file.
 * \param pomcp AIToolbox pomcp instantiation.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for POMCP.
 * \param rewards stored reward values.
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
template<typename M>
void evaluate_memcp(std::string sfile,
		    const Model& model,
		    AIToolbox::POMDP::MEMCP<M> memcp,
		    unsigned int horizon,
		    bool verbose=false,
		    bool supervised=true)
{
  // Aux variables
  size_t observation, action, prediction;
  int cluster, session_length, chorizon, user = 0;
  double cdiscount, accuracy, precision, total_reward, discounted_reward, identity, identity_precision;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [model.getE()] = {0};
  double mean_accuracy [model.getE()] = {0};
  double mean_precision [model.getE()] = {0};
  double mean_total_reward [model.getE()] = {0};
  double mean_discounted_reward [model.getE()] = {0};
  double mean_identification [model.getE()] = {0};
  double mean_identification_precision [model.getE()] = {0};

  // Init belief over the environments
  AIToolbox::POMDP::Belief env_belief = AIToolbox::POMDP::Belief(model.getE());
  env_belief.fill(1.0 / model.getE());
  size_t init_state = 0;

  for (auto it = begin(aux); it != end(aux); ++it) {
    // Identity
    user++;
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));

    // Reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0, identity = 0, identity_precision = 0;
    cdiscount = 1.;
    chorizon = horizon;
    std::vector< double > action_scores(model.getA(), 0);
    prediction =  memcp.sampleAction(env_belief, init_state, chorizon, true);

    if (!verbose) {std::cerr.setstate(std::ios_base::failbit);}
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      // Update reward
      if (!model.isInitial(std::get<0>(*it2)) && (prediction == action)) {
	double r = model.getExpectedReward(cluster * model.getO() + observation, prediction, cluster * model.getO() + std::get<0>(*it2));
	total_reward += r;
	discounted_reward += cdiscount * r;
      }
      cdiscount *= model.getDiscount();
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1 );

      // Predict
      observation  = std::get<0>(*it2);
      if (observation != init_state) {
       	prediction = (supervised ? memcp.sampleAction(action, observation, chorizon) : memcp.sampleAction(prediction, observation, chorizon));
      }

      // Get graph and action scores
      auto & graph_ = memcp.getGraph();
      for (size_t a = 0; a < model.getA(); a++) {
	action_scores.at(a) = graph_.children[a].V;
      }

      // Evaluate
      action = std::get<1>(*it2);
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      auto aux = identification_score_particles(memcp.getGraph().belief, cluster, model);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
    }

    if (!verbose) {std::cerr.clear();}
    mean_accuracy[cluster] += accuracy / session_length;
    mean_precision[cluster] += precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
    mean_identification[cluster] += identity / session_length;
    mean_identification_precision[cluster] += identity_precision / session_length;
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"acc", "avgpr", "avgrw", "discrw", "idac", "idpr"};
  std::vector<double*> results {mean_accuracy, mean_precision, mean_total_reward, mean_discounted_reward, mean_identification, mean_identification_precision};
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
}


/**

template<typename M>
void evaluate_memcp_interactive(int n_sessions,
		    const Model& model,
		    AIToolbox::POMDP::MEMCP<M> memcp,
		    unsigned int horizon,
		    bool verbose=false,
		    bool supervised=true)
{
  // Aux variables
  size_t observation, action, prediction, state;
  int cluster, session_length, chorizon, user = 0;
  double cdiscount, reward;
  double steps = 0, identity = 0, identity_precision = 0;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [model.getE()] = {0};
  double mean_steps [model.getE()] = {0};
  double mean_identification [model.getE()] = {0};
  double mean_identification_precision [model.getE()] = {0};

  // Init belief over the environments
  AIToolbox::POMDP::Belief env_belief = AIToolbox::POMDP::Belief(model.getE());
  env_belief.fill(1.0 / model.getE());
  size_t init_state = 0;

  for (int run = 0; run < n_sessions; run++) {
    // Identity
    cluster = rand() % (int)(model.getE());
    set_lengths[cluster] += 1;
    steps = 0
    user++;

    // Reset
    steps = 0, identity = 0, identity_precision = 0;
    cdiscount = 1.;
    chorizon = horizon;
    std::cerr << "\r     User " << run << "/" << n_sessions << std::flush;

    // Init
    std::vector< double > action_scores(model.getA(), 0);
    prediction =  memcp.sampleAction(env_belief, init_state, chorizon, true);
    state = cluster * model.getO() + 0;

    // Run
    while(!model.isTerminal(state)) {
      std::tie(state, observation, reward) = model.sampleSOR(state, prediction);
      prediction = memcp.sampleAction(prediction, observation, chorizon);
      auto aux = identification_score_particles(memcp.getGraph().belief, cluster, model);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
      steps++;
    }

  // Print results for each environment, as well as global result
    mean_steps[cluster] += steps;
    mean_identification[cluster] += identity / steps;
    mean_identification_precision[cluster] += identity_precision / steps;
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"steps", "idac", "idpr"};
  std::vector<double*> results {mean_steps, mean_identification, mean_identification_precision};
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
  }*/

#endif //UTILS_H_
