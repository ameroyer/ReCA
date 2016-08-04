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

/*! \brief Builds a belief over environments corresponding to the
 * given observation.
 *
 * \param o observation.
 * \param n_states total number of states.
 * \param n_observations total number of observations.
 * \param n_environments total number of environments.
 */
AIToolbox::POMDP::Belief build_belief(size_t o, size_t n_states, size_t n_observations, size_t n_environments);

/*! \brief Belief update for a MEMDP.
 *
 * \param b current belief.
 * \param a last action taken.
 * \param o observation seen after applying a.
 */
AIToolbox::POMDP::Belief update_belief(AIToolbox::POMDP::Belief b, size_t a, size_t o, const Model& model);

/*! \brief Returns the initial prediction and belief for a given model and solver.
 *
 * \param model the underlying model.
 * \param solver the solver to evaluate (MDP policy, POMDP policy, POMCP or MEMCP).
 * \param horizon the horizon to predict for, if applicable.
 * \param action_scores array to store the probability distribution over predictions, if applicable.
 *
 * \return belief the initial belief over states (or over environments for MEMCP).
 * \return prediction the initial prediction.
 */
// MDP
std::pair<AIToolbox::POMDP::Belief, size_t> make_initial_prediction(const Model& model, AIToolbox::MDP::Policy &policy, int horizon, std::vector<double> &action_scores);

// POMDP policy
std::pair<AIToolbox::POMDP::Belief, size_t> make_initial_prediction(const Model& model, AIToolbox::POMDP::Policy &policy, int horizon, std::vector<double> &action_scores);

// POMCP
template<typename M>
std::pair<AIToolbox::POMDP::Belief, size_t> make_initial_prediction(const Model& model, AIToolbox::POMDP::POMCP<M> &pomcp, int horizon, std::vector<double> &action_scores) {
  size_t init_observation = 0;
  AIToolbox::POMDP::Belief belief = build_belief(init_observation, model.getS(), model.getO(), model.getE());
  size_t prediction = pomcp.sampleAction(belief, horizon);

  auto & graph_ = pomcp.getGraph();
  for (size_t a = 0; a < model.getA(); a++) {
    action_scores.at(a) = graph_.children[a].V;
  }

  return std::make_pair(belief, prediction);
}

// MEMCP
template<typename M>
std::pair<AIToolbox::POMDP::Belief, size_t> make_initial_prediction(const Model& model, AIToolbox::POMDP::MEMCP<M> &memcp, int horizon, std::vector<double> &action_scores) {
  size_t init_observation = 0;
  AIToolbox::POMDP::Belief env_belief = AIToolbox::POMDP::Belief(model.getE());
  env_belief.fill(1.0 / model.getE());
  size_t prediction = memcp.sampleAction(env_belief, init_observation, horizon, true);

  auto & graph_ = memcp.getGraph();
  for (size_t a = 0; a < model.getA(); a++) {
    action_scores.at(a) = graph_.children[a].V;
  }

  return std::make_pair(env_belief, prediction);
}

/*! \brief Returns the prediction of the solver for a given action and observation -a-> o.
 *
 * \param model the underlying model.
 * \param solver the solver to evaluate (MDP policy, POMDP policy, POMCP or MEMCP).
 * \param b current belief.
 * \param o last seen observation.
 * \param a last action.
 * \param horizon the horizon to predict for, if applicable.
 * \param action_scores array to store the probability distribution over predictions, if applicable.
 *
 * \return belief the initial belief over states (or over environments for MEMCP).
 * \return prediction the initial prediction.
 */
//MDP
size_t make_prediction(const Model& model, AIToolbox::MDP::Policy &policy, AIToolbox::POMDP::Belief &b, size_t o, size_t a, int horizon, std::vector<double> &action_scores);

// POMDP policy
size_t make_prediction(const Model& model, AIToolbox::POMDP::Policy &policy, AIToolbox::POMDP::Belief &b, size_t o, size_t a, int horizon, std::vector<double> &action_scores);

// POMCP
template<typename M>
size_t make_prediction(const Model& model, AIToolbox::POMDP::POMCP<M> &pomcp, AIToolbox::POMDP::Belief &b, size_t o, size_t a, int horizon, std::vector<double> &action_scores) {
  size_t prediction = pomcp.sampleAction(a, o, horizon);
  auto & graph_ = pomcp.getGraph();
  for (size_t action = 0; action < model.getA(); action++) {
    action_scores.at(action) = graph_.children[action].V;
  }
  return prediction;
}

// MEMCP
template<typename M>
size_t make_prediction(const Model& model, AIToolbox::POMDP::MEMCP<M> &memcp, AIToolbox::POMDP::Belief &b, size_t o, size_t a, int horizon, std::vector<double> &action_scores) {
  size_t prediction = memcp.sampleAction(a, o, horizon);
  auto & graph_ = memcp.getGraph();
  for (size_t action = 0; action < model.getA(); action++) {
    action_scores.at(action) = graph_.children[action].V;
  }
  return prediction;
}

/*! \brief Returns the accuracy and prediction for the profile detection of the solver in the current state of the simulation..
 *
 * \param model the underlying model.
 * \param solver the solver to evaluate (MDP policy, POMDP policy, POMCP or MEMCP).
 * \param current belief.
 * \param o last seen observation.
 * \param cluster ground-truth cluster.
 *
 * \return acc the accuracy score.
 * \return prec the precision score.
 */
// MDP policy
std::pair<double, double> identification_score(const Model& model, AIToolbox::MDP::Policy policy, AIToolbox::POMDP::Belief b, size_t o, int cluster);

// POMDP policy
std::pair<double, double> identification_score(const Model& model, AIToolbox::POMDP::Policy policy, AIToolbox::POMDP::Belief b, size_t o, int cluster);

// POMCP
template<typename M>
std::pair<double, double> identification_score(const Model& model, AIToolbox::POMDP::POMCP<M> pomcp, AIToolbox::POMDP::Belief b, size_t o, int cluster) {
  std::vector<size_t> sampleBelief = pomcp.getGraph().belief;
  std::vector<int> scores(model.getE());
  for (auto it = begin(sampleBelief); it != end(sampleBelief); ++it) {
    scores.at(model.get_env(*it))++;
  }
  double accuracy = ((std::max_element(scores.begin(), scores.end()) - scores.begin() == cluster) ? 1.0 : 0.0);
  int rank = 0.;
  double value = scores.at(cluster);
  for (auto it = begin(scores); it != end(scores); ++it) {
    if ( *it >= value) {
      rank += 1;
    }
  }
  return std::make_pair(accuracy, 1.0 / rank);
}

// MEMCP
template<typename M>
std::pair<double, double> identification_score(const Model& model, AIToolbox::POMDP::MEMCP<M> memcp, AIToolbox::POMDP::Belief b, size_t o, int cluster) {
  std::vector<size_t> sampleBelief = memcp.getGraph().belief;
  std::vector<int> scores(model.getE());
  for (auto it = begin(sampleBelief); it != end(sampleBelief); ++it) {
    scores.at(model.get_env(*it))++;
  }
  double accuracy = ((std::max_element(scores.begin(), scores.end()) - scores.begin() == cluster) ? 1.0 : 0.0);
  int rank = 0.;
  double value = scores.at(cluster);
  for (auto it = begin(scores); it != end(scores); ++it) {
    if ( *it >= value) {
      rank += 1;
    }
  }
  return std::make_pair(accuracy, 1.0 / rank);
}

/*! \brief Evaluates a given solver using external test sequences (sequence of (observation, action)) stored in a file.
 *
 * \param sfile full path to the base_name.test file.
 * \param model underlying MEMDP model.
 * \param solver the solver to be evaluated.
 * \param policy AIToolbox POMDP::policy.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for action sampling.
 * \param rewards stored reward values.
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
template<typename M>
void evaluate_from_file(std::string sfile,
			const Model& model,
			M solver,
			unsigned int horizon,
			bool verbose=false,
			bool supervised=true) {
  // Aux variables
  size_t observation = 0, action, prediction;
  int user = 0, cluster, session_length, chorizon;
  double cdiscount, accuracy, precision, total_reward, discounted_reward, identity, identity_precision;

  // Initialize arrays
  AIToolbox::POMDP::Belief belief;
  std::vector< double > action_scores(model.getA(), 0);
  int set_lengths [model.getE()] = {0};
  double mean_accuracy [model.getE()] = {0};
  double mean_precision [model.getE()] = {0};
  double mean_total_reward [model.getE()] = {0};
  double mean_discounted_reward [model.getE()] = {0};
  double mean_identification [model.getE()] = {0};
  double mean_identification_precision [model.getE()] = {0};
  double total_length = 0.;

  // Load test sessions
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  for (auto it = begin(aux); it != end(aux); ++it) {
    // Identity
    user++;
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    total_length += session_length;
    assert(("Empty test user session", session_length > 0));
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;

    // Reset
    cdiscount = 1.;
    chorizon = horizon;
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0, identity = 0, identity_precision = 0;
    std::vector< double > action_scores(model.getA(), 0);

    // Make initial guess
    std::tie(belief, prediction) = make_initial_prediction(model, solver, chorizon, action_scores);
    if (!verbose) {std::cerr.setstate(std::ios_base::failbit);}
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      // Update
      if (!model.isInitial(std::get<0>(*it2))) {
	double r = (model.mdp_enabled() ? model.getExpectedReward(observation, prediction, std::get<0>(*it2)) : model.getExpectedReward(cluster * model.getO() + observation, prediction, cluster * model.getO() + std::get<0>(*it2)));
	total_reward += r;
	discounted_reward += cdiscount * r;
      }
      cdiscount *= model.getDiscount();
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1 );

      // Predict
      observation  = std::get<0>(*it2);
      if (!model.isInitial(observation)) {
	prediction = make_prediction(model, solver, belief, observation, (supervised ? action : prediction), chorizon, action_scores);
      }

      // Evaluate
      action = std::get<1>(*it2);
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      auto aux = identification_score(model, solver, belief, observation, cluster);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
    }

    // Update scores
    if (!verbose) {std::cerr.clear();}
    mean_accuracy[cluster] += accuracy / session_length;
    mean_precision[cluster] += precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
    mean_identification[cluster] += identity / session_length;
    mean_identification_precision[cluster] += identity_precision / session_length;
  }

  // Only output relevant metrics
  bool has_identity = (identity >= 0);
  bool has_total_reward = (model.getDiscount() < 1);

  // Output
  std::cout << "\n\n";
  std::vector<std::string> titles {"discrw", "acc", "avgpr"}; std::vector<double*> results {mean_discounted_reward, mean_accuracy, mean_precision};

  if (has_total_reward) {
    titles.insert(titles.begin(), "avgrw");
    results.insert(results.begin(), mean_total_reward);
  }
  if (has_identity) {
    titles.push_back("idac"); titles.push_back("idpr");
    results.push_back(mean_identification); results.push_back(mean_identification_precision);
  }
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
  std::cout << "\n      > avglng: " << total_length / user;
  std::cout << "\n\n";
}

/*! \brief Evaluates a given solver on on-the-fly generated test sequences.
 *
 * \param sfile full path to the base_name.test file.
 * \param model underlying MEMDP model.
 * \param solver the solver to be evaluated.
 * \param policy AIToolbox POMDP::policy.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for action sampling.
 * \param rewards stored reward values.
 * \param verbose if true, increases the verbosity. Defaults to false.
 */
template<typename M>
void evaluate_interactive(int n_sessions,
			  const Model& model,
			  M solver,
			  unsigned int horizon,
			  bool verbose=false,
			  bool supervised=false, //true only works if full policy is computed (i.e. pbvi)
			  int session_length_max=400) {
  // Aux variables
  size_t observation = 0, prev_observation, action, prediction;
  size_t state, prev_state;
  int cluster, chorizon;
  double r, session_length, total_reward, identity, identity_precision;

  // Initialize arrays
  AIToolbox::POMDP::Belief belief;
  std::vector< double > action_scores(model.getA(), 0);
  int n_failures = 0;
  int set_lengths [model.getE()] = {0};
  double mean_session_length [model.getE()] = {0};
  double mean_success [model.getE()] = {0};
  double mean_total_reward [model.getE()] = {0};
  double mean_goal_reward [model.getE()] = {0};
  double mean_identification [model.getE()] = {0};
  double mean_identification_precision [model.getE()] = {0};

  // Generate test sessions
  int subgroup_size = n_sessions / (int)(model.getE());
  n_sessions = n_sessions - n_sessions % (int)(model.getE());
  for (int user = 0; user < n_sessions; user++) {
    cluster = user / subgroup_size;
    //cluster = (model.mdp_enabled() ? 0 : rand() % (int)(model.getE()));
    set_lengths[cluster] += 1;
    std::cerr << "\r     User " << user + 1 << "/" << n_sessions << std::string(15, ' ');

    // Reset
    chorizon = horizon;
    session_length = 0, total_reward = 0, identity = 0, identity_precision = 0;
    std::vector< double > action_scores(model.getA(), 0);

    // Make initial guess
    state = cluster * model.getO() + 0;
    std::tie(belief, prediction) = make_initial_prediction(model, solver, chorizon, action_scores);
    if (!verbose) {std::cerr.setstate(std::ios_base::failbit);}
    while(!model.isTerminal(state) && session_length < session_length_max) {
      // Sample next state
      prev_state = state;
      std::tie(state, observation, r) = model.sampleSOR(state, prediction);
      // Update
      total_reward += r;
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1 );
      // Predict
      prediction = make_prediction(model, solver, belief, observation, (supervised ? model.is_connected(prev_state, state) : prediction), horizon, action_scores);

      // Evaluate
      session_length++;
      auto aux = identification_score(model, solver, belief, observation, cluster);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
    }

    // Update scores
    if (!verbose) {std::cerr.clear();}
    // Not reaching anything
    if (!model.isTerminal(state)) {
      if (verbose) {
	std::cerr << " run " << user + 1 << " ignored: did not reach final state.";
      }
      set_lengths[cluster] -= 1;
      n_failures += 1;
      continue;
    }

    // id score
    mean_identification[cluster] += identity / session_length;
    mean_identification_precision[cluster] += identity_precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    // If Trap, do not count the rest
    if (model.get_rep(state) != 1) { 
      continue;
    }
    // Normal execution, i.e. goal state
    mean_session_length[cluster] += session_length;
    mean_success[cluster] += ((model.get_rep(state) == 1) ? 1. : 0.); // Goal in robot maze
    mean_goal_reward[cluster] += total_reward / session_length;
  }

  // Only output relevant metrics
  bool has_identity = (identity >= 0);

  // Output
  std::cout << "\n\n";
  std::vector<std::string> titles {"goalrw", "avgrw", "avgllng", "avgsuc"}; std::vector<double*> results {mean_goal_reward, mean_total_reward, mean_session_length, mean_success};

  if (has_identity) {
    titles.push_back("idac"); titles.push_back("idpr");
    results.push_back(mean_identification); results.push_back(mean_identification_precision);
  }
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
  std::cout << "\n      > " << n_failures << " / " << n_sessions << " reach failures\n";
  std::cout << "\n\n";
}
#endif
