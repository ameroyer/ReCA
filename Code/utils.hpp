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


#ifndef UTILS_H_
#define UTILS_H_

#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <AIToolbox/MDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include "AIToolBox/MEMCP.hpp"
//#include <AIToolbox/POMDP/Algorithms/MEMCP.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/Types.hpp>
#include <AIToolbox/Utils.hpp>


/*!
 * Static parameters [set before compilation]
 */
static const int NITEMS =
#ifdef NITEMSPRM
  NITEMSPRM;
#undef NITEMSPRM
#else
3;
#endif        /*!< Number of items/products available */
static const int HIST =
#ifdef HISTPRM
  HISTPRM;
#undef HISTPRM
#else
2;
#endif        /*!< History length */
static const int NPROFILES =
#ifdef NPROFILESPRM
  NPROFILESPRM;
#undef NPROFILESPRM
#else
6;
#endif       /*!< Number of environments in the MEMDP */





/*!
 * Global variables
 */
const int hlength = HIST;             /*!< History length */
const size_t n_actions = NITEMS;      /*!< Number of items */
const size_t n_environments = NPROFILES;  /*!< Number of environments */
const size_t n_observations = (pow(NITEMS, HIST + 1) - 1) / (NITEMS - 1); /*!< Number of oservations */
const size_t n_states = NPROFILES * n_observations; /*!< Number of states in the MEMDP */


/*! \brief Asserts that the information contained in the summary file match the
 * parameters given at compilation time.
 *
 * \param sfile full path to the base_name.summary file.
 * \param mode respectively false for MDP and true for MEMDP.
 */
void check_summary_file(std::string sfile, bool mode);


/*! \brief Returns a string representation of the current system time.
 *
 * \return current time in a readable string format.
 */
std::string current_time_str();


/*! \brief Returns the number of actions (possible product
 * recommendations) in the MEMDP model.
 *
 * \return number of actions.
 */
size_t get_nactions();


/*! \brief Returns the number of environments in the MEMDP model.
 *
 * \return number of environments.
 */
size_t get_nenvironments();


/*! \brief Returns the number of observations in the MEMDP model.
 *
 * \return number of observations.
 */
size_t get_nobservations();


/*! \brief Returns the number of states in the underlying MDP model.
 *
 * \return number of states in the MDP.
 */
size_t get_nstates_MDP();


/*! \brief Returns the number of states in the MEMDP model.
 *
 * \return number of states in the MEMDP.
 */
size_t get_nstates_MEMDP();


/*!
 * \brief Given a state of the MEMDP, returns the corresponding environment.
 *
 * \param s a state in the MEMDP.
 *
 * \return the environment to which s belongs.
 */
size_t get_env(size_t s);


/*!
 * \brief Given a state of the MEMDP, returns its representative/observation.
 *
 * \param s a state in the MEMDP.
 *
 * \return the corresponding observation.
 */
size_t get_rep(size_t s);


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


/*! \brief Precomputes the ``n_actions`` exponents for conversion of decimals
 * to and from the ``n_actions`` base.
 */
void init_pows();


/*! \brief Returns the index of the state corresponding to a given sequence of item selections.
 * Note 1: Items correspond to actions with a +1 index shift, in order to allow the empty
 * selection to be represented by index 0.
 * Note 2: Items are ordered in decreasing age; the first time is the oldest in the history.
 *
 * \param state a state, represented by a sequence of selected items.
 *
 * \return the unique index representing the given state in the model.
 */
size_t state_to_id(std::vector<size_t> state);


/*! \brief Returns the sequence of items selection corresponding to the given state index.
 *
 * \param id unique state index.
 *
 * \return state a state, represented by a sequence of selected items.
 */
std::vector<size_t> id_to_state(size_t id);


/*! \brief Given a state and item choice, return the next user state.
 *
 * \param state unique state index.
 * \param item action chosen by the user [0 to n_actions - 1].
 *
 * \return next_state index of the state corresponding to the user choosing ``item`` in ``state``.
 */
size_t next_state(size_t state, size_t item);


/*! \brief Given two states s1 and s2, return the action a such that s2 = s1.a if it exists,
 * or the value ``n_actions`` otherwise.
 *
 * \param s1 unique state index.
 * \param s2 unique state index
 *
 * \return link a valid action index [0 to n_actions - 1] if s1 and s2 can be connected, n_actions otherwise.
 */
size_t is_connected(size_t s1, size_t s2);


/*! \brief Returns the best action (greedy strategy).
 *
 * \param action_scores vector mapping an action to its score.
 *
 * \return action with the best score.
 */
size_t get_prediction(std::vector<double> action_scores);


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


/*! \brief Returns accuracy and precision for the identification ability
 * of the model.
 *
 * \param scores vector mapping an environment to its score.
 * \param action the ground-truth environment.
 *
 * \return accuracy and average precision for the retrieved list.
 */
std::pair<double, double> identification_score(std::vector<size_t> sampleBelief, int cluster);


/*! \brief Pretty-printer for the results returned by one of the
 * evaluation routines.
 *
 * \param set_length contains the number of test sessions per cluster
 * \param results contains the various evaluation measures per cluster
 * \param titles contains the name of each evaluation measures
 */
void print_evaluation_result(int set_lengths[n_environments],
			     std::vector<double*> results,
			     std::vector<std::string> titles);


/*! \brief Evaluates a given policy (MDP) on a sequence of test user sessions
 * and prints the score for each user profiles (environments).
 *
 * \param sfile full path to the base_name.test file.
 * \param policy AIToolbox policy.
 * \param discount discount factor in the MDP model.
 * \param rewards stored reward values.
 */
void evaluate_policyMDP(std::string sfile,
			AIToolbox::MDP::Policy policy,
			double discount,
			double rewards [n_observations][n_actions]);


/*! \brief Builds the belief (distribution over states) correpsonding to the
 * given observation.
 *
 * \param o obsevation.
 */
AIToolbox::POMDP::Belief build_belief(size_t o);


/*! \brief Evaluates a given policy (POMDP) on a sequence of test user sessions
 * and prints the score for each user profiles (environments).
 *
 * \param sfile full path to the base_name.test file.
 * \param policy AIToolbox POMDP::policy.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for action sampling.
 * \param rewards stored reward values.
 */
void evaluate_policyMEMDP(std::string sfile,
			  AIToolbox::POMDP::Policy policy,
			  double discount,
			  unsigned int horizon,
			  double rewards [n_observations][n_actions]);


/*! \brief Returns a string representation of the internal tree for the POMCP algorithm.
 *
 * \param sfile full path to the base_name.test file.
 * \param pomcp AIToolbox pomcp instantiation.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for POMCP.
 * \param rewards stored reward values.
 */
template<typename M>
void pomcp_tree_to_string(AIToolbox::POMDP::POMCP< M > pomcp) {
  auto tree = pomcp.getGraph();
  for (size_t a = 0; a < n_actions; a++) {
    auto anode = tree.children[a];
    std::cout <<  " - " << a << "-> (" << anode.V << ")\n" ;
    std::cout << "      obs: ";
    for (auto b = anode.children.begin(); b != anode.children.end(); ++b) {
      std::cout << b->first << " ";;
    }
    std::cout << "\n";
  }
  //return;
}


/*! \brief Evaluates the sequence of actions recommended by POMCP.
 *
 * \param sfile full path to the base_name.test file.
 * \param pomcp AIToolbox pomcp instantiation.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for POMCP.
 * \param rewards stored reward values.
 */

template<typename M>
void evaluate_pomcp(std::string sfile,
		    AIToolbox::POMDP::POMCP<M> pomcp,
		    double discount,
		    unsigned int horizon,
		    double rewards [n_observations][n_actions])
{
  // Aux variables
  int cluster, session_length, chorizon;
  double cdiscount;
  double accuracy, precision, total_reward, discounted_reward, identity, identity_precision;
  size_t action, prediction;
  int user = 0;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [n_environments] = {0};
  double mean_accuracy [n_environments] = {0};
  double mean_precision [n_environments] = {0};
  double mean_total_reward [n_environments] = {0};
  double mean_discounted_reward [n_environments] = {0};
  double mean_identification [n_environments] = {0};
  double mean_identification_precision [n_environments] = {0};

  // For each user
  for (auto it = begin(aux); it != end(aux); ++it) {
    user++;
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    // update
    cluster = std::get<0>(*it);
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));
    set_lengths[cluster] += 1;

    // reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0, identity = 0, identity_precision = 0;
    cdiscount = discount;
    chorizon = horizon;
    std::vector< double > action_scores(n_actions, 0);

    // init belief
    AIToolbox::POMDP::Belief init_belief = AIToolbox::POMDP::Belief::Zero(n_states);
    size_t init_state = 0;
    for (int i = 0; i < n_environments; i++) {
      init_belief(i * n_observations + init_state) = 1.0 / n_environments;
    }
    prediction =  pomcp.sampleAction(init_belief, chorizon);
    action = n_actions;

    int i = 0;
    // For each (state, action) in the session
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      size_t observation  = std::get<0>(*it2);
      // If not init state, predict from past action and observation
      if (action < n_actions) {
       	prediction = pomcp.sampleAction(action, observation, chorizon);
      }
      // Get graph and action scores
      auto & graph_ = pomcp.getGraph();
      for (size_t a = 0; a < n_actions; a++) {
	action_scores.at(a) = graph_.children[a].V;
      }

      // Evaluate
      action = std::get<1>(*it2);
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      if (prediction == action) {
	total_reward += rewards[observation][prediction];
	discounted_reward += cdiscount * rewards[observation][prediction];
      }
      std::pair<double, double> aux = identification_score(pomcp.getGraph().belief, cluster);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
      cdiscount *= discount;
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1 );
    }
    // Set score
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
  print_evaluation_result(set_lengths, results, titles);
}




/*! \brief Evaluates the sequence of actions recommended by POMCP.
 *
 * \param sfile full path to the base_name.test file.
 * \param pomcp AIToolbox pomcp instantiation.
 * \param discount discount factor in the POMDP model.
 * \param horizon planning horizon for POMCP.
 * \param rewards stored reward values.
 */

template<typename M>
void evaluate_memcp(std::string sfile,
		    AIToolbox::POMDP::MEMCP<M> memcp,
		    double discount,
		    unsigned int horizon,
		    double rewards [n_observations][n_actions])
{
  // Aux variables
  int cluster, session_length, chorizon;
  double cdiscount;
  double accuracy, precision, total_reward, discounted_reward, identity, identity_precision;
  size_t action, prediction;
  int user = 0;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [n_environments] = {0};
  double mean_accuracy [n_environments] = {0};
  double mean_precision [n_environments] = {0};
  double mean_total_reward [n_environments] = {0};
  double mean_discounted_reward [n_environments] = {0};
  double mean_identification [n_environments] = {0};
  double mean_identification_precision [n_environments] = {0};

  // init belief
  AIToolbox::POMDP::Belief init_belief = AIToolbox::POMDP::Belief(n_environments);
  init_belief.fill(1.0 / n_environments);
  size_t init_state = 0;

  // For each user
  for (auto it = begin(aux); it != end(aux); ++it) {
    user++;
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    // update
    cluster = std::get<0>(*it);
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));
    set_lengths[cluster] += 1;

    // reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0, identity = 0, identity_precision = 0;
    cdiscount = discount;
    chorizon = horizon;
    std::vector< double > action_scores(n_actions, 0);

    // init belief
    prediction =  memcp.sampleAction(init_belief, init_state, chorizon, true);
    action = n_actions;

    int i = 0;
    // For each (state, action) in the session
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      size_t observation  = std::get<0>(*it2);
      // If not init state, predict from past action and observation
      if (action < n_actions) {
       	prediction = memcp.sampleAction(action, observation, chorizon);
      }
      // Get graph and action scores
      auto & graph_ = memcp.getGraph();
      for (size_t a = 0; a < n_actions; a++) {
	action_scores.at(a) = graph_.children[a].V;
      }

      // Evaluate
      action = std::get<1>(*it2);
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      if (prediction == action) {
	total_reward += rewards[observation][prediction];
	discounted_reward += cdiscount * rewards[observation][prediction];
      }
      std::pair<double, double> aux = identification_score(memcp.getGraph().belief, cluster);
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
      cdiscount *= discount;
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1 );
    }
    // Set score
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
  print_evaluation_result(set_lengths, results, titles);
}


#endif //UTILS_H_
