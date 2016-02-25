/* ---------------------------------------------------------------------------
** utils.cpp
** See utils.hpp for a description
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/



#include "utils.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>


/**
 * CHECK_SUMMARY_FILE
 */

void check_summary_file(std::string sfile, bool mode) {
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  size_t aux;

  // Load summary file
  infile.open(sfile, std::ios::in);
  assert((".summary file not found", infile.is_open()));
  // Check number of states
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  assert(("Number of states do not match", n_observations == aux));
  // Check number of actions
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  assert(("Number of actions do not match", n_actions == aux));
  // Check number of environments
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  assert(("Number of environments do not match", n_environments == aux));

  // Summary
  if (!mode) { // MDP
    std::cout << "   -> The model contains " << n_actions << " actions\n";
    std::cout << "   -> The model contains " << n_observations << " states\n";
  } else { // MEMDP
    std::cout << "   -> The model contains " << n_observations << " observations\n";
    std::cout << "   -> The model contains " << n_actions << " actions\n";
    std::cout << "   -> The model contains " << n_states << " states\n";
    std::cout << "   -> The model contains " << n_environments << " environments\n";
  }

  // End
  infile.close();
}


/**
 * CURRENT_TIME_STR
 */
std::string current_time_str() {
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, 80, "%d-%m-%Y %I:%M:%S", timeinfo);
  return std::string(buffer);
}


/**
 * GET_NACTIONS
 */
size_t get_nactions() {
  return n_actions;
}


/**
 * GET_NENVIRONMENTS
 */
size_t get_nenvironments() {
  return n_environments;
}


/**
 * GET_NOBSERVATIONS
 */
size_t get_nobservations() {
  return n_observations;
}

/**
 * GET_NSTATES_MDP
 */
size_t get_nstates_MDP() {
  return n_observations;
}


/**
 * GET_NSTATES_MEMDP
 */
size_t get_nstates_MEMDP() {
  return n_states;
}


/**
 * GET_ENV
 */
size_t get_env(size_t s) {
  return s / n_observations;
}


/**
 * GET_REP
 */
size_t get_rep(size_t s) {
  return s % n_observations;
}


/**
 * Static
 */
static int pows[hlength];   /*!< Precomputed exponents for conversion to base n_items */
static int acpows[hlength]; /*!< Cumulative exponents for conversion from base n_items */


/**
 * INIT_POWS
 */
void init_pows() {
  pows[hlength - 1] = 1;
  acpows[hlength - 1] = 1;
  for (int i = hlength - 2; i >= 0; i--) {
    pows[i] = pows[i + 1] * n_actions;
    acpows[i] = acpows[i + 1] + pows[i];
  }
}


/**
 * STATE_TO_ID
 */

size_t state_to_id(std::vector<size_t> state) {
  size_t id = 0;
  for (int i = 0; i < hlength; i++) {
    id += state.at(i) * pows[i];
  }
  return id;
}


/**
 * ID_TO_STATE
 */
std::vector<size_t> id_to_state(size_t id) {
  std::vector<size_t> state (hlength);
  int indx = 0;
  while (id > n_actions) {
    div_t divresult = div(id, pows[indx]);
    if (divresult.rem < acpows[indx + 1]) {
      state.at(indx) = divresult.quot - 1;
      id = pows[indx] + divresult.rem;
    } else  {
      state.at(indx) = divresult.quot;
      id = divresult.rem;
    }
    indx++;
  }
  state.at(hlength - 1) = id;
  return state;
}


/**
 * NEXT_STATE
 */
size_t next_state(size_t state, size_t item) {
  size_t aux = state % pows[0];
  if (aux >= acpows[1] || state < pows[0]) {
    return aux * n_actions + item + 1;
  } else {
    return (pows[0] + aux) * n_actions + item + 1;
  }
}


/**
 * IS_CONNECTED
 */
size_t is_connected(size_t s1, size_t s2) {
  // Find suffix of first state
  int suffix_s1 = s1 % pows[0];
  suffix_s1 = ((suffix_s1 >= acpows[1] || s1 < pows[0])  ? suffix_s1 - acpows[1] : suffix_s1 + pows[0] - acpows[1]);
  // Find prefix of second state
  div_t aux = div(s2, n_actions);
  int prefix_s2 = aux.quot - acpows[1];
  size_t last_s2 = aux.rem - 1;
  if (aux.rem == 0) {
    prefix_s2 -= 1;
    last_s2 = n_actions - 1;
  }
  // Check connexion
  //std::cout << "State " << s1 << " to " << s2 << ": " << suffix_s1 << " =? " << prefix_s2 << " and " << last_s2 << "\n";
  if (prefix_s2 == suffix_s1) {
    return last_s2;
  } else {
    return n_actions;
  }
}


/**
 * LOAD_TEST_SESSIONS
 */
std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > load_test_sessions(std::string sfile) {
  // Variables
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  int n_env = 0, cluster, user;
  size_t s, a;
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > test_sessions;

  // Load test sessions file
  infile.open(sfile, std::ios::in);
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::vector<std::pair<size_t, size_t> > aux;
    iss >> user >> cluster;
    if (cluster > n_env) {
      n_env = cluster;
    }
    while ( iss >> s >> a ) {
      aux.push_back(std::make_pair(s, a - 1));
    }
    test_sessions.push_back(std::make_pair(cluster, aux));
  }
  assert(("number of environments do not match the clustering in test sessions",
  	  n_environments == n_env + 1));
  return test_sessions;
}


/**
 * GET_PREDICTION
 */
size_t get_prediction(std::vector<double> action_scores) {
  return std::distance(action_scores.begin(),
		       max_element(action_scores.begin(),
				   action_scores.end())
		       );
}


/**
 * ACCURACY_SCORE
 */
double accuracy_score(size_t predicted, size_t action) {
  if (action == predicted) {
    return 1.0;
  } else {
    return 0.0;
  }
}


/**
 * AVPRECISION_SCORE
 */
double avprecision_score(std::vector<double> action_scores, size_t action) {
  double value = action_scores[action];
  float rank = 0;
  for (auto it = begin(action_scores); it != end(action_scores); ++it) {
    if ( *it >= value) {
      rank += 1;
    }
  }
  return 1.0 / rank;
}


/**
 * IDENTIFICATION_SCORE
 */
std::pair<double, double> identification_score(std::vector<size_t> sampleBelief, int cluster) {
  // Build scores per cluster
  std::vector<int> scores(n_environments);
  for (auto it = begin(sampleBelief); it != end(sampleBelief); ++it) {
    scores.at(get_env(*it))++;
  }
  // Accuracy
  double accuracy = ((std::max_element(scores.begin(), scores.end()) - scores.begin() == cluster) ? 1.0 : 0.0);
  // Precision
  int rank = 0.;
  double value = scores.at(cluster);
  for (auto it = begin(scores); it != end(scores); ++it) {
    if ( *it >= value) {
      rank += 1;
    }
  }
  // Return
  return std::make_pair(accuracy, 1.0 / rank);
}


/**
 * PRINT_EVALUATION_RESULT
 */
void print_evaluation_result(int set_lengths[n_environments],
			     std::vector<double*> results,
			     std::vector<std::string> titles)
{

  // Print results for each environment, as well as global result
  int n_results = results.size();
  int session_length = 0;
  std::vector<double> acc(n_results, 0);

  std::cout << "> Results by cluster ----------------\n";
  for (int i = 0; i < n_environments; i++) {
    std::cout << "   cluster " << i;
    for (int j = 0; j < n_results; j++) {
      acc.at(j) += results.at(j)[i];
      std::cout << "\n      > " << titles[j] << ": " << results.at(j)[i] / set_lengths[i];
    }
    session_length += set_lengths[i];
    std::cout << "\n\n";
  }

  std::cout << "> Global results ----------------";
  for (int j = 0; j < n_results; j++) {
    std::cout << "\n      > " << titles[j] << ": " << acc.at(j) / session_length;
  }
  std::cout << "\n\n";
}


/**
 * EVALUATE_POLICYMDP
 */
void evaluate_policyMDP(std::string sfile,
			AIToolbox::MDP::Policy policy,
			double discount,
			double rewards [n_observations][n_actions]) {
  // Aux variables
  int cluster, session_length;
  double cdiscount;
  double accuracy, precision, total_reward, discounted_reward;
  int user = 0;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [n_environments] = {0};
  double mean_accuracy [n_environments] = {0};
  double mean_precision [n_environments] = {0};
  double mean_total_reward [n_environments] = {0};
  double mean_discounted_reward [n_environments] = {0};

  // For each user
  for (auto it = begin(aux); it != end(aux); ++it) {
    user++;
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    // Update
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));

    // Reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0;
    cdiscount = discount;

    // For each (state, action) in the session
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      size_t state = std::get<0>(*it2), action = std::get<1>(*it2);
      std::vector< double > action_scores = policy.getStatePolicy(state);
      size_t prediction = get_prediction(action_scores);

      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      if (prediction == action) {
	total_reward += rewards[state][prediction];
	discounted_reward += cdiscount * rewards[state][prediction];
      }
      cdiscount *= discount;
    }
    // Accumulate
    mean_accuracy[cluster] += accuracy / session_length;
    mean_precision[cluster] += precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"acc", "avgpr", "avgrw", "discrw"};
  std::vector<double*> results {mean_accuracy, mean_precision, mean_total_reward, mean_discounted_reward};
  print_evaluation_result(set_lengths, results, titles);
}


/**
 * BUILD_BELIEF
 */
AIToolbox::POMDP::Belief build_belief(size_t o) {
  AIToolbox::POMDP::Belief belief = AIToolbox::POMDP::Belief::Zero(n_states);
  for (int i = 0; i < n_environments; i++) {
    belief(i * n_observations + o) = 1.0 / n_environments;
  }
  return belief;
}


/**
 * EVALUATE_POLICYMEMDP
 */
void evaluate_policyMEMDP(std::string sfile,
			  AIToolbox::POMDP::Policy policy,
			  double discount,
			  unsigned int horizon,
			  double rewards [n_observations][n_actions]) {
  // Aux variables
  int cluster, session_length;
  double cdiscount;
  double accuracy, precision, total_reward, discounted_reward;
  int user = 0;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [n_environments] = {0};
  double mean_accuracy [n_environments] = {0};
  double mean_precision [n_environments] = {0};
  double mean_total_reward [n_environments] = {0};
  double mean_discounted_reward [n_environments] = {0};
  std::vector< double > action_scores(n_actions, 0);

  // For each user
  for (auto it = begin(aux); it != end(aux); ++it) {
    user++;
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    // Update
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));

    // Reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0;
    cdiscount = discount;

    // For each (state, action) in the session
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      // current state
      size_t state = std::get<0>(*it2), action = std::get<1>(*it2);

      // get a prediction
      AIToolbox::POMDP::Belief belief = build_belief(state);
      for (size_t a = 0; a < n_actions; a++) {
	//action_scores.at(a) = 1. / n_actions;
	//action_scores.at(a) = policy.getActionProbability (belief, a, horizon);
	action_scores.at(a) = policy.getActionProbability (belief, a);
      }
      size_t prediction = get_prediction(action_scores);

      // evaluate
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      if (prediction == action) {
	total_reward += rewards[state][prediction];
	discounted_reward += cdiscount * rewards[state][prediction];
      }
      cdiscount *= discount;
    }

    // accumulate
    mean_accuracy[cluster] += accuracy / session_length;
    mean_precision[cluster] += precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"acc", "avgpr", "avgrw", "discrw"};
  std::vector<double*> results {mean_accuracy, mean_precision, mean_total_reward, mean_discounted_reward};
  print_evaluation_result(set_lengths, results, titles);
}
