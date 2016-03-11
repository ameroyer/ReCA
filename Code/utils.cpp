/* ---------------------------------------------------------------------------
** utils.cpp
** See utils.hpp for a description
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include "utils.hpp"


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
 * LOAD_TEST_SESSIONS
 */
std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > load_test_sessions(std::string sfile) {
  // Variables
  std::string line;
  std::ifstream infile;
  std::istringstream iss;
  size_t s, a;
  int n_env = 0, cluster, user;
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > test_sessions;

  // Load test sessions file
  infile.open(sfile, std::ios::in);
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::vector<std::pair<size_t, size_t> > aux;
    iss >> user >> cluster;
    if (cluster > n_env) { n_env = cluster; }
    while ( iss >> s >> a ) {
      aux.push_back(std::make_pair(s, a - 1));
    }
    test_sessions.push_back(std::make_pair(cluster, aux));
  }

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
  return ((action == predicted) ? 1 : 0);
}


/**
 * AVPRECISION_SCORE
 */
double avprecision_score(std::vector<double> action_scores, size_t action) {
  float rank = 0;
  double value = action_scores[action];
  for (auto it = begin(action_scores); it != end(action_scores); ++it) {
    if ( *it >= value) {
      rank += 1;
    }
  }
  return 1.0 / rank;
}


/**
 * IDENTIFICATION_SCORE_PARTICLES
 */
std::pair<double, double> identification_score_particles(std::vector<size_t> sampleBelief, int cluster, const Model& model) {

  // Build scores per cluster
  std::vector<int> scores(model.getE());
  for (auto it = begin(sampleBelief); it != end(sampleBelief); ++it) {
    scores.at(model.get_env(*it))++;
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

  return std::make_pair(accuracy, 1.0 / rank);
}


/*! \brief Returns the accuracy and precision for the identification ability for a given belief.
 *
 * \param belief current belief of the model.
 * \param o last seen observatin.
 * \param cluster ground-truth identity o the current user.
 * \param n_environments total number of environments
 * \param n_observations total number of observations
 *
 * \return accuracy and average precision for the retrieved list.
 */
std::pair<double, double> identification_score_belief(AIToolbox::POMDP::Belief b, size_t o, int cluster, size_t n_environments, size_t n_observations) {

  // Build scores per cluster
  std::vector<double> scores(n_environments);
  for (int e = 0; e < n_environments; e++) {
    scores.at(e) = b(e * n_observations + o);
  }

  // Accuracy
  double accuracy = ((std::max_element(scores.begin(), scores.end()) - scores.begin() == cluster) ? 1.0 : 0.0);
  // Precision
  int rank = 1.;
  double value = scores.at(cluster);
  for (auto it = begin(scores); it != end(scores); ++it) {
    if ( *it > value) {
      rank += 1;
    }
  }

  return std::make_pair(accuracy, 1.0 / rank);
}


/**
 * PRINT_EVALUATION_RESULT
 */
void print_evaluation_result(int* set_lengths,
			     int n_environments,
			     std::vector<double*> results,
			     std::vector<std::string> titles,
			     bool verbose /* = false*/)
{
  // Print results for each environment, as well as global result
  int session_length = 0;
  int n_results = results.size();
  std::vector<double> acc(n_results, 0);
  if (verbose) { std::cout << "> Results by cluster ----------------\n";}
  for (int i = 0; i < n_environments; i++) {
    if (verbose) { std::cout << "   cluster " << i;}
    for (int j = 0; j < n_results; j++) {
      acc.at(j) += results.at(j)[i];
      if (verbose) { std::cout << "\n      > " << titles[j] << ": " << results.at(j)[i] / set_lengths[i];}
    }
    session_length += set_lengths[i];
    if (verbose) {std::cout << "\n\n";}
  }

  // Global
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
			const Model& model,
			AIToolbox::MDP::Policy policy,
			bool verbose /* = false*/) {
  // Aux variables
  size_t state, action, prediction;
  int cluster, session_length, user = 0;
  double cdiscount, accuracy, precision, total_reward, discounted_reward;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [model.getE()] = {0};
  double mean_accuracy [model.getE()] = {0};
  double mean_precision [model.getE()] = {0};
  double mean_total_reward [model.getE()] = {0};
  double mean_discounted_reward [model.getE()] = {0};

  for (auto it = begin(aux); it != end(aux); ++it) {
    // Identity
    user++;
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));

    // Reset
    accuracy = 0, precision = 0, total_reward = 0, discounted_reward = 0;
    cdiscount = model.getDiscount();

    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      // Predict
      std::tie(state, action) = *it2;
      std::vector< double > action_scores = policy.getStatePolicy(state);
      prediction = get_prediction(action_scores);

      // Evaluate
      accuracy += accuracy_score(prediction, action);
      precision += avprecision_score(action_scores, action);
      total_reward += model.getExpectedReward(state, prediction, model.next_state(state, action));
      discounted_reward += cdiscount * model.getExpectedReward(state, prediction, model.next_state(state, action));

      // Update
      cdiscount *= model.getDiscount();
    }

    mean_accuracy[cluster] += accuracy / session_length;
    mean_precision[cluster] += precision / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
    std::cout << user << " " << cluster << " " << accuracy / session_length << "\n";
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"acc", "avgpr", "avgrw", "discrw"};
  std::vector<double*> results {mean_accuracy, mean_precision, mean_total_reward, mean_discounted_reward};
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
}



/**
 * BUILD_BELIEF
 */
AIToolbox::POMDP::Belief build_belief(size_t o, size_t n_states, size_t n_observations, size_t n_environments) {
  AIToolbox::POMDP::Belief belief = AIToolbox::POMDP::Belief::Zero(n_states);
  for (int i = 0; i < n_environments; i++) {
    belief(i * n_observations + o) = 1.0 / n_environments;
  }
  return belief;
}


/*! \brief Belief update for our particular MEMDP structure.
 *
 * \param b current belief.
 * \param a last action taken.
 * \param o observation seen after applying a.
 */
AIToolbox::POMDP::Belief update_belief(AIToolbox::POMDP::Belief b, size_t a, size_t o, const Model& model) {
  AIToolbox::POMDP::Belief bp =  AIToolbox::POMDP::Belief::Zero(model.getS());
  double normalization = 0.;

  // Belief is non-zero only for states with observation o
  std::vector<size_t> prev = model.previous_states(o);
  for (int e = 0; e < model.getE(); e++) {
    size_t s = e * model.getO() + o;
    for (auto it = prev.begin(); it != prev.end(); ++it) {
      size_t pres = e * model.getO() + *it;
      bp(s) += model.getTransitionProbability(pres, a, s) * b(pres);
    }
    normalization += bp(s);
  }
  bp /= normalization;
  return bp;
}


/**
 * EVALUATE_POLICYMEMDP
 */
void evaluate_policyMEMDP(std::string sfile,
			  const Model& model,
			  AIToolbox::POMDP::Policy policy,
			  unsigned int horizon,
			  bool verbose /* =false */,
			  bool supervised /* =true */) {
  // Aux variables
  size_t id, prediction, action;
  int cluster, session_length, chorizon, user = 0;
  double cdiscount, accuracy, total_reward, discounted_reward,identity, identity_precision;

  // Initialize arrays
  std::vector<std::pair<int, std::vector<std::pair<size_t, size_t> > > > aux = load_test_sessions(sfile);
  int set_lengths [model.getE()] = {0};
  double mean_accuracy [model.getE()] = {0};
  double mean_total_reward [model.getE()] = {0};
  double mean_discounted_reward [model.getE()] = {0};
  double mean_identification [model.getE()] = {0};
  double mean_identification_precision [model.getE()] = {0};

  for (auto it = begin(aux); it != end(aux); ++it) {
    // Identity
    user++;
    cluster = std::get<0>(*it);
    set_lengths[cluster] += 1;
    session_length = std::get<1>(*it).size();
    assert(("Empty test user session", session_length > 0));

    // Reset
    accuracy = 0, total_reward = 0, discounted_reward = 0, identity = 0, identity_precision = 0;
    cdiscount = 1.;
    chorizon = horizon;

    // Initial belief and first action
    size_t init_state = 0;
    std::vector< double > action_scores(model.getA(), 0);
    AIToolbox::POMDP::Belief belief = build_belief(init_state, model.getS(), model.getO(), model.getE());
    std::tie(prediction, id) = policy.sampleAction(belief, chorizon);

    // For each (state, action) in the session
    std::cerr << "\r     User " << user << "/" << aux.size() << std::flush;
    for (auto it2 = begin(std::get<1>(*it)); it2 != end(std::get<1>(*it)); ++it2) {
      // Update
      cdiscount *= model.getDiscount();
      chorizon = ((chorizon > 1) ? chorizon - 1 : 1);

      // Predict
      size_t observation = std::get<0>(*it2);
      if (!model.isInitial(observation)) {
	belief = (supervised ? update_belief(belief, action, observation, model) : update_belief(belief, prediction, observation, model));
	std::tie(prediction, id) = policy.sampleAction(belief, chorizon);
      }
      action = std::get<1>(*it2);

      // Evaluate
      accuracy += accuracy_score(prediction, action);
      // TODO STATE RATHER THAN OBSEVRATION
      total_reward += model.getExpectedReward(observation, prediction, model.next_state(observation, action));
      discounted_reward += cdiscount * model.getExpectedReward(observation, prediction, model.next_state(observation, action));
      auto aux = identification_score_belief(belief, observation, cluster, model.getE(), model.getO());
      identity += std::get<0>(aux);
      identity_precision += std::get<1>(aux);
    }

    mean_accuracy[cluster] += accuracy / session_length;
    mean_total_reward[cluster] += total_reward / session_length;
    mean_discounted_reward[cluster] += discounted_reward;
    mean_identification[cluster] += identity / session_length;
    mean_identification_precision[cluster] += identity_precision / session_length;
  }

  // Print results for each environment, as well as global result
  std::cout << "\n\n";
  std::vector<std::string> titles {"acc", "avgrw", "discrw", "idac", "idpr"};
  std::vector<double*> results {mean_accuracy, mean_total_reward, mean_discounted_reward, mean_identification, mean_identification_precision};
  print_evaluation_result(set_lengths, model.getE(), results, titles, verbose);
}
