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
 * STATS
 */
Stats::Stats(int s) {
    size = s;
    acc_mean = new double[size]();
    acc_var = new double[size]();
    lengths = new double[size]();
  }


Stats::~Stats() {
  //delete []acc_mean;
  //delete []acc_var;
  //delete []lengths;
  }


void Stats::update(int cluster, double v) {
    assert(("overflow error", cluster < size));
    acc_mean[cluster] += v;
    acc_var[cluster] += v * v;
    lengths[cluster] += 1;
  }

double Stats::get_mean(int cluster) {
    if (cluster >= 0) {
      if (lengths[cluster] > 0) {
	return acc_mean[cluster] / lengths[cluster];
      } else {
	return 0.;
      }
    } else {
      double v, l = 0;
      for (int i = 0; i < size; i++) {
	v += acc_mean[i];
	l += lengths[i];
      }
      return v / l;
    }
  }

double Stats::get_var(int cluster) {
    if (cluster >= 0) {
      if (lengths[cluster] > 0) {
	double mean = get_mean(cluster);
	return acc_var[cluster] / lengths[cluster] - mean * mean;
      } else {
	return 0.;
      }
    } else {
      double v = 0;
      for (int i = 0; i < size; i++) {
	v += get_var(i);
      }
      return v / size;
    }
  }

double Stats::get_std(int cluster) {
    return sqrt(get_var(cluster));
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
 * PRINT_EVALUATION_RESULT
 */
void print_evaluation_result(int n_environments,
			     std::vector<Stats> results,
			     std::vector<std::string> titles,
			     bool verbose /* = false*/)
{
  // Print results for each environment, as well as global result
  int n_results = results.size();
  std::vector<double> acc(n_results, 0);
  if (verbose) { std::cout << "> Results by cluster ----------------\n";}
  for (int i = 0; i < n_environments; i++) {
    if (verbose) { std::cout << "   cluster " << i;}
    for (int j = 0; j < n_results; j++) {
      if (verbose) { std::cout << "\n      > " << titles[j] << ": " << results.at(j).get_mean(i) << " +/- " << results.at(j).get_std(i);}
    }
    if (verbose) {std::cout << "\n\n";}
  }

  // Global
  std::cout << "> Global results ----------------";
  for (int j = 0; j < n_results; j++) {
    std::cout << "\n      > " << titles[j] << ": " << results.at(j).get_mean(-1) << " +/- " << results.at(j).get_std(-1);
  }
}

/**
 * MAKE_INITIAL_PREDICTION (POMDP policy)
 */
std::pair<AIToolbox::POMDP::Belief, size_t> make_initial_prediction(const Model& model, AIToolbox::POMDP::Policy &policy, int horizon, std::vector<double> &action_scores) {
  size_t init_observation = 0;
  AIToolbox::POMDP::Belief belief = build_belief(init_observation, model.getS(), model.getO(), model.getE());
  size_t id, prediction;
  std::tie(prediction, id) = policy.sampleAction(belief, horizon);

  return std::make_pair(belief, prediction);
};

/**
 * MAKE_INITIAL_PREDICTION (MDP policy)
 */
std::pair<AIToolbox::POMDP::Belief, size_t> make_initial_prediction(const Model& model, AIToolbox::MDP::Policy &policy, int horizon, std::vector<double> &action_scores) {
  size_t init_observation = 0;
  AIToolbox::POMDP::Belief belief = build_belief(init_observation, 0, 0, 0);
  action_scores = policy.getStatePolicy(init_observation);
  size_t prediction = get_prediction(action_scores);

  return std::make_pair(belief, prediction);
};

/**
 * MAKE_PREDICTION (POMDP policy)
 */
size_t make_prediction(const Model& model, AIToolbox::POMDP::Policy &policy, AIToolbox::POMDP::Belief &b, size_t o, size_t a, int horizon, std::vector<double> &action_scores) {
  b = update_belief(b, a, o, model);
  size_t id, prediction;
  std::tie(prediction, id) = policy.sampleAction(b, horizon);

  return prediction;
}

/**
 * MAKE_PREDICTION (MDP policy)
 */
size_t make_prediction(const Model& model, AIToolbox::MDP::Policy &policy, AIToolbox::POMDP::Belief &b, size_t o, size_t a, int horizon, std::vector<double> &action_scores) {
  action_scores = policy.getStatePolicy(o);
  return get_prediction(action_scores);
}

/**
 * IDENTIFICATION_SCORE (POMDP policy)
 */
std::pair<double, double> identification_score(const Model& model, AIToolbox::POMDP::Policy policy, AIToolbox::POMDP::Belief b, size_t o, int cluster) {
  std::vector<double> scores(model.getE());
  for (int e = 0; e < model.getE(); e++) {
    scores.at(e) = b(e * model.getO() + o);
  }
  double accuracy = ((std::max_element(scores.begin(), scores.end()) - scores.begin() == cluster) ? 1.0 : 0.0);
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
 * IDENTIFICATION_SCORE (MDP policy)
 */
std::pair<double, double> identification_score(const Model& model, AIToolbox::MDP::Policy policy, AIToolbox::POMDP::Belief b, size_t o, int cluster) {
  return std::make_pair(-1., -1.);
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

/**
 * UPDATE_BELIEF
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
