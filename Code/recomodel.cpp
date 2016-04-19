/* ---------------------------------------------------------------------------
** recomodel.cpp
** see recomodel.hpp
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include "recomodel.hpp"
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <algorithm>

/**
 * RANDOM ENGINE
 */
std::default_random_engine Recomodel::generator(time(NULL));

/**
 * INDEX
 */
int Recomodel::index(size_t env, size_t s1, size_t a, size_t s2_link) const {
  return s2_link + n_actions * (a + n_actions * (s1 + n_observations * env));
}

/**
 * STATE_TO_ID
 */
size_t Recomodel::state_to_id(std::vector<size_t> state) const {
  size_t id = 0;
  for (int i = 0; i < hlength; i++) {
    id += state.at(i) * pows[i];
  }
  return id;
}

/**
 * ID_TO_STATE
 */
std::vector<size_t> Recomodel::id_to_state(size_t id) const {
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
 * STATE_TO_STRING
 */
std::string Recomodel::state_to_string(size_t s) const {
  std::vector<size_t> items = id_to_state(s);
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < items.size(); i++) {
    ss << items.at(i);
    if (i < items.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

/**
 * IS_CONNECTED
 */
size_t Recomodel::is_connected(size_t s1, size_t s2) const {
  // Check environments
  if (get_env(s1) != get_env(s2)) {
    return n_actions;
  } else if (!is_mdp) {
    s1 = get_rep(s1);
    s2 = get_rep(s2);
  }
  // Check if corresponding observations are connected
  // Suffix of s1
  int suffix_s1 = s1 % pows[0];
  suffix_s1 = ((suffix_s1 >= acpows[1] || s1 < pows[0])  ? suffix_s1 - acpows[1] : suffix_s1 + pows[0] - acpows[1]);
  // Prefix of s2
  div_t aux = div(s2, n_actions);
  int prefix_s2 = aux.quot - acpows[1];
  size_t last_s2 = aux.rem - 1;
  if (aux.rem == 0) {
    prefix_s2 -= 1;
    last_s2 = n_actions - 1;
  }
  return ((prefix_s2 == suffix_s1)? last_s2 : n_actions);
}

/**
 * NEXT_STATE
 */
size_t Recomodel::next_state(size_t state, size_t item) const {
  size_t obs = get_rep(state), env = get_env(state);
  size_t aux = obs % pows[0];
  if (aux >= acpows[1] || obs < pows[0]) {
    return env * n_observations + (aux * n_actions + item + 1);
  } else {
    return env * n_observations + ((pows[0] + aux) * n_actions + item + 1);
  }
}

/**
 * CONSTRUCTOR
 */
Recomodel::Recomodel(std::string sfile, double discount_, bool is_mdp_) {

  //********** Load summary information
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  size_t aux;
  infile.open(sfile, std::ios::in);
  assert((".summary file not found", infile.is_open()));
  // number of observations
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  n_observations = aux;
  // number of actions
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  n_actions = aux;
  // number of environments
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  n_environments = aux;
  // history length
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  hlength = aux;
  assert(("number of observations and actions do not match",
  	  (pow(n_actions, hlength + 1) - 1) / (n_actions - 1)));
  infile.close();

  //********** Initialize
  discount = discount_;
  is_mdp = is_mdp_;
  n_states = (is_mdp ? n_observations : n_environments * n_observations);
  rewards = new double[n_actions]();
  if (is_mdp) {
    transition_matrix = new double[n_observations * n_actions * n_actions]();
  } else {
    transition_matrix = new double[n_environments * n_observations * n_actions * n_actions]();
  }

  //********** Summary of model parameters
  if (is_mdp) { // MDP
    std::cout << "   -> The model contains " << n_actions << " actions\n";
    std::cout << "   -> The model contains " << n_observations << " states\n";
  } else { // MEMDP
    std::cout << "   -> The model contains " << n_observations << " observations\n";
    std::cout << "   -> The model contains " << n_actions << " actions\n";
    std::cout << "   -> The model contains " << n_states << " states\n";
    std::cout << "   -> The model contains " << n_environments << " environments\n";
  }

  //********** Precompute exponents for base conversion
  pows = new int[hlength];
  acpows = new int[hlength];
  pows[hlength - 1] = 1;
  acpows[hlength - 1] = 1;
  for (int i = hlength - 2; i >= 0; i--) {
    pows[i] = pows[i + 1] * n_actions;
    acpows[i] = acpows[i + 1] + pows[i];
  }
}

/**
 * DESTRUCTOR
 */
Recomodel::~Recomodel() {
  //delete []transition_matrix;
  //delete []rewards;
  //delete []pows;
  //delete []acpows;
}

/**
 * LOAD_REWARDS
 */
void Recomodel::load_rewards(std::string rfile) {
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  double v;
  size_t a;
  int rewards_found = 0;
  size_t s1, s2; // TODO

  infile.open(rfile, std::ios::in);
  assert((".rewards file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    //if (!(iss >> a >> v)) { break; } TODO
    if (!(iss >> s1 >> a >> s2 >> v)) { break; }
    assert(("Unvalid reward entry", a <= n_actions));
    rewards[a - 1] = v;
    rewards_found++;
  }
  //assert(("Missing item while parsing .rewards file",
  //	  rewards_found == n_actions)); TODO
  infile.close();
}


/**
 * LOAD_TRANSITIONS
 */
void Recomodel::load_transitions(std::string tfile, bool precision /* =false */, std::string pfile /* ="" */) {
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  double v;
  size_t s1, a, s2, link, p;
  int transitions_found = 0, profiles_found = 0;

  // If MDP mode, load profiles proportions for weighted average
  std::vector<double> profiles_prop;
  if (is_mdp) {
    infile.open(pfile, std::ios::in);
    assert((".profiles file not found", infile.is_open()));
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      if (!(iss >> s1 >> s2 >> v)) { break; }
      profiles_prop.push_back(v);
    }
    infile.close();
    assert(("Missing profiles in .profiles file", profiles_prop.size() == n_environments));
  }

  // Load transitions
  infile.open(tfile, std::ios::in);
  assert((".transitions file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    // Change of environment
    if (!(iss >> s1 >> a >> s2 >> v)) {
      profiles_found += 1;
      assert(("Incomplete transition function in current profile in .transitions",
	      transitions_found == n_observations * n_actions * n_actions));
      assert(("Too many profiles found in .transitions file",
	      profiles_found <= n_environments));
      transitions_found = 0;
      continue;
    }
    // Set transition probability
    link = is_connected(s1, s2);
    assert(("Unfeasible transition with >0 probability", link < n_actions));
    if (is_mdp) {
      transition_matrix[index(0, s1, a - 1, link)] += profiles_prop.at(profiles_found) * v;
    } else {
      transition_matrix[index(profiles_found, s1, a - 1, link)] = v;
    }
    transitions_found++;
  }
  assert(("Missing profiles in .transitions file", profiles_found == n_environments));
  infile.close();

  //Normalization
  double nrm;
  int env_loop = (is_mdp ? 1 : n_environments);
  for (int p = 0; p < env_loop; p++) {
    for (s1 = 0; s1 < n_observations; s1++) {
      for (a = 0; a < n_actions; a++) {
	nrm = 0.0;
	// If asking for precision, use kahan summation [slightly slower]
	if (precision) {
	  double kahan_correction = 0.0;
	  for (s2 = 0; s2 < n_actions; s2++) {
	    double val = transition_matrix[index(p, s1, a, s2)] - kahan_correction;
	    double aux = nrm + val;
	    kahan_correction = (aux - nrm) - val;
	    nrm = aux;
	  }
	}
	// Else basic sum
	else{
	  nrm = std::accumulate(&transition_matrix[index(p, s1, a, 0)],
				&transition_matrix[index(p, s1, a, n_actions)], 0.);
	}
	// Normalize
	std::transform(&transition_matrix[index(p, s1, a, 0)],
		       &transition_matrix[index(p, s1, a, n_actions)],
		       &transition_matrix[index(p, s1, a, 0)],
		       [nrm](const double t){ return t / nrm; }
		       );
      }
    }
  }
}

/**
 * GET_TRANSITION_PROBABILITY
 */
double Recomodel::getTransitionProbability(size_t s1, size_t a, size_t s2) const {
  size_t link = is_connected(s1, s2);
  if (link >= n_actions) {
    return 0.;
  } else {
    return (is_mdp ? transition_matrix[index(0, s1, a, link)] : transition_matrix[index(get_env(s1), get_rep(s1), a, link)]);
  }
}

/**
 * GET_EXPECTED_REWARD
 */
double Recomodel::getExpectedReward(size_t s1, size_t a, size_t s2) const {
  size_t link = is_connected(s1, s2);
  return ((link == a) ? rewards[a] : 0.);
}

/**
 * SAMPLESR
 */
std::tuple<size_t, double> Recomodel::sampleSR(size_t s,size_t a) const {
  // Sample next state according to transition function
  std::discrete_distribution<int> distribution (&transition_matrix[index(get_env(s), get_rep(s), a, 0)], &transition_matrix[index(get_env(s), get_rep(s), a, n_actions)]);
  size_t s2_link = distribution(generator);
  // Return sampled state and rewards
  size_t s2 = get_env(s) * n_observations + next_state(get_rep(s), s2_link);
  return std::make_tuple(s2, ((s2_link == a) ? rewards[a] : 0));
}

/**
 * ISTERMINAL
 */
bool Recomodel::isTerminal(size_t) const {
  return false;
}

/**
 * ISINITIAL
 */
bool Recomodel::isInitial(size_t s) const {
  return get_rep(s) == 0;
}

/**
 * PREVIOUS_STATES
 */
std::vector<size_t> Recomodel::previous_states(size_t state) const {
  size_t obs = get_rep(state), env = get_env(state);
  div_t aux = div(obs, n_actions);
  int prefix_s2 = ((aux.rem == 0) ? aux.quot - 1 : aux.quot);
  // If starting states
  if (obs == 0) {
    std::vector<size_t> prev;
    return prev;
  }
  // If contains empty selections
  if (prefix_s2 < acpows[1]) {
    std::vector<size_t> prev(1);
    prev.at(0) = env * n_observations + prefix_s2;
    return prev;
  }
  // Else, consider all possible incoming actions
  else {
    std::vector<size_t> prev(n_actions + 1);
    for (size_t a = 0; a <= n_actions; a++) {
      prev.at(a) = env * n_observations + (prefix_s2 + a * pows[0]);
    }
    return prev;
  }
}

/**
 * REACHABLE_STATES
 */
std::vector<size_t> Recomodel::reachable_states(size_t state) const {
  std::vector<size_t> aux (n_actions);
  for (int a = 0; a < n_actions; a++) {
    aux.at(a) = next_state(state, a);
  }
  return aux;
}



