/* ---------------------------------------------------------------------------
** mazemodel.cpp
** see mazemodel.cpp
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include "mazemodel.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <ctime>

/**
 * RANDOM ENGINE
 */
std::default_random_engine Mazemodel::generator(time(NULL));

/**
 * INDEX
 */
// Ignore S->, ->G and T->T transitions
int Mazemodel::index(size_t env, size_t s, size_t a, size_t link) const {
  return link + n_links * (a + n_actions * (s - 3 + (n_observations - 3) * env));
}

/**
 * STATE_TO_ID
 */
size_t Mazemodel::state_to_id(int x, int y, int orientation) const {
  return 3 + (y - min_y) + (max_y - min_y + 1) * ((x - min_x) + (max_x - min_x + 1) * orientation);
}

/**
 * ID_TO_STATE
 */
std::tuple<int, int, int> Mazemodel::id_to_state(size_t state) const {
  // Special observations (0, 1, 2)
  if (state == 0) {
    return std::make_tuple(0, -1, -1);
  } else if (state == 1) {
    return std::make_tuple(1, -1, -1);
  } else if (state == 2) {
    return std::make_tuple(2, -1, -1);
  }
  // Others
  else {
    int y = (state - 3) % (max_y - min_y + 1);
    int x = ((state - 3 - y) / (max_y - min_y + 1)) % (max_x - min_x + 1);
    int orientation = ((state - 3) / (max_y - min_y + 1)) / (max_x - min_x + 1);
    return std::make_tuple(x + min_x, y + min_y, orientation);
  }
}

/**
 * ISGOAL
 */
bool Mazemodel::isGoal(size_t state) const{
  int env = get_env(state);
  return std::find(goal_states.at(env).begin(), goal_states.at(env).end(), state) != goal_states.at(env).end();
}

/**
 * ISSTARTING
 */
bool Mazemodel::isStarting(size_t state) const {
  int env = get_env(state);
  return std::find(starting_states.at(env).begin(), starting_states.at(env).end(), state) != starting_states.at(env).end();
}

/**
 * MAYTRAP
 */
bool Mazemodel::MayTrap(size_t state) const {
  int env = get_env(state);
  for (int a = 0; a < n_actions; a++) {
    if (transition_matrix[index(env, get_rep(state), a, n_links - 1)] > 0) {
      return true;
    }
  }
  return false;
}

/**
 * IS_CONNECTED
 */
size_t Mazemodel::is_connected(size_t s1, size_t s2) const {
  // Environment check
  int env = get_env(s1);
  if (env != get_env(s2)) {
    return n_links;
  }
  // No Move
  else if (s1 == s2) {
    return n_links - 2;
  }
  // Trap
  else if (get_rep(s2) == 2 && MayTrap(s1)) {
    return n_links - 1;
  }
  // -> S
  else if (get_rep(s2) == 0) {
    return n_links;
  }
  // S ->
  if (get_rep(s1) == 0) {
    return (isStarting(s2) ? 2 : n_links);
  }
  // Absorbing states
  if (get_rep(s1) == 1 || get_rep(s1) == 2) {
    return ((s1 == s2) ? n_links - 2 : n_links);
  }
  // Final state
  if (isGoal(s1)) {
    return (get_rep(s2) == 1) ? 2 : n_links;
  }
  // Others
  int x1, y1, o1, x2, y2, o2;
  std::tie(x1, y1, o1) = id_to_state(get_rep(s1));
  std::tie(x2, y2, o2) = id_to_state(get_rep(s2));
  // If change in orentation (left or right)
  if (x1 == x2 && y1 == y2) {
    // Left
    if (o2 == (o1 + 3) % 4) {
      return 0;
    }
    // Right
    else if ( o2 == (o1 + 1) % 4) {
      return 1;
    }
    // No connection
    else {
      return n_actions + 1;
    }
  }
  // If change in position (forward)
  else if (o1 == o2) {
    // North
    if (o1 == 0 && x2 == x1 - 1 && y1 == y2) {
      return 2;
    }
    // East
    else if (o1 == 1 && y2 == y1 + 1 && x1 == x2) {
      return 2;
    }
    // South
    else if (o1 == 2 && x2 == x1 + 1 && y1 == y2) {
      return 2;
    }
    // West
    else if (o1 == 3 && y2 == y1 - 1 && x1 == x2) {
      return 2;
    }
    // No connections
    else {
      return n_links;
    }
  } else {
    return n_links;
  }
}

/**
 * NEXT_STATE
 */
size_t Mazemodel::next_state(size_t s, size_t link) const {
  // No Move
  if (link == n_links - 2) {
    return s;
  }
  // Trap
  else if (link == n_links - 1) {
    return get_env(s) * n_observations + 2;
  }
  // Directions
  else {
    size_t state = get_rep(s);
    size_t x, y, orientation;
    std::tie(x, y, orientation) = id_to_state(state);
    // Absorbing state
    if (y < 0 && orientation < 0) {
      return s;
    }
    // Final state go to G no matter the action
    if (isGoal(s)) {
      return get_env(s) * n_observations + 1;
    }
    // Else
    if (link == 0) {
      orientation = (orientation + 1) % 4;
    } else if (link == 1) {
      orientation = (orientation + 3) % 4;
    } else if (link == 2) {
      if (orientation == 0) {
	x = ((x > min_x) ? x - 1 : min_x);
      } else if (orientation == 1) {
	y = ((y < max_y) ? y + 1 : max_y);
      } else if (orientation == 2) {
	x = ((x < max_x) ? x + 1 : max_x);
      } else if (orientation == 3) {
	y = ((y > min_y) ? y - 1 : min_y);
      }
    }
    return get_env(s) * n_observations + state_to_id(x, y, orientation);
  }
}

/**
 * CONSTRUCTOR
 */
Mazemodel::Mazemodel(std::string sfile, double discount_) {
  //********** Load summary information
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  size_t aux;
  infile.open(sfile, std::ios::in);
  assert((".summary file not found", infile.is_open()));
  // min x
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  min_x = aux;
  // max x
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  max_x = aux;
  assert(("Invalid x boundaries", min_x <= max_x));
  // min y
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  min_y = aux;
  // max y
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  max_y = aux;
  assert(("Invalid y boundaries", min_y <= max_y));
  // number of environments
  std::getline(infile, line);
  iss.str(line);
  iss >> aux;
  n_environments = aux;

  //********** Initialize
  is_mdp = false;
  discount = discount_;
  n_actions = 3;  // Left, Right, Forward
  n_links = 5;    // Left, Right, Forward, No Move, s->T
  n_observations = 3 + (max_x - min_x + 1) * (max_y - min_y + 1) * 4;
  n_states = n_environments * n_observations;
  transition_matrix = new double[n_environments * (n_observations - 2) * n_actions * n_links]();


  //********** Summary of model parameters
  std::cout << "   -> The model contains " << n_observations << " observations\n";
  std::cout << "   -> The model contains " << n_actions << " actions\n";
  std::cout << "   -> The model contains " << n_states << " states\n";
  std::cout << "   -> The model contains " << n_environments << " environments\n";
}

/**
 * DESTRUCTOR
 */
Mazemodel::~Mazemodel() {
  //delete []transition_matrix;
}

/**
 * STRING_TO_ACTION
 */
int string_to_action(std::string s) {
  if (!s.compare("L")) {
    return 0;
  } else if (!s.compare("R")) {
    return 1;
  } else if (!s.compare("F")) {
    return 2;
  } else {
    return -1;
  }
}

/**
 * STRING_TO_ORIENTATION
 */
int string_to_orientation(char c) {
  std::string s(1, c);
  if (!s.compare("N")) {
    return 0;
  } else if (!s.compare("E")) {
    return 1;
  } else if (!s.compare("S")) {
    return 2;
  } else if (!s.compare("W")) {
    return 3;
  } else {
    return -1;
  }
}


/**
 * LOAD_REWARDS
 */
void Mazemodel::load_rewards(std::string rfile) {
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  std::string s1, a, s2;
  int x, y, env = 0;
  char o;
  double v;

  infile.open(rfile, std::ios::in);
  assert((".rewards file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> s1 >> a >> s2 >> v)) {
      env++;
      continue;
    }
    assert(("Unvalid reward entry", !s2.compare("G")));
    sscanf(s1.c_str(), "%dx%dx%c", &x, &y, &o);
    // Initialize env
    size_t sg = env * n_observations + state_to_id(x, y, string_to_orientation(o));
    if (goal_states.size() <= env) {
      std::vector <size_t> aux;
      goal_states.push_back(aux);
    }
    // Initialize state
    if (!isGoal(sg)) {
      goal_states.at(env).push_back(sg);
      std::vector <double> aux2 (n_actions, 0);
      goal_rewards[sg] = aux2;
    }
    goal_rewards.at(sg).at(string_to_action(a)) = v;
  }
  infile.close();
}

/**
 * LOAD_TRANSITIONS
 */
void Mazemodel::load_transitions(std::string tfile, bool precision /* =false */) {
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  std::string s1, a, s2;
  int x, y, env = 0;
  char o;
  double v;

  // Load transitions
  infile.open(tfile, std::ios::in);
  assert((".transitions file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);

    // Change profile
    if (!(iss >> s1 >> a >> s2 >> v)) {
      env++;
      assert(("Too many profiles found in .transitions file",
	      env <= n_environments));
      continue;
    }

    // Ignore T and G states, already taken into account
    if (!s1.compare("T") || !s1.compare("G") || !s2.compare("G")) {
      continue;
    }

    // Find starting states for current environment
    if (!s1.compare("S")) {
      sscanf(s2.c_str(), "%dx%dx%c", &x, &y, &o);
      size_t s = env * n_observations + state_to_id(x, y, string_to_orientation(o));
      if (starting_states.size() <= env) {
	std::vector<size_t> aux;
	starting_states.push_back(aux);
      }
      if (! isStarting(s)) {
	starting_states.at(env).push_back(s);
      }
      continue;
    }

    // Parse input states
    sscanf(s1.c_str(), "%dx%dx%c", &x, &y, &o);
    size_t state1 = env * n_observations + state_to_id(x, y, string_to_orientation(o));
    size_t state2 = env * n_observations + 2;
    if (s2.compare("T")) {
      sscanf(s2.c_str(), "%dx%dx%c", &x, &y, &o);
      state2 = env * n_observations + state_to_id(x, y, string_to_orientation(o));
    }
    // Add transition
    size_t action = string_to_action(a);
    size_t link = is_connected(state1, state2);
    assert(("Unfeasible transition with >0 probability", link < n_links || !s2.compare("T")));
    transition_matrix[index(env, get_rep(state1), action, link)] = v;
  }
  assert(("Missing profiles in .transitions file", env == n_environments));
  infile.close();
  // Normalization
  double nrm;
  for (int p = 0; p < n_environments; p++) {
    for (size_t state1 = 2; state1 < n_observations; state1++) {
      for (size_t action = 0; action < n_actions; action++) {
	nrm = 0.0;
	// If asking for precision, use kahan summation [slightly slower]
	if (precision) {
	  double kahan_correction = 0.0;
	  for (size_t state2 = 0; state2 < n_links; state2++) {
	    double val = transition_matrix[index(p, state1, action, state2)] - kahan_correction;
	    double aux = nrm + val;
	    kahan_correction = (aux - nrm) - val;
	    nrm = aux;
	  }
	}
	// Else basic sum
	else{
	  nrm = std::accumulate(&transition_matrix[index(p, state1, action, 0)],
				&transition_matrix[index(p, state1, action, n_links)], 0.);
	}
	// Normalize
	std::transform(&transition_matrix[index(p, state1, action, 0)],
		       &transition_matrix[index(p, state1, action, n_links)],
		       &transition_matrix[index(p, state1, action, 0)],
		       [nrm](const double t){ return t / nrm; }
		       );
      }
    }
  }
}


/**
 * GET_TRANSITION_PROBABILITY
 */
double Mazemodel::getTransitionProbability(size_t s1, size_t a, size_t s2) const {
  // Start state
  if (get_rep(s2) == 0) {
    return 0.;
  }
  else if (get_rep(s1) == 0) {
    int env = get_env(s1);
    if (env != get_env(s2) || !isStarting(s2)) {
      return 0.;
    } else {
      return 1.0 / starting_states.at(env).size();
    }
  }
  // Absorbing transitions
  else if (get_rep(s1) == 1 || get_rep(s1) == 2) {
    return ((s1 == s2) ? 1.0 : 0.0);
  }
  // Goal state to G
  else if (isGoal(s1)) {
    return ((get_env(s1) == get_env(s2)) && get_rep(s2) == 1) ? 1. : 0.;
  }
  // Others
  else {
    size_t link = is_connected(s1, s2);
    if (link >= n_links) {
      return 0.;
    } else {
      return transition_matrix[index(get_env(s1), get_rep(s1), a, link)];
    }
  }
}

/**
 * GET_EXPECTED_REWARD
 */
double Mazemodel::getExpectedReward(size_t s1, size_t a, size_t s2) const {
  if (!(get_rep(s2) == 1 && isGoal(s1))) {
    return 0.;
  } else {
    return goal_rewards.at(s1).at(a);
  }
}

/**
 * SAMPLESR
 */
std::tuple<size_t, double> Mazemodel::sampleSR(size_t s, size_t a) const {
  // Start state
  if (get_rep(s) == 0) {
    int env = get_env(s);
    size_t s2 = env * n_observations + starting_states.at(env).at(rand() % starting_states.at(env).size());
    return std::make_tuple(s2, 0);
  }
  // Absorbing state
  else if (get_rep(s) == 1 || get_rep(s) == 2) {
    return std::make_tuple(s, 0.);
  }
  // Goal state, get reward
  else if (isGoal(s)) {
    return std::make_tuple(get_env(s) * n_observations + 1, goal_rewards.at(s).at(a));
  }
  // Others
  else {
    // Sample random transition
    std::discrete_distribution<int> distribution (&transition_matrix[index(get_env(s), get_rep(s), a, 0)], &transition_matrix[index(get_env(s), get_rep(s), a, n_links)]);
    size_t link = distribution(generator);
    // Return values
    size_t s2 = next_state(s, link);
    return std::make_tuple(s2, 0.);
  }
}

/**
 * ISTERMINAL
 */
bool Mazemodel::isTerminal(size_t s) const {
  return (get_rep(s) == 1) || (get_rep(s) == 2);
}

/**
 * ISINITIAL
 */
bool Mazemodel::isInitial(size_t s) const {
  return get_rep(s) == 0;
}

/**
 * PREVIOUS_STATES
 */
std::vector<size_t> Mazemodel::previous_states(size_t state) const {
  // Initial state
  if (get_rep(state) == 0) {
    std::vector<size_t> aux;
    return aux;
  } // Goal state
  else if (get_rep(state) == 1) {
    std::vector<size_t> aux = goal_states.at(get_env(state));
    aux.push_back(state);
    return aux;
  } // Trap state
  else if (get_rep(state) == 2) {
    int env = get_env(state);
    std::vector<size_t> aux;
    aux.push_back(state);
    for (size_t s = 3; s < n_observations; s++) {
      if (MayTrap(env * n_observations + s)) {
	aux.push_back(env * n_observations + s);
      }
    }
  }
  // others
  else {
    int x, y, o;
    std::tie(x, y, o) = id_to_state(state);
    std::vector<size_t> aux;
    // Left;
    aux.push_back(state_to_id(x, y, (o + 1) % 4));
    // Right;
    aux.push_back(state_to_id(x, y, (o + 3) % 4));
    // Forward, if possible;
    if (o == 0 && x < max_x) {
      aux.push_back(state_to_id(x + 1, y, o));
    } else if (o == 1 && y > min_y) {
      aux.push_back(state_to_id(x, y - 1, o));
    } else if (o == 2 && x > min_x) {
      aux.push_back(state_to_id(x - 1, y, o));
    } else if (o == 3 && y < max_y) {
      aux.push_back(state_to_id(x, y + 1, o));
    }
    // No Move
    aux.push_back(state);
    // Starting state if applicable
    if (isStarting(state)) {
      aux.push_back(get_env(state) * n_observations + 0);
    }
  }
}

/**
 * REACHABLE_STATES
 */
std::vector<size_t> Mazemodel::reachable_states(size_t state) const {
  // Start state
  if (get_rep(state) == 0) {
    return starting_states.at(get_env(state));
  } //Absorbing states
  else if (get_rep(state) == 1 || get_rep(state == 2)) {
    std::vector<size_t> result(1);
    result.at(0) = state;
    return result;
  } // Final state
  else if (isGoal(state)) {
    std::vector<size_t> result(1);
    result.at(0) = get_env(state) * n_observations + 1;
    return result;
  } // others
  else {
    std::vector<size_t> aux (n_actions);
    for (int a = 0; a < n_actions; a++) {
      aux.at(a) = next_state(state, a);
    }
    aux.push_back(state);
    if (MayTrap(state)) {
      aux.push_back(get_env(state) * n_observations + 2);
    }
    return aux;
  }
}

