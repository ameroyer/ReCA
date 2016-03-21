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
int Mazemodel::index(size_t env, size_t s, size_t a, size_t link) const {
  // TODO
  return link + (n_actions + 1) * (a + n_actions * (s - 2 + (n_observations - 2) * env));
}


/**
 * STATE_TO_ID
 */
// 0 = S
// 1 = G
// 2 = T
size_t Mazemodel::state_to_id(int x, int y, int orientation) const {
  //std::cout << y << " " << min_y << " " << max_y  << " " <<x << " " << min_x << " " << max_x << " " << orientation << "\n"; 
 return 3 + (y - min_y) + (max_y - min_y + 1) * ((x - min_x) + (max_x - min_x + 1) * orientation);
}


/**
 * ID_TO_STATE
 */
std::tuple<int, int, int> Mazemodel::id_to_state(size_t state) const {
  if (state == 0) {
    return std::make_tuple(0, -1, -1);
  } else if (state == 1) {
    return std::make_tuple(1, -1, -1);
  } else if (state == 2) {
    return std::make_tuple(2, -1, -1);
  } else {
    int y = (state - 3) % (max_y - min_y + 1);
    int x = ((state - 3 - y) / (max_y - min_y + 1)) % (max_x - min_x + 1);
    int orientation = ((state - 3) / (max_y - min_y + 1)) / (max_x - min_x + 1);
    return std::make_tuple(x + min_x, y + min_y, orientation);
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
  n_actions = 3;
  n_observations = 3 + (max_x - min_x + 1) * (max_y - min_y + 1) * 4;
  n_states = n_environments * n_observations;
  transition_matrix = new double[n_environments * (n_observations - 2) * n_actions * (n_actions + 1)]();
  

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
  delete []transition_matrix;
}


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

int string_to_orientation(std::string s) {
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
//TODO: assumption one goal state per environment and unique reward
void Mazemodel::load_rewards(std::string rfile) {
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  std::string s1, a, s2;
  int x, y;
  char o[1];
  int i = 0;
  double v;
  
  infile.open(rfile, std::ios::in);
  assert((".rewards file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> s1 >> a >> s2 >> v)) {
      i++;
      break;
    }
    assert(("Unvalid reward entry", !s2.compare("G")));
    sscanf(s1.c_str(), "%dx%dx%s", &x, &y, &o);
    if (goal_states.size() <= i) {
      size_t sg = state_to_id(x, y, 0);
      goal_states.push_back(i * n_observations + sg);
      goal_rewards.push_back(v);
    } else {
      size_t sg = i * n_observations + state_to_id(x, y, string_to_orientation(o));
      assert(("not unique goal state", isGoal(sg)));
      assert(("not unique goal state reward", v == goal_rewards.at(i)));
    }
  }
  std::cout << "DOne";
  infile.close();
}

/**
 * LOAD_TRANSITIONS
 */
void Mazemodel::load_transitions(std::string tfile, bool precision /* =false */) {
  //TODO
  std::ifstream infile;
  std::string line;
  std::istringstream iss;
  double v;
  std::string s1, a, s2;
  int x, y;
  char o[1];
  int transitions_found = 0, profiles_found = 0;
  
  // Load transitions
  infile.open(tfile, std::ios::in);
  assert((".transitions file not found", infile.is_open()));
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    // Change profile
    if (!(iss >> s1 >> a >> s2 >> v)) {
      profiles_found += 1;
      //assert(("Incomplete transition function in current profile in .transitions",
      // transitions_found == n_observations * n_actions * n_actions));
      assert(("Too many profiles found in .transitions file",
	      profiles_found <= n_environments));
      transitions_found = 0;
      continue;
    }
    // Ignore T and G states, already taken into account
    if (!s1.compare("T") || !s1.compare("G") || !s2.compare("G")) {
      continue;
    }
    // Find initial states
    if (!s1.compare("S")) {
      sscanf(s2.c_str(), "%dx%dx%s", &x, &y, &o);
      size_t s = state_to_id(x, y, string_to_orientation(o));
      if (std::find(initial_states.begin(), initial_states.end(), get_rep(s)) != initial_states.end()) {
	initial_states.push_back(s);
      }
      continue;
    }
  
    // Set transitions for other combination
    //Parse
    //std::string aux1, aux2;
    //sscanf(s1.c_str(), "%dx%dx%s", &x, &y, &o);
    std::stringstream ss(s1);
    //ss >> x >> aux1 >> y >> aux2 >> o;
    int truc;
    ss >> truc;
    if (ss.peek() == 'x') {ss.ignore();}
    ss >> y;
    if (ss.peek() == 'x') { ss.ignore();}
    ss >> o;
//std::cout << "a" << s1.c_str() << "b " << truc << " " << x << " " << y << " " << o << string_to_orientation(o) << "\n";
    size_t state1 = state_to_id(truc, y, string_to_orientation(o));
    size_t state2 = 2; // Trap
    if (s2.compare("T")) {
      int truc2;
      sscanf(s2.c_str(), "%dx%dx%s", &truc2, &y, &o);
      //std::cout << s2 << " " << truc2 << " " << y << " " << o <<string_to_orientation(o) << "\n";
      state2 = state_to_id(truc2, y, string_to_orientation(o));
    }
    size_t action = string_to_action(a);
    // Assign
    size_t link = is_connected(state1, state2); // chqnge orientqtion left, right, go forward or do not move.
// std::cout << s1 << " " << s2 << " " << link << " " << state1 << " " << state2 << "\n";
    assert(("Unfeasible transition with >0 probability", link < n_actions + 1 || !s2.compare("T")));
    // std::cout << "Assign " << profiles_found << " " << state1 << action << " " << link << "\n";
    // TODO link trap
    transition_matrix[index(profiles_found, state1, action, link)] = v;  
    transitions_found++;
  }
  assert(("Missing profiles in .transitions file", profiles_found == n_environments));
  infile.close();
  //std::cout << "start normalization\n";
  //Normalization
  double nrm;
  for (int p = 0; p < n_environments; p++) {
    for (size_t state1 = 2; state1 < n_observations; state1++) {
      for (size_t action = 0; action < n_actions; action++) {
	nrm = 0.0;
	// If asking for precision, use kahan summation [slightly slower]
	if (precision) {
	  double kahan_correction = 0.0;
	  for (size_t state2 = 0; state2 < n_actions; state2++) {
	    double val = transition_matrix[index(p, state1, action, state2)] - kahan_correction;
	    double aux = nrm + val;
	    kahan_correction = (aux - nrm) - val;
	    nrm = aux;
	  }
	}
	// Else basic sum
	else{
	  nrm = std::accumulate(&transition_matrix[index(p, state1, action, 0)],
				&transition_matrix[index(p, state1, action, n_actions + 1)], 0.);
	}
	// Normalize
	std::transform(&transition_matrix[index(p, state1, action, 0)],
		       &transition_matrix[index(p, state1, action, n_actions + 1)],
		       &transition_matrix[index(p, state1, action, 0)],
		       [nrm](const double t){ return t / nrm; }
		       );
      }
    }
  }
  std::cout << "Done2";
}


/**
 * GET_TRANSITION_PROBABILITY
 */
double Mazemodel::getTransitionProbability(size_t s1, size_t a, size_t s2) const {
  // Start state
  if (get_rep(s2) == 0) {
    return 0.;
  }
  if (get_rep(s1) == 0) {
    if (get_env(s1) != get_env(s2) || std::find(initial_states.begin(), initial_states.end(), get_rep(s2)) == initial_states.end()) {
      return 0.;
    } else {
      return 1.0 / initial_states.size();
    }
  }
  // Absorbing states
  else if (get_rep(s1) == 1 || get_rep(s1) == 2) {
    return ((s1 == s2) ? 1.0 : 0.0);
  }
  // Final state to Goal state
  else if (isGoal(s1)) {
    return ((get_env(s1) == get_env(s2)) && get_rep(s2) == 2) ? 1. : 0.;
  }
  // Others
  else {
    size_t link = is_connected(s1, s2);
    if (link > n_actions) {
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
  size_t link = is_connected(s1, s2);
  size_t e = get_env(s1);
  if (link > n_actions || ! isGoal(s1)) {
    return 0.;
  } else {
    return goal_rewards.at(e);
  }
}


/**
 * SAMPLESR
 */
std::tuple<size_t, double> Mazemodel::sampleSR(size_t s, size_t a) const {
  // Start state
  if (get_rep(s) == 0) {
    size_t s2 = get_env(s) * n_observations + initial_states.at(rand() % initial_states.size());
    double r = ((s2 == goal_states.at(get_env(s))) ? goal_rewards.at(get_env(s)) : 0.);
    return std::make_tuple(s2, r);
  }
  // Absorbing state
  else if (get_rep(s) == 1 || get_rep(s) == 2) {
    return std::make_tuple(s, 0.);
  }
  // Final state
  else if (isGoal(s)) {
    return std::make_tuple(get_env(s) * n_observations + 1, goal_rewards.at(get_env(s)));
  }
  // Others
  else {
    // Sample random transition
    std::discrete_distribution<int> distribution (&transition_matrix[index(get_env(s), get_rep(s), a, 0)], &transition_matrix[index(get_env(s), get_rep(s), a, n_actions + 1)]);
    size_t link = distribution(generator);
    // Return values
    size_t s2 = next_state(s, link);
    if (s2 == goal_states.at(get_env(s))) {
      return std::make_tuple(s2, goal_rewards.at(get_env(s)));
    } else {
      return std::make_tuple(s2, 0.);
    }
  }
}


/**
 * SAMPLESOR
 */
std::tuple<size_t, size_t, double> Mazemodel::sampleSOR(size_t s, size_t a) const {
  size_t s2;
  double reward;
  std::tie(s2, reward) = sampleSR(s, a);
  return std::make_tuple(s2, get_rep(s2), reward);
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
 * ISGOAL
 */
bool Mazemodel::isGoal(size_t state) const{
  size_t x, y, orientation;
  std::tie(x, y, orientation) = id_to_state(state);
  // Representer = Goal state with orientation 0 (North)
  return goal_states.at(get_env(state)) == state_to_id(x, y, 0);
}


/**
 * NEXT_STATE
 */
size_t Mazemodel::next_state(size_t s, size_t direction) const {
  size_t state = get_rep(s);
  size_t x, y, orientation;
  std::tie(x, y, orientation) = id_to_state(state);
  if (direction == n_actions) {
    return s;
  }
  // Absorbing state
  if (y < 0 && orientation < 0) {
    return s;
  }
  // FInal state go to G no matter the action
  if (isGoal(s)) {
    return get_env(s) * n_observations + 1;
  }
  // Else
  if (direction == 0) {
    orientation = (orientation + 1) % 4;
  } else if (direction == 1) {
    orientation = (orientation + 3) % 4;
  } else if (direction == 2) {
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



/**
 * PREVIOUS_STATES
 */
std::vector<size_t> Mazemodel::previous_states(size_t state) const {
  /*
    size_t obs = get_rep(state), env = get_env(state);
    div_t aux = div(obs, n_actions);
    int prefix_s2 = ((aux.rem == 0) ? aux.quot - 1 : aux.quot);
    if (obs == 0) {
    std::vector<size_t> prev;
    return prev;
    }
    if (prefix_s2 < acpows[1]) {
    std::vector<size_t> prev(1);
    prev.at(0) = env * n_observations + prefix_s2;
    return prev;
    } else {
    std::vector<size_t> prev(n_actions + 1);
    for (size_t a = 0; a <= n_actions; a++) {
    prev.at(a) = env * n_observations + (prefix_s2 + a * pows[0]);
    }
    return prev;
    }*/
  // TODO
}


std::vector<size_t> Mazemodel::reachable_states(size_t state) const {
  // Start states
  if (get_rep(state) == 0) {
    std::vector<size_t> result(initial_states.size());
    for (int i = 0; i < initial_states.size(); i++) {
      result.at(i) = get_env(state) * n_observations + initial_states.at(i);
    }
    return result;
    //Absorbing states
  } else if (get_rep(state) == 1 || get_rep(state == 2)) {
    std::vector<size_t> result(1);
    result.at(0) = state;
    return result;
    // Final state
  } else if (isGoal(state)) {
    std::vector<size_t> result(1);
    result.at(0) = get_env(state) * n_observations + 1;
    return result;
  } else {
    std::vector<size_t> aux (n_actions);
    for (int a = 0; a < n_actions; a++) {
      aux.at(a) = next_state(state, a);
    }
    return aux;
  }
}


/**
 * IS_CONNECTED
 */
size_t Mazemodel::is_connected(size_t s1, size_t s2) const {

  if (s1 == s2) {
    return n_actions;
  }
  // Environment check
  if (get_env(s1) != get_env(s2)) {
    return n_actions + 1;
  }
  // State start
  if (get_rep(s2) == 0) {
    return n_actions + 1;
  }
  if (get_rep(s1) == 0) {
    return ((std::find(initial_states.begin(), initial_states.end(), get_rep(s2)) != initial_states.end()) ? 2 : n_actions + 1);
  }
  // Absorbing states
  if (get_rep(s1) == 1 || get_rep(s1) == 2) {
    return ((s1 == s2) ? true : false);
  }
  // Final state
  if (isGoal(s1)) {
    return (get_rep(s2) == 1) ? true : false;
  }
  // Others
  int x1, y1, o1, x2, y2, o2;
  std::tie(x1, y1, o1) = id_to_state(s1);
  std::tie(x2, y2, o2) = id_to_state(s2);
  //std::cout << x1 << y1 <<o1 << " " << x2 << y2 << o2 << "\n";
  // If change in orentqtion (left or right)
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
    if (o1 == 0 && x2 == x1 - 1) {
      return 2;
    }
    // East
    else if (o1 == 1 && y2 == y1 + 1) {
      return 2;
    }
    // South
    else if (o1 == 2 && x2 == x1 + 1) {
      return 2;
    }
    // West
    else if (o1 == 3 && y2 == y1 - 1) {
      return 2;
    }
    // No connections
    else {
      return n_actions + 1;
    }
  } else {
    return n_actions + 1;
  }
}
