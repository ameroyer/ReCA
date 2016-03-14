#ifndef RECOMODEL_H_INCLUDED
/* ---------------------------------------------------------------------------
** recomodel.hpp
** Represent a MEMDP model constructed from data for a recommender system task
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include "model.hpp"
#include <iostream>
#include <tuple>
#include <random>
#include <string>
#include <ctime>


/**
 * INDEX
 */
int index(size_t env, size_t s, size_t a, size_t link) const {
  // TODO
  return 0;
}


/**
 * STATE_TO_ID
 */
size_t Mazemodel::state_to_id(int x, int y, int orientation) const {
  return 3 + (y - min_y) + (max_y - min_y) * ((x - min_x) + (max_x - min_x) * orientation)
}


/**
 * ID_TO_STATE
 */
size_t Mazemodel::id_to_state(size_t state) const {
  if (state == 0) {
    return std::tuple(0, -1, -1);
  } else if (state == 1) {
    return std::tuple(1, -1, -1);
  } else if (state == 2) {
    return std::tuple(2, -1, -1);
  } else {
    int y = (state - 3) % (max_y - min_y);
    int x = ((state - 3 - y) / (max_y - min_y)) % (max_x - min_x);
    int orientation = ((state - 3 - y) / (max_y - min_y)) / (max_x - min_x);
    return std::tuple(x + min_x, y + min_y, orientation);
  }


/**
 * CONSTRUCTOR
 */
Mazemodel::Mazemodel(double discount_) {
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
  has_mdp = false;
  discount = discount_;
  n_actions = 3;
  n_observations = 3 + (max_x - min_x) * (max_y - min_y) * 4;
  n_states = n_environments * n_actions;
  if (is_mdp) {
    transition_matrix = new double[(n_observations - 3) * n_actions * n_actions]();
  } else {
    transition_matrix = new double[n_environments * (n_observations - 3) * n_actions * n_actions]();
  }

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


/**
 * NEXT_STATE
 */
 size_t Mazemodel::next_state(size_t state, size_t direction) {
   size_t x, y, orientation;
   std::tie(x, y, orientation) = id_to_state(state);
   if (x < 0 && y < 0) {
     return state;
   }
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
 }
