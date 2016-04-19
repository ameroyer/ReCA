#ifndef MAZEMODEL_H_INCLUDED
#define MAZEMODEL_H_INCLUDED

/* ---------------------------------------------------------------------------
** mazemodel.hpp
** Represent a MEMDP model constructed from data for a robot navigation task.
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
#include <map>


class Mazemodel: public Model {

private:
  int min_x;   /*!< Minimum reachable row index */
  int max_x;   /*!< Maximum reachable row index */
  int min_y;   /*!< Minimum reachable column index */
  int max_y;   /*!< Maximum reachale column index */
  size_t n_links = 6;  /*< Link Targets: Left, right, forward, no change, G, T*/
  size_t nomove_link = n_links - 3;
  size_t goal_link = n_links - 2;
  size_t trap_link = n_links - 1;
  size_t S = 0; /*< Special observations */
  size_t G = 1;
  size_t T = 2;
  double* transition_matrix;         /*!< Transition matrix. Ignore S-> and absorbing transitions */
  std::vector<std::vector <size_t> > goal_states;  /*!< List of states leading to G for each environment */
  std::vector<std::vector <size_t> > starting_states;  /*!< List of states reachable from S for each environment */
  std::map<size_t, std::vector <double> > goal_rewards;  /*!< Associate a (goal state, input action) to the corresponding reward */
  static std::default_random_engine generator;

  /*! \brief Given an environment e, state s1, action a and state s2 (suffix),
   * returns the corresponding index in an 1D array.
   */
  int index(size_t env, size_t s1, size_t a, size_t s2_link) const;

  /*! \brief Returns the index of the observation corresponding to a given position and orientation.
   *
   * \param x line index of the state.
   * \param y column index of the state.
   * \param orientation North(0), East(1), South(2) or West(3).
   *
   * \return the unique index representing the given state in the model. In particular, 0 = S, 1 = G, 2 = T.
   */
  size_t state_to_id(int x, int y, int orientation) const;

  /*! \brief Returns the position and orientation corresponding to the index of a given observation.
   *
   * \param id state index.
   *
   * \return (x, y, orientation) the position and orientation of the corresponding state
   * Note: Special cases are (S/T/G, -1, -1) for the start, trap and goal states.
   */
  std::tuple<int, int, int> id_to_state(size_t state) const;

  /*! \brief Given two states s1 and s2, return the link L such that s2 = s1.L if it exists,
   * or the value ``n_links`` otherwise.
   *
   * \param s1 unique state index.
   * \param s2 unique state index
   *
   * \return link a valid link index if s1 and s2 can be connected, n_links otherwise. In practice Left (0), Right (1), Forward (2), No Move (3).
   */
  size_t is_connected(size_t s1, size_t s2) const;

  /*! \brief Given a state and chosen action, return the next logical user state (if no mistake).
   *
   * \param state unique state index.
   * \param direction Left(0), Right(1), Forward(2).
   *
   * \return next_state index of the state corresponding to the agent applying ``direction`` in ``state``.
   */
  size_t next_state(size_t state, size_t direction) const;

  /*! \brief Returns whether a state is a goal state in its environment (-> G).
   *
   * \param s state
   *
   * \return whether the state s is final or not.
   */
  bool isGoal(size_t s) const;

  /*! \brief Returns whether a state is a starting state in its environment (S ->).
   *
   * \param s state
   *
   * \return whether the state s is final or not.
   */
  bool isStarting(size_t s) const;

  /*! \brief Returns True iff the trap state T can be reached from the given state.
   *
   * \param s state
   *
   * \return whether the state s can reach T.
   */
  bool isTrap(size_t s) const;

  /*! \brief Returns True iff the given state can never be reached. Used for debugging and maze representation purposes.
   *
   * \param s state
   *
   * \return whether the state s is a wall or not.
   */
  bool isWall(size_t s) const;

  /*! \brief Prints the maze of each environment. For debugging purposes.
   *
   * \param s state
   *
   * \return whether the state s is a wall or not.
   */
  void print_maze()  const;


public:
  /*! \brief Initialize a MEMDP model from a given recommendation dataset.
   */
  Mazemodel(std::string sfile, double discount_);

  /*! \brief Destructor
   */
  ~Mazemodel();

  /*! \brief Returns a string representation of the given state.
   *
   * \param s state index.
   *
   * \return str string representation of s.
   */
  std::string state_to_string(size_t s) const;

  /*! \brief Load rewards of the model from file
   *
   * \param rfile Rewards file
   */
  void load_rewards(std::string rfile);

  /*! \brief Load transitions of the model from file
   *
   * \param tfile Transition file.
   * \param pfile Profiles distribution file.
   * \param precision If true, precise normalization is enabled.
   */
  void load_transitions(std::string tfile, bool precision=false);

  /*! \brief Returns a given transition probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return P( s2 | s1 -a-> ).
   */
  double getTransitionProbability(size_t s1, size_t a, size_t s2) const;

  /*! \brief Returns a given reward.
   *
   * \param s1 origin state.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return R(s1, a, s2).
   */
  double getExpectedReward(size_t s1, size_t a, size_t s2) const;

  /*! \brief Sample a state and reward given an origin state and chosen acion.
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated reward R(s, a, s2).
   */
  std::tuple<size_t, double> sampleSR(size_t s,size_t a) const;

  /*! \brief Rwturns whether a state is terminal or not.
   *
   * \param s state
   *
   * \return whether the state s is terminal or not.
   */
  bool isTerminal(size_t s) const;

  /*! \brief Returns whether a state is initial or not.
   *
   * \param s state
   *
   * \return whether the state s is initial or not.
   */
  bool isInitial(size_t s) const;

  /*! \brief Given a state, returns all its predecessors.
   *
   * \param state unique state index.
   *
   * \return next_state index of the state corresponding to the user choosing ``item`` in ``state``.
   */
  std::vector<size_t> previous_states(size_t state) const;

  /*! \brief Given a state, returns all its possible successors.
   *
   * \param state unique state index.
   *
   * \return reachable_states the state's possible successors.
   */
  std::vector<size_t> reachable_states(size_t state) const;
};

#endif
