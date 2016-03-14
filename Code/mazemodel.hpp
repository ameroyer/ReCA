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



class Mazemodel: public Model {

private:
  int min_x;   /*!< Minimum line index reachable by the robot */
  int max_x;   /*!< Maximum line index reachable by the robot */
  int min_y;   /*!< Minimum column index reachable by the robot */
  int max_y;   /*!< Minimum line index reachable by the robot */
  double* transition_matrix;        /*!< Transition matrix */
  std::vector <size_t> goal_states;  /*!< Indices of the goal states */
  std::vector <size_t> trap_states;
  std::vector <size_t> initial_states;
  std::vector <double> goal_rewards;
  static std::default_random_engine generator;

  
  /*! \brief Given an environment e, state s1, action a and state s2 (suffix),
   * returns the corresponding index in an 1D array.
   */
  int index(size_t env, size_t s, size_t a, size_t link) const;


  /*! \brief Returns the index of the state corresponding to a given position and orientation.
   *
   * \param x line index of the state.
   * \param y column index of the state.
   * \param orientation North(0), East(1), South(2) or West(3).
   *
   * \return the unique index representing the given state in the model.
   */
  size_t state_to_id(int x, int y, int orientation) const;


  /*! \brief Returns the index of the state corresponding to a given position and orientation.
   *
   * \param id state index.
   *
   * \return (x, y, orientation) the position and orientation of the corresponding state
   * Note: Special cases are (S/T/G, -1, -1) for the start, trap and goal states.
   */
  std::tuple<int, int, int> id_to_state(size_t state) const;



public:
  /*! \brief Initialize a MEMDP model from a given recommendation dataset.
   */
  Mazemodel(std::string sfile, double discount_);


  /*! \brief Destructor
   */
  ~Mazemodel();


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
  virtual std::tuple<size_t, double> sampleSR(size_t s,size_t a) const;


  /*! \brief Sample a state and reward given an origin state and chosen acion.
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated reward R(s, a, s2).
   */
  virtual std::tuple<size_t, size_t, double> sampleSOR(size_t s, size_t a) const;


  /*! \brief Rwturns whether a state is terminal or not.
   *
   * \param s state
   *
   * \return whether the state s is terminal or not.
   */
  bool isTerminal(size_t s) const;


  /*! \brief Rwturns whether a state is initial or not.
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


  /*! \brief Given a state and choice (e.g. item, direction) , return the next user state.
   *
   * \param state unique state index.
   * \param direction Left(0), Right(1), Forward(2).
   *
   * \return next_state index of the state corresponding to the agent applying ``direction`` in ``state``.
   */
  size_t next_state(size_t state, size_t direction) const;

  /*! \brief Given two states s1 and s2, return the action a such that s2 = s1.a if it exists,
   * or the value ``n_actions`` otherwise.
   *
   * \param s1 unique state index.
   * \param s2 unique state index
   *
   * \return link a valid action index [0 to n_actions - 1] if s1 and s2 can be connected, n_actions otherwise.
   */
  size_t is_connected(size_t s1, size_t s2) const;

  std::vector<size_t> reachable_states(size_t state) const;

};

#endif
