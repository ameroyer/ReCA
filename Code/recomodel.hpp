#ifndef RECOMODEL_H_INCLUDED
#define RECOMODEL_H_INCLUDED

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



class Recomodel: public Model {

private:
  double* transition_matrix; /*!< Transition matrix */
  double* rewards;           /*!< Rewards matrix */
  int hlength;               /*!< History length */
  int* pows;                 /*!< Precomputed exponents for conversion to base n_items */
  int* acpows;               /*!< Cumulative exponents for conversion from base n_items */
  static std::default_random_engine generator;


  /*! \brie fGiven an environment e, state s1, action a and state s2 (suffix),
   * returns the corresponding index in an 1D array.
   */
  int index(size_t env, size_t s, size_t a, size_t link) const;


  /*! \brief Returns the index of the state corresponding to a given sequence of item selections.
   * Note 1: Items correspond to actions with a +1 index shift, in order to allow the empty
   * selection to be represented by index 0.
   * Note 2: Items are ordered in decreasing age; the first time is the oldest in the history.
   * Mostly used for debugging purposes.
   *
   * \param state a state, represented by a sequence of selected items.
   *
   * \return the unique index representing the given state in the model.
   */
  size_t state_to_id(std::vector<size_t> state) const;


  /*! \brief Returns the sequence of items selection corresponding to the given state index.
   * Mostly used for debugging purposes.
   *
   * \param id unique state index.
   *
   * \return state a state, represented by a sequence of selected items.
   */
  std::vector<size_t> id_to_state(size_t id) const;


public:
  /*! \brief Initialize a MEMDP model from a given recommendation dataset.
   */
  Recomodel(std::string sfile, double discount_, bool is_mdp_);


  /*! \brief Destructor
   */
  ~Recomodel();


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
  void load_transitions(std::string tfile, bool precision=false, std::string pfile="");


  /*! \brief Returns a given transition probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return P( s2 | s1 -a-> ).
   */
  double getTransitionProbability(size_t s1, size_t a, size_t s2) const ;


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
   * \param item user choice [0 to n_actions - 1].
   *
   * \return next_state index of the state corresponding to the user choosing ``choice`` in ``state``.
   */
  size_t next_state(size_t state, size_t item) const;

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
