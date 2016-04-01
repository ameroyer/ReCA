#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

/* ---------------------------------------------------------------------------
** model.hpp
** Main abstract class to represent a MEMDP model
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include <vector>
#include <iostream>
#include <tuple>

class Model {
public:
  /*! \brief Default constructor
   *
   */
  Model() {};

  /*! \brief Default destructor.
   */
  ~Model() {};

  /*! \brief Returns true iff the model should be interpreted as as MDP.
   *
   * \return true if the model is in MDP mode (see Recomodel for instance).
   */
  bool mdp_enabled() const { return is_mdp; };

  /*! \brief Returns the number of states in the model.
   * @AIToolBox Model interface
   *
   * \return the number of states in the model.
   */
  size_t getS() const { return n_states; };

  /*! \brief Returns the number of observations in the model.
   * @AIToolBox Model interface
   *
   * \return number of observations in the model.
   */
  size_t getO() const { return n_observations; };

  /*! \brief Returns the number of actionss in the model.
   * @AIToolBox Model interface
   *
   * \return number of actions in the model.
   */
  size_t getA() const { return n_actions; };

  /*! \brief Returns the number of environments in the model.
   *
   * \return number of environments in the model.
   */
  size_t getE() const { return n_environments; };

  /*! \brief Returns the discount factor in the model.
   * @AIToolBox Model interface
   *
   * \return Discount factor in the MDP.
   */
  double getDiscount() const { return discount; };

  /*! \brief Returns a given transition probability.
   * @AIToolBox Model interface
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return P( s2 | s1 -a-> ).
   */
  virtual double getTransitionProbability( size_t s1, size_t a, size_t s2 ) const = 0;

  /*! \brief Returns a given observation probability.
   * @AIToolBox Model interface
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param o observation.
   *
   * \return P( o | -a-> s1 ).
   */
  virtual double getObservationProbability(size_t s1, size_t a, size_t o) const {
    return ((get_rep(s1)) ? 1. : 0.);
  };

  /*! \brief Returns a given reward.
   * @AIToolBox Model interface
   *
   * \param s1 origin state.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return R(s1, a, s2).
   */
  virtual double getExpectedReward( size_t s1, size_t a, size_t s2 ) const = 0;

  /*! \brief Sample a state and reward given an origin state and chosen acion.
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated reward R(s, a, s2).
   */
  virtual std::tuple<size_t, double> sampleSR(size_t s,size_t a) const = 0;

  /*! \brief Sample a state, observation and reward given an origin state and chosen acion.
   * @AIToolBox Model interface
   *
   * \param s origin state.
   * \param a chosen action.
   *
   * \return s2 such that s -a-> s2, and the associated observation and reward R(s, a, s2).
   */
  virtual std::tuple<size_t, size_t, double> sampleSOR(size_t s, size_t a) const {
    size_t s2;
    double reward;
    std::tie(s2, reward) = sampleSR(s, a);
    return std::make_tuple(s2, get_rep(s2), reward);
  }

  /*! \brief Rwturns whether a state is terminal or not.
   * @AIToolBox Model interface
   *
   * \param s state
   *
   * \return whether the state s is terminal or not.
   */
  virtual bool isTerminal(size_t s) const = 0;


  /*! \brief Rwturns whether a state is initial or not.
   *
   * \param s state
   *
   * \return whether the state s is initial or not.
   */
  virtual bool isInitial(size_t s) const = 0;

  /*!
   * \brief Given a state of the MEMDP, returns the corresponding environment.
   *
   * \param s a state in the MEMDP.
   *
   * \return the environment to which s belongs.
   */
  size_t get_env(size_t s) const { return s / n_observations; };

  /*!
   * \brief Given a state of the MEMDP, returns its representative/observation.
   *
   * \param s a state in the MEMDP.
   *
   * \return the corresponding observation.
   */
  size_t get_rep(size_t s) const { return s % n_observations; };

  /*! \brief Given a state, returns all its possible predecessors.
   *
   * \param state unique state index.
   *
   * \return previous_states the state's possible predecessors.
   */
  virtual std::vector<size_t> previous_states(size_t state) const = 0;

  /*! \brief Given a state, returns all its possible successors.
   *
   * \param state unique state index.
   *
   * \return reachable_states the state's possible successors.
   */
  virtual std::vector<size_t> reachable_states(size_t state) const = 0;


protected:
  bool is_mdp; /*!< True iff mdp interpretation is possible */
  size_t n_states; /*!< Number of states in the model */
  size_t n_actions;  /*!< Number of actions in the model */
  size_t n_observations;  /*!< Number of observations in the model */
  size_t n_environments;  /*!< Number of environments */
  double discount; /*!< Discount factor */
};

#endif
