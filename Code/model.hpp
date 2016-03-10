/* ---------------------------------------------------------------------------
** model.hpp
** Main abstract class to represent a MEMDP model
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/


#include <iostream>
 
class Model {
public:
  /*! \brief Default constructor
   *
   */
  Model();

  /*! \brief Default destructor.
   */
  ~Model();

  /*! \brief Returns true iff the model can also be used as a MDP .
   * In this case, select the correct setting with the protected variable ``mode``.
   *
   * \return true if the model can be interpreted as a MDP.
   */
  bool mdp_enabled const { return has_mdp; }

  /*! \brief Returns the number of states in the model.
   *
   * \return the number of states in the model.
   */
  size_t getS() const { return n_states; };

  /*! \brief Returns the number of observations in the model.
   *
   * \return number of observations in the model.
   */
  size_t getO() const { return n_observations; }

  /*! \brief Returns the number of actionss in the model.
   *
   * \return number of actions in the model.
   */
  size_t getA() const { return n_actions; }

  /*! \brief Returns the discount factor in the model.
   *
   * \return Discount factor in the MDP.
   */
  double getDiscount() const { return discount; }

  /*! \brief Returns a given transition probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return P( s2 | s1 -a-> ).
   */
  virtual double getTransitionProbability( size_t s1, size_t a, size_t s2 ) const;

  /*! \brief Returns a given observation probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param o observation.
   *
   * \return P( o | -a-> s1 ).
   */
  virtual getObservationProbability(size_t s1, size_t a, size_t o) const;

  /*! \brief Returns a given reward.
   *
   * \param s1 origin state.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return R(s1, a, s2).
   */
  virtual double getExpectedReward( size_t s1, size_t a, size_t s2 ) const;

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
  virtual bool isTerminal(size_t s) const;



protected:
  bool has_mdp; /*!< True iff mdp interpretation is possible */
  bool with_structure; /*< True iff the structure of the model can be used for optimizations (through functions previous_states etc > */
  size_t n_states; /*!< Number of states in the model */
  size_t n_actions;  /*!< Number of actions in the model */
  size_t n_observations;  /*!< Number of observations in the model */
  size_t n_environments;  /*!< Number of environments */
  double discount; /*!< Discount factor */
  

  /*!
   * \brief Given a state of the MEMDP, returns the corresponding environment.
   *
   * \param s a state in the MEMDP.
   *
   * \return the environment to which s belongs.
   */
  size_t get_env(size_t s) { return s / n_observations; }

  /*!
   * \brief Given a state of the MEMDP, returns its representative/observation.
   *
   * \param s a state in the MEMDP.
   *
   * \return the corresponding observation.
   */
  size_t get_rep(size_t s) { return s % n_observations; }

  /*! \brief Given a state, returns all its predecessors.
   *
   * \param state unique state index.
   *
   * \return next_state index of the state corresponding to the user choosing ``item`` in ``state``.
   */
  virtual std::vector<size_t> previous_states(size_t state);

  /*! \brief Given a state and choice (e.g. item, direction) , return the next user state.
   *
   * \param state unique state index.
   * \param item user choice [0 to n_actions - 1].
   *
   * \return next_state index of the state corresponding to the user choosing ``choice`` in ``state``.
   */
  virtual size_t next_state(size_t state, size_t choice);

};
 
