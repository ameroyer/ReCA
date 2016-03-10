/* ---------------------------------------------------------------------------
** recomodel.hpp
** Represent a MEMDP model constructed from data for a recommender system task
**
** Author: Amelie Royer
** Email: amelie.royer@ist.ac.at
** -------------------------------------------------------------------------*/

#include "model.hpp"
#include <iostream>


 
class Recomodel: public Model {
private:
  double* transition_matrix; /*!< Transition matrix */
  double* rewards; /*!< Rewards matrix */
  int hlength; /*!< History length */
  bool is_mdp; /*!< True if mdp mode is activated */
  int pows[hlength];   /*!< Precomputed exponents for conversion to base n_items */
  int acpows[hlength]; /*!< Cumulative exponents for conversion from base n_items */

  /*! \briefGiven an environment e, state s1, action a and state s2 (suffix), 
   * returns the corresponding index in an 1D array.
   */
  int index(size_t env, size_t s, size_t a, size_t link) {
    //TODO

  }
  /*! \brief Precomputes the ``n_actions`` exponents for conversion of decimals
   * to and from the ``n_actions`` base.
   */
  void init_pows() {
    pows[hlength - 1] = 1;
    acpows[hlength - 1] = 1;
    for (int i = hlength - 2; i >= 0; i--) {
      pows[i] = pows[i + 1] * n_actions;
      acpows[i] = acpows[i + 1] + pows[i];
    }
  }

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
  size_t state_to_id(std::vector<size_t> state) {
    size_t id = 0;
    for (int i = 0; i < hlength; i++) {
      id += state.at(i) * pows[i];
    }
    return id;
  }


  /*! \brief Returns the sequence of items selection corresponding to the given state index.
   * Mostly used for debugging purposes.
   *
   * \param id unique state index.
   *
   * \return state a state, represented by a sequence of selected items.
   */
  std::vector<size_t> id_to_state(size_t id) {
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


public:
  /*! \brief Construct a MEMDP model from a given recommendation dataset.
   *
   */
  Recomodel(std::string tfile, std::string rfile, std::string sfile, 
	    double precision, bool is_mdp_) {

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
    assert((pow(n_actions, hlength + 1) - 1) / (n_actions - 1), 
	   "number of observations and actions do not match");
    
    //********** Initialize
    has_mdp = true;
    is_mdp = is_mdp_;
    n_states = (is_mdp ? n_observations : n_environments * n_observations);
    rewards = new double[n_actions](); // rewards[a] = profit for item a
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
    infile.close();
  }


  /*! \brief Destructor
   */
  ~Recomodel {
    delete []transition_matrix;
    delete []rewards;
  }


  /*! \brief Load rewards of the model from file
   *
   * \param rfile Rewards file
   */
  void load_rewards(std::string rfile) {    
    double v;
    size_t a;
    int rewards_found = 0;

    infile.open(rfile, std::ios::in);
    assert((".rewards file not found", infile.is_open()));
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      if (!(iss >> a >> v)) { break; }
      assert(("Unvalid reward entry", a <= n_actions));
      rewards[a - 1] = v;
      rewards_found++;
    }
    assert(("Missing item while parsing .rewards file",
	    rewards_found == n_actions));
    infile.close();
  }


  /*! \brief Load transitions of the model from file
   *
   * \param tfile Transition file.
   * \param precision If true, precise normalization is enabled.
   */
  void load_transitions(std::string tfile, bool precision) {    
    double v;
    size_t s1, a, s2, link, p;
    int transitions_found = 0, profiles_found = 0;
  
    infile.open(tfile, std::ios::in);
    assert((".transitions file not found", infile.is_open()));
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      // Change profile
      if (!(iss >> s1 >> a >> s2 >> v)) {
	profiles_found += 1;
	assert(("Incomplete transition function in current profile in .transitions",
		transitions_found == n_observations * n_actions * n_actions));
	assert(("Too many profiles found in .transitions file",
		profiles_found <= n_environments));
	transitions_found = 0;
	continue;
      }
      // Set transition
      link = this->is_connected(s1, s2);
      assert(("Unfeasible transition with >0 probability", link < n_actions));
      if (is_mdp) {
	transition_matrix[index(0, s1, a - 1, link)] += v;
      } else {
	transition_matrix[index(profiles_found, s1, a - 1, link)] = v;
      }
      transitions_found++;
    }
    assert(("Missing profiles in .transitions file", profiles_found == n_environments));
    infile.close();

    //TODO Normalization
    double nrm;
    for (p = 0; p < n_environments; p++) {
      for (s1 = 0; s1 < n_observations; s1++) {
	for (a = 0; a < n_actions; a++) {
	  // If asking for precision, use kahan summation [slightly slower]
	  if (precision) {
	    double kahan_correction = 0.0;
	    nrm = 0.0;
	    for (s2 = 0; s2 < n_actions; s2++) {
	      double val = transition_matrix[p][s1][a][s2] - kahan_correction;
	      double aux = nrm + val;
	      kahan_correction = (aux - nrm) - val;
	      nrm = aux;
	    }
	  }
	  // Else basic sum
	  else {
	    nrm = std::accumulate(transition_matrix[p][s1][a],
				  transition_matrix[p][s1][a] + n_actions, 0.);
	  }
	  std::transform(transition_matrix[p][s1][a],
			 transition_matrix[p][s1][a] + n_actions,
			 transition_matrix[p][s1][a],
			 [nrm](const double t){ return t / nrm ; }
			 );
	}
      }
    }
  }


  /*! \brief Returns a given transition probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return P( s2 | s1 -a-> ).
   */
  double getTransitionProbability(size_t s1, size_t a, size_t s2) const {
    size_t link = is_connected(s1, s2);
    if (link >= n_actions) {
      return 0.;
    } else {
      return (is_mdp ? transition_matrix[index(0, s1, a, link)] : transition_matrix[index(get_env(s1), get_rep(s1), a, link)]);
    }   
  }

  /*! \brief Returns a given observation probability.
   *
   * \param s1 origin statte.
   * \param a chosen action.
   * \param o observation.
   *
   * \return P( o | -a-> s1 ).
   */
  double getObservationProbability(size_t s1, size_t a, size_t o) const {
    if (get_rep(s1) == o) {
      return 1.;
    } else {
      return 0.;
    }
  }

  /*! \brief Returns a given reward.
   *
   * \param s1 origin state.
   * \param a chosen action.
   * \param s2 arrival state.
   *
   * \return R(s1, a, s2).
   */
  virtual double getExpectedReward(size_t s1, size_t a, size_t s2) const {
    size_t link = is_connected(s1, s2);
    if (link != a) {
      return 0.;
    } else {
      return rewards[link];
    }
  }

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
  bool isTerminal(size_t s) const {
    return false;
  }



protected:  

  /*! \brief Given a state, returns all its predecessors.
   *
   * \param state unique state index.
   *
   * \return next_state index of the state corresponding to the user choosing ``item`` in ``state``.
   */
  virtual std::vector<size_t> previous_states(size_t state) {
    div_t aux = div(state, n_actions);
    int prefix_s2 = ((aux.rem == 0) ? aux.quot - 1 : aux.quot);
    if (state == 0) {
      std::vector<size_t> prev;
      return prev;
    }
    if (prefix_s2 < acpows[1]) {
      std::vector<size_t> prev(1);
      prev.at(0) = prefix_s2;
      return prev;
    } else {
      std::vector<size_t> prev(n_actions + 1);
      for (size_t a = 0; a <= n_actions; a++) {
	prev.at(a) = prefix_s2 + a * pows[0];
      }
      return prev;
    }
  }

  /*! \brief Given a state and choice (e.g. item, direction) , return the next user state.
   *
   * \param state unique state index.
   * \param item user choice [0 to n_actions - 1].
   *
   * \return next_state index of the state corresponding to the user choosing ``choice`` in ``state``.
   */
  virtual size_t next_state(size_t state, size_t choice) {
    size_t aux = state % pows[0];
    if (aux >= acpows[1] || state < pows[0]) {
      return aux * n_actions + choice + 1;
    } else {
      return (pows[0] + aux) * n_actions + choice + 1;
    }
  }


  /*! \brief Given two states s1 and s2, return the action a such that s2 = s1.a if it exists,
   * or the value ``n_actions`` otherwise.
   *
   * \param s1 unique state index.
   * \param s2 unique state index
   *
   * \return link a valid action index [0 to n_actions - 1] if s1 and s2 can be connected, n_actions otherwise.
   */
  size_t is_connected(size_t a, size_t b) {
    size_t s1, s2;
    if (is_mdp) {
      s1 = a;
      s2 = b;
    } else {
      if (get_env(s1) != get_env(s2)) {
	return n_actions;
      } else {
	s1 = get_rep(a);
	s2 = get_rep(b);
      }
    }
    // Find suffix of first state
    int suffix_s1 = s1 % pows[0];
    suffix_s1 = ((suffix_s1 >= acpows[1] || s1 < pows[0])  ? suffix_s1 - acpows[1] : suffix_s1 + pows[0] - acpows[1]);
    // Find prefix of second state
    div_t aux = div(s2, n_actions);
    int prefix_s2 = aux.quot - acpows[1];
    size_t last_s2 = aux.rem - 1;
    if (aux.rem == 0) {
      prefix_s2 -= 1;
      last_s2 = n_actions - 1;
    }
    if (prefix_s2 == suffix_s1) {
      return last_s2;
    } else {
      return n_actions;
    }
  }


};
 
