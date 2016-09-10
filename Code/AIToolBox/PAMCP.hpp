#ifndef AI_TOOLBOX_POMDP_PAMCP_HEADER_FILE
#define AI_TOOLBOX_POMDP_PAMCP_HEADER_FILE

#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/Impl/Seeder.hpp>

#include <unordered_map>
#include <iostream>

namespace AIToolbox {
  namespace POMDP {

#ifndef DOXYGEN_SKIP
    // This is done to avoid bringing around the enable_if everywhere.
    template <typename M, typename = typename std::enable_if<is_generative_model<M>::value>::type>
    class PAMCP;
#endif

    /**
     * @brief This class represents the POMCP optimized for our MEMDP model.
     *
     */
    template <typename M>
    class PAMCP<M> {
    public:
      using SampleBelief = std::vector<size_t>;

      struct BeliefNode;
      using BeliefNodes = std::unordered_map<size_t, BeliefNode>;

      struct ActionNode {
	BeliefNodes children;
	double V = 0.0;
	unsigned N = 0;
      };
      using ActionNodes = std::vector<ActionNode>;

      struct BeliefNode {
	BeliefNode() : N(0) {}
	BeliefNode(size_t s) : belief(1, s), N(0) {}
	ActionNodes children;
	SampleBelief belief;
	unsigned N;
      };

      /**
       * @brief Basic constructor.
       *
       * @param m The POMDP model that POMCP will operate upon.
       * @param n_environments the number of environments in the MEMDP.
       * @param beliefSize The size of the initial particle belief.
       * @param iterations The number of episodes to run before completion.
       * @param exp The exploration constant. This parameter is VERY important to determine the final POMCP performance.
       */
      PAMCP(const M& m, size_t beliefSize, unsigned iterations, double exp);

      /**
       * @brief This function resets the internal graph and samples
       * for the provided belief and horizon.
       *
       *
       * @param be The initial belief online environment.
       * @param o The initial observation / user history
       * @param horizon The horizon to plan for.
       *
       * @return The best action.
       */
      size_t sampleAction(const Belief& be, size_t o, unsigned horizon, bool start_session=false);

      /**
       * @brief This function uses the internal graph to plan.
       *
       * This function can be called after a previous call to
       * sampleAction with a Belief. Otherwise, it will invoke it
       * anyway with a random belief.
       *
       * If a graph is already present though, this function will
       * select the branch defined by the input action and
       * observation, and prune the rest. The search will be started
       * using the existing graph: this should make search faster,
       * and also not require any belief updates.
       *
       * NOTE: Currently there is no particle reinvigoration
       * implemented, so for long horizons you can expect
       * progressively degrading performances.
       *
       * @param a The action taken in the last timestep.
       * @param o The observation received in the last timestep.
       * @param horizon The horizon to plan for.
       *
       * @return The best action.
       */
      size_t sampleAction(size_t a, size_t o, unsigned horizon);

      /**
       * @brief This function sets the new size for initial beliefs created from sampleAction().
       *
       * Note that this parameter does not bound particle beliefs
       * created within the tree by result of rollouts: only the ones
       * directly created from true Beliefs.
       *
       * @param beliefSize The new particle belief size.
       */
      void setBeliefSize(size_t beliefSize);

      /**
       * @brief This function sets the number of performed rollouts in POMCP.
       *
       * @param iter The new number of rollouts.
       */
      void setIterations(unsigned iter);

      /**
       * @brief This function sets the new exploration constant for POMCP.
       *
       * This parameter is EXTREMELY important to determine POMCP
       * performance and, ultimately, convergence. In general it is
       * better to find it empirically, by testing some values and
       * see which one performs best. Tune this parameter, it really
       * matters!
       *
       * @param exp The new exploration constant.
       */
      void setExploration(double exp);

      /**
       * @brief This function returns the POMDP generative model being used.
       *
       * @return The POMDP generative model.
       */
      const M& getModel() const;

      /**
       * @brief This function returns a reference to the internal graph structure holding the results of rollouts.
       *
       * @return The internal graph.
       */
      const BeliefNode& getGraph() const;

      /**
       * @brief This function returns the initial particle size for converted Beliefs.
       *
       * @return The initial particle count.
       */
      size_t getBeliefSize() const;

      /**
       * @brief This function returns the number of iterations performed to plan for an action.
       *
       * @return The number of iterations.
       */
      unsigned getIterations() const;

      /**
       * @brief This function returns the currently set exploration constant.
       *
       * @return The exploration constant.
       */
      double getExploration() const;

    private:
      const M& model_;
      size_t S, A, O, E, beliefSize_;
      unsigned iterations_, maxDepth_;
      double exploration_;

      SampleBelief sampleBelief_;
      BeliefNode graph_;
      BeliefNode fullgraph_;
      std::vector<std::pair<size_t, size_t> > history;
      bool reset_belief = true;
      bool to_update = true;

      mutable std::default_random_engine rand_;

      /**
       * @brief This function starts the simulation process.
       *
       * This function simply calls simulate() for the number of
       * times specified by POMCP's parameters. While doing so it
       * builds a tree of explored outcomes, from which POMCP will
       * then extract the best expected action for the current
       * belief.
       *
       * @param horizon The horizon for which to plan.
       *
       * @return The best action to take given the final built tree.
       */
      size_t runSimulation(unsigned horizon);

      /**
       * @brief This function recursively simulates the model while building the tree.
       *
       * From the given belief node, state and horizon, this function
       * selects an action based on UCT (so that estimated good
       * actions are taken more often than estimated bad actions) and
       * samples a new state, observation and reward. Based on the
       * observation, the function detects whether it is at the end
       * of the tree or not. If it is, it adds a new node to the tree
       * and rollouts the rest of the episode. Otherwise it
       * recursively traverses the tree.
       *
       * The states and rewards obtained on the way are used to
       * update particle beliefs within the tree and the value
       * estimations for those beliefs.
       *
       * @param b The tree node to simulate from.
       * @param s The state from which we are simulating, possibly a particle of a previous particle belief.
       * @param horizon The depth within the tree already reached.
       *
       * @return The discounted reward obtained from the simulation performed from here to the end.
       */
      double simulate(BeliefNode & b, size_t s, unsigned horizon);

      /**
       * @brief This function implements the rollout policy for POMCP.
       *
       * This function extracts some cumulative reward from a
       * particular state, given that we have reached a particular
       * horizon. The idea behind this function is to approximate the
       * true value of the state; since this function is called when
       * we are at the leaves of our tree, the only way for us to
       * extract more information is to simply simulate the rest of
       * the episode directly.
       *
       * However, in order to speed up the process and store only
       * useful data, we avoid inserting every single state that we
       * see here into the tree, preferring to add a single state at
       * a time. This avoids wasting lots of computation and memory
       * on states far from our root that we will probably never see
       * again, while at the same time still getting an estimate for
       * the rest of the simulation.
       *
       * @param s The state from which to start the rollout.
       * @param horizon The horizon already reached while simulating inside the tree.
       *
       * @return An estimate return computed from simulating until max depth.
       */
      double rollout(size_t s, unsigned horizon);


      /**
       * @brief This function finds the best action based on value.
       *
       * @tparam Iterator An iterator to an ActionNode.
       * @param begin The beginning of a list of ActionNodes.
       * @param end The end of the list.
       *
       * @return The iterator to the ActionNode with the best value.
       */
      template <typename Iterator>
      Iterator findBestA(Iterator begin, Iterator end);

      /**
       * @brief This function finds the best action based on UCT.
       *
       * UCT gives a bonus to actions that have been tried very few
       * times, in order to void thinking that a bad action is bad
       * just because it got unlucky the few times that it tried it.
       *
       * @tparam Iterator An iterator to an ActionNode.
       * @param begin The beginning of a list of ActionNodes.
       * @param end The end of the list.
       * @param count The sum of all action counts.
       *
       * @return The iterator to the ActionNode to be selected based on UCT.
       */
      template <typename Iterator>
      Iterator findBestBonusA(Iterator begin, Iterator end, unsigned count);

      /**
       * @brief This function samples a given belief in order to produce a particle approximation of it.
       *
       * @param b The belief (over environment) to be approximated.
       * @param o The current observation to determine the belief
       * over states.
       *
       * @return A particle belief approximating the input belief.
       */
      SampleBelief makeSampledBelief(const Belief & b, size_t o);

      /**
       * @brief This function updates the full graph with the current state of the simulation.
       * More precisely, the last branch/actionNode corresponding to the current history is
       * replaced by the currently build tree.
       *
       * @param current The subtree computed for action a.
       * @param a The last action taken in the simulation.
       */
      //void update_fullgraph(ActionNode current, size_t a);
      void update_fullgraph(BeliefNode current, size_t a);
    };

    template <typename M>
    PAMCP<M>::PAMCP(const M& m, size_t beliefSize, unsigned iter, double exp) : model_(m), S(model_.getS()), A(model_.getA()), O(model_.getO()), E(model_.getE()), beliefSize_(beliefSize), iterations_(iter),
												       exploration_(exp), graph_(), rand_(Impl::Seeder::getSeed()) {}

    template <typename M>
    size_t PAMCP<M>::sampleAction(const Belief& be, size_t o, unsigned horizon, bool start_session /* false */) {
      // Reset graph initially or with new belief (e.g. observation missing)
      if (reset_belief) {
	graph_ = BeliefNode(A);
	graph_.children.resize(A);
	reset_belief = false;
      }
      // Reset with the stored information
      else {
	graph_ = fullgraph_;
      }
      // Initialize full graph
      if (start_session && history.size() == 0) {
	fullgraph_ = BeliefNode(A);
	fullgraph_.children.resize(A);
      }
      // Clear history if beginning
      if (start_session) {
	history.clear();
	to_update = true;
      }

      // Init the belief
      auto b = Belief(be);
      graph_.belief = makeSampledBelief(b, o);

      return runSimulation(horizon);
    }

    template <typename M>
    size_t PAMCP<M>::sampleAction(size_t a, size_t o, unsigned horizon) {
      // Update full graph
      if (to_update) {
	update_fullgraph(graph_, a);
	history.push_back(std::make_pair(a, o));
      }

      // Run simulation
      auto & obs = graph_.children[a].children;

      auto it = obs.find(o);
      if ( it == obs.end() ) {
	std::cerr << "\nObservation " << o << " never experienced in simulation, restarting belief from " << o << "\n";
	auto b = Belief(E); b.fill(1.0 / E);
	reset_belief = true;
	to_update = false;
	return sampleAction(b, o, horizon, false);
      }

      // Here we need an additional step, because *it is contained by graph_.
      // If we just move assign, graph_ is first going to delete everything it
      // contains (included *it), and then we are going to move unallocated memory
      // into graph_! So we move *it outside of the graph_ hierarchy, so that
      // we can then assign safely.
      { auto tmp = std::move(it->second); graph_ = std::move(tmp); }

      if ( ! graph_.belief.size() ) {
	std::cerr << "POMCP Lost track of the belief, restarting with uniform..\n";
	auto b = Belief(E); b.fill(1.0 / E);
	reset_belief = true;
	to_update = false;
	return sampleAction(b, o, horizon);
      }

      // We resize here in case we didn't have time to sample the new
      // head node. In this case, the new head may not have children.
      // This would break the UCT call.
      graph_.children.resize(A);

      return runSimulation(horizon);
    }


    template <typename M>

    void PAMCP<M>::update_fullgraph(BeliefNode current, size_t a) {
      auto & current_branch = fullgraph_;
      // Browse history
      for (auto it = history.begin(); it != history.end(); ++it) {
	auto  an = current_branch.children[std::get<0>(*it)];
	current_branch = an.children[std::get<1>(*it)];
      }
      // Modify
      current_branch = current;
    }



    template <typename M>
    size_t PAMCP<M>::runSimulation(unsigned horizon) {
      if ( !horizon ) return 0;

      maxDepth_ = horizon;
      std::uniform_int_distribution<size_t> generator(0, graph_.belief.size()-1);

      for (unsigned i = 0; i < iterations_; ++i )
	simulate(graph_, graph_.belief.at(generator(rand_)), 0);

      auto begin = std::begin(graph_.children);
      return std::distance(begin, findBestA(begin, std::end(graph_.children)));
    }

    template <typename M>
    double PAMCP<M>::simulate(BeliefNode & b, size_t s, unsigned depth) {
      b.N++;

      auto begin = std::begin(b.children);
      size_t a = std::distance(begin, findBestBonusA(begin, std::end(b.children), b.N));

      size_t s1, o; double rew;
      std::tie(s1, o, rew) = model_.sampleSOR(s, a);

      auto & aNode = b.children[a];

      {
	double futureRew = 0.0;
	// We need to append the node anyway to perform the belief
	// update for the next timestep.
	auto ot = aNode.children.find(o);
	if ( ot == std::end(aNode.children) ) {
	  aNode.children.emplace(std::piecewise_construct,
				 std::forward_as_tuple(o),
				 std::forward_as_tuple(s1));
	  // This stops automatically if we go out of depth
	  futureRew = rollout(s1, depth + 1);
	}
	else {
	  ot->second.belief.push_back(s1);
	  // We only go deeper if needed (maxDepth_ is always at least 1).
	  if ( depth + 1 < maxDepth_ && !model_.isTerminal(s1) ) {
	    // Since most memory is allocated on the leaves,
	    // we do not allocate on node creation but only when
	    // we are actually descending into a node. If the node
	    // already has memory this should not do anything in
	    // any case.
	    ot->second.children.resize(A);
	    futureRew = simulate( ot->second, s1, depth + 1 );
	  }
	}

	rew += model_.getDiscount() * futureRew;
      }

      // Action update
      aNode.N++;
      aNode.V += ( rew - aNode.V ) / static_cast<double>(aNode.N);

      return rew;
    }

    template <typename M>
    double PAMCP<M>::rollout(size_t s, unsigned depth) {
      double rew = 0.0, totalRew = 0.0, gamma = 1.0;

      std::uniform_int_distribution<size_t> generator(0, A-1);
      for ( ; depth < maxDepth_; ++depth ) {
	std::tie( s, rew ) = model_.sampleSR( s, generator(rand_) );

	totalRew += gamma * rew;
	gamma *= model_.getDiscount();
      }
      return totalRew;
    }

    template <typename M>
    template <typename Iterator>
    Iterator PAMCP<M>::findBestA(Iterator begin, Iterator end) {
      return std::max_element(begin, end, [](const ActionNode & lhs, const ActionNode & rhs){ return lhs.V < rhs.V; });
    }

    template <typename M>
    template <typename Iterator>
    Iterator PAMCP<M>::findBestBonusA(Iterator begin, Iterator end, unsigned count) {
      // Count here can be as low as 1.
      // Since log(1) = 0, and 0/0 = error, we add 1.0.
      double logCount = std::log(count + 1.0);
      // We use this function to produce a score for each action. This can be easily
      // substituted with something else to produce different POMCP variants.
      auto evaluationFunction = [this, logCount](const ActionNode & an){
	return an.V + exploration_ * std::sqrt( logCount / an.N );
      };

      auto bestIterator = begin++;
      double bestValue = evaluationFunction(*bestIterator);

      for ( ; begin < end; ++begin ) {
	double actionValue = evaluationFunction(*begin);
	if ( actionValue > bestValue ) {
	  bestValue = actionValue;
	  bestIterator = begin;
	}
      }

      return bestIterator;
    }

    template <typename M>
    typename PAMCP<M>::SampleBelief PAMCP<M>::makeSampledBelief(const Belief & b, size_t o) {
      SampleBelief belief;
      belief.reserve(beliefSize_);

      for ( size_t i = 0; i < beliefSize_; ++i )
	belief.push_back(sampleProbability(E, b, rand_) * O + o);

      return belief;
    }

    template <typename M>
    void PAMCP<M>::setBeliefSize(size_t beliefSize) {
      beliefSize_ = beliefSize;
    }

    template <typename M>
    void PAMCP<M>::setIterations(unsigned iter) {
      iterations_ = iter;
    }

    template <typename M>
    void PAMCP<M>::setExploration(double exp) {
      exploration_ = exp;
    }

    template <typename M>
    const M& PAMCP<M>::getModel() const {
      return model_;
    }

    template <typename M>
    const typename PAMCP<M>::BeliefNode& PAMCP<M>::getGraph() const {
      return graph_;
    }

    template <typename M>
    size_t PAMCP<M>::getBeliefSize() const {
      return beliefSize_;
    }

    template <typename M>
    unsigned PAMCP<M>::getIterations() const {
      return iterations_;
    }

    template <typename M>
    double PAMCP<M>::getExploration() const {
      return exploration_;
    }
  }
}

#endif
