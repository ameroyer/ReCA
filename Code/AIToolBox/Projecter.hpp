#ifndef AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE

#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/POMDP/Types.hpp>

namespace AIToolbox {
  namespace POMDP {

#ifndef DOXYGEN_SKIP
    // This is done to avoid bringing around the enable_if everywhere.
    template <typename M, typename = typename std::enable_if<is_model<M>::value>::type>
    class Projecter;
#endif
    /**
     * @brief This class offers projecting facilities for Models.
     */
    template <typename M>
    class Projecter<M> {
    public:
      using ProjectionsTable          = boost::multi_array<VList, 2>;
      using ProjectionsRow            = boost::multi_array<VList, 1>;

      /**
       * @brief Basic constructor.
       *
       * This constructor initializes the internal immediate reward table and the
       * table containing what are the possible observations for the model (this
       * may speed up the computation of the projections).
       *
       * @param model The model that is used as a base for all projections.
       */
      Projecter(const M & model);

      /**
       * @brief This function returns all possible projections for the provided VList.
       *
       * @param w The list that needs to be projected.
       *
       * @return A 2d array of projection lists.
       */
      ProjectionsTable operator()(const VList & w);

      /**
       * @brief This function returns all possible projections for the provided VList and action.
       *
       * @param w The list that needs to be projected.
       * @param a The action used for projecting the list.
       *
       * @return A 1d array of projection lists.
       */
      ProjectionsRow operator()(const VList & w, size_t a);

      /**
       * @brief This function returns the precomputed immediate rewards for the given action.
       *
       * @param a an action in the model
       *
       * @return irw the immediate rewards for action a (normalized by the number of observations.
       */
      Vector getImmediateRewards(size_t a);

    private:

      /**
       * @brief This function precomputes which observations are possible from specific actions.
       */
      //void computePossibleObservations();

      /**
       * @brief This function precomputes immediate rewards for the POMDP state-action pairs.
       */
      void computeImmediateRewards();

      const M & model_;
      size_t S, A, O;
      double discount_;

      Matrix2D immediateRewards_;
    };

    template <typename M>
    Vector Projecter<M>::getImmediateRewards(size_t a) {
      return immediateRewards_.row(a);
    }

    template <typename M>
    Projecter<M>::Projecter(const M& model) : model_(model), S(model_.getS()), A(model_.getA()), O(model_.getO()), discount_(model_.getDiscount()),
					      immediateRewards_(A, S)/*, possibleObservations_(boost::extents[A][O])*/
    {
      //computePossibleObservations(); // No need in our model. All observations, except 0 are possible after executing any action.
      computeImmediateRewards();
    }

    template <typename M>
    typename Projecter<M>::ProjectionsTable Projecter<M>::operator()(const VList & w) {
      ProjectionsTable projections( boost::extents[A][O] );

      for ( size_t a = 0; a < A; ++a ) {
	std::cerr << "\r          projection " << a + 1 << "/" << A;
	projections[a] = operator()(w, a);
      }
      std::cerr << "\r          projection " << A << "/" << A <<"             \n";

      return projections;
    }

    template <typename M>
    typename Projecter<M>::ProjectionsRow Projecter<M>::operator()(const VList & w, size_t a) {
      ProjectionsRow projections( boost::extents[O] );

      // Other (valid) observations
      for ( size_t o = 0; o < O; ++o ) {
	// OPT: We only consider the subset of pairs (s, s1) such that
	// - Obs(s1) = o
	// - T(s, a, s1) > 0 (ie Obs(s) = o' s.t. o' -> o and s same environment as s1)
	std::vector<size_t> aux = model_.previous_states(o);
	std::vector<std::pair<size_t, size_t> > pairs (model_.getE() * aux.size());
	size_t i = 0;
	for (int e = 0; e < model_.getE(); e++) {
	  size_t s1 = e * O + o;
	  for (auto it = aux.begin(); it != aux.end(); ++it) {
	    size_t s = e * O + *it;
	    pairs.at(i++) = std::make_pair(s, s1);
	  }
	}

	// Update vproj for every w
	for (i = 0; i < w.size(); ++i) {
	  auto & v = std::get<VALUES>(w[i]);
	  MDP::Values vproj(S); vproj.fill(0.0);
	  for (auto it = pairs.begin(); it != pairs.end(); ++it) {
	    vproj[std::get<0>(*it)] += model_.getTransitionProbability(std::get<0>(*it), a, std::get<1>(*it)) * v[std::get<1>(*it)];
	  }
	  // Set new projection with found value and previous V id.
	  projections[o].emplace_back(vproj * discount_ + immediateRewards_.row(a).transpose(), a, VObs(1,i));
	}
      }

      return projections;
    }


    template <typename M>
    void Projecter<M>::computeImmediateRewards() {
      immediateRewards_.fill(0.0);
      for ( size_t a = 0; a < A; ++a ) {
	for ( size_t s = 0; s < S; ++s ) {
	  std::vector<size_t> target = model_.reachable_states(s);
	  for (auto it = target.begin(); it != target.end(); ++it) {
	    // OPT: Only one s1 such that T(s, a, s1) and R(s, a, s1) are both non-null
	    immediateRewards_(a, s) += model_.getTransitionProbability(s, a, *it) * model_.getExpectedReward(s, a, *it);
	  }
	}
      }

      // You can find out why this is divided in the incremental pruning paper =)
      // The idea is that at the end of all the cross sums it's going to add up to the correct value.
      immediateRewards_ /= static_cast<double>(O);
    }
  }
}

#endif
