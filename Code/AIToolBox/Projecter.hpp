#ifndef AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE
#define AI_TOOLBOX_POMDP_PROJECTER_HEADER_FILE

#include <AIToolbox/ProbabilityUtils.hpp>
#include <AIToolbox/POMDP/Types.hpp>
//#include "../utils.hpp"

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

    private:
      // using PossibleObservationsTable = boost::multi_array<bool,  2>;

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
      //PossibleObservationsTable possibleObservations_;
    };

    template <typename M>
    Projecter<M>::Projecter(const M& model) : model_(model), S(model_.getS()), A(model_.getA()), O(model_.getO()), discount_(model_.getDiscount()),
					      immediateRewards_(A, S)/*, possibleObservations_(boost::extents[A][O])*/
    {
      //computePossibleObservations();
      computeImmediateRewards();
    }

    template <typename M>
    typename Projecter<M>::ProjectionsTable Projecter<M>::operator()(const VList & w) {
      ProjectionsTable projections( boost::extents[A][O] );

      for ( size_t a = 0; a < A; ++a )
	projections[a] = operator()(w, a);

      return projections;
    }

    template <typename M>
    typename Projecter<M>::ProjectionsRow Projecter<M>::operator()(const VList & w, size_t a) {
      ProjectionsRow projections( boost::extents[O] );

      // Observation 0 (impossible observation)
      projections[0].emplace_back(immediateRewards_.row(a), a, VObs(1,0));

      // Other obsevrations
      for ( size_t o = 1; o < O; ++o ) {

	for ( size_t i = 0; i < w.size(); ++i ) {
	  auto & v = std::get<VALUES>(w[i]);
	  MDP::Values vproj(S); vproj.fill(0.0);
	  // For each value function in the previous timestep, we compute the new value
	  // if we performed action a and obtained observation o.
	  std::vector<size_t> aux = previous_states(o);
	  for (auto it = aux.begin(); it != aux.end(); ++it) {
	    for (int e = 0; e < NPROFILES; e++) {
	      size_t s = e * O + *it;
	      size_t s1 = e * O + o;
	      vproj[s] += model_.getTransitionProbability(s, a, s1) * v[s1];
	    }
	  }
	  // Set new projection with found value and previous V id.
	  // projections[o].emplace_back(vproj, a, VObs(1,i));
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
	  size_t s1 = get_env(s) * O + next_state(get_rep(s), a);
	  immediateRewards_(a, s) += model_.getTransitionProbability(s,a,s1) * model_.getExpectedReward(s,a,s1);}
      }

      // You can find out why this is divided in the incremental pruning paper =)
      // The idea is that at the end of all the cross sums it's going to add up to the correct value.
      immediateRewards_ /= static_cast<double>(O);
    }
  }
}

#endif