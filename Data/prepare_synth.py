#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a synthetic POMDP model with high discrepancy between environments.
"""
__author__ = "Amelie Royer"
__email__ = "amelie.royer@ist.ac.at"


import sys, os
import csv
import argparse
import numpy as np
from random import randint
from utils import *

def init_output_dir(nitems, hlength):
    """
    Initializa the output directory.

    Args:
     * ``nitems`` (*int*): Number of actions/items in the dataset.
     * ``hlength`` (*int*): history length.

    Returns:
     * ``output_base`` (*str*): base name for output files.
    """
    import shutil
    output_base = "synth_u%d_k%d_pl%d" % (nitems, hlength, nitems)
    output_dir = os.path.join(args.output, "Synth%d%d%d" % (nitems, hlength, nitems))
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return os.path.join(output_dir, output_base)

#####################################################   M A I N    R O U T I N E  #######
if __name__ == "__main__":
    ###### 0. Set Parameters
    base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description='Generate syntehtic POMCP parameters with a high discrepancy between environments.')
    parser.add_argument('-o', '--output', type=str, default=os.path.join(base_folder, "Code", "Models"), help="Path to output directory.")
    parser.add_argument('-n', '--nactions', type=int, default=3, help="Number of items (actions) and clusters.")
    parser.add_argument('-k', '--history', type=int, default=2, help="Length of the history to consider for one state of the MEMDP.")
    parser.add_argument('-t', '--test', type=int, default=2000, help="Number of test sessions to generate.")
    parser.add_argument('--norm', action='store_true', help="If present, normalize the output transition probabilities.")
    parser.add_argument('-a', '--alpha', type=float, default=1.1, help="Positive rescaling of transition probabilities matching the recommendation.")
    args = parser.parse_args()

    ###### 0-bis. Check assertions
    assert(args.nactions > 0), "plevel argument must be strictly positive"
    assert(args.history > 1), "history length must be strictly greater than 1"
    assert(args.test > 0), "Number of test sessions must be strictly positive"
    logger = Logger(sys.stdout)
    sys.stdout = logger

    ###### 1. Load data and create product/user profile
    n_items = args.nactions
    n_users = args.nactions
    actions = range(1, n_items + 1)
    init_base_writing(n_items, args.history)
    n_states = get_nstates(n_items, args.history)
    output_base = init_output_dir(args.nactions, args.history)

    #### 2. Write .items and .profiles dummy files
    with open("%s.items" % output_base, "w") as f:
        f.write("\n".join("Item %d" % i for i in xrange(n_items)))

    with open("%s.profiles" % output_base, "w") as f:
        f.write("\n".join("%d\t1\t1" % i for i in xrange(n_users)))

    ##### 3. Create dummy test sessions
    exc = 4 * (n_users - 1)  # Sample size. Ensure 0.8 probability given to action i
    with open("%s.test" % output_base, 'w') as f:
        for user in xrange(args.test):
            cluster = randint(0, n_users - 1)
            lgth = randint(10, 100)
            session = [0]
            for _ in xrange(lgth):
                a = randint(0, exc + n_users - 2)
                if a < n_users - 1:
                    if a == cluster:
                        a = n_users - 1
                    a += 1
                else:
                    a = cluster + 1
                s2 = get_next_state_id(session[-1], a)
                session.append(a)
                session.append(s2)
            f.write("%d\t%d\t%s\n" % (user, cluster, ' '.join(str(x) for x in session) ))

    ###### 4. Set rewards
    with open("%s.rewards" % output_base, 'w') as f:
        for s1 in xrange(n_states):
            for item in actions:
                f.write("%d\t%d\t%d\t%.5f\n" % (s1, item, get_next_state_id(s1, item), 1))

    ###### 5. Create transition function
    print "\n\033[91m-----> Probability inference\033[0m"
    total_count = exc + n_items - 1
    with open("%s.transitions" % output_base, 'w') as f:
        for user_profile in xrange(n_users):
            print >> sys.stderr, "\n   > Profile %d / %d: \n" % (user_profile + 1, n_users),
            sys.stderr.flush()
            # For fixed s1
            for s1 in xrange(n_states):
                sys.stderr.write("      state: %d / %d   \r" % (s1 + 1, n_states))
                sys.stderr.flush()
                # For fixed a
                for a in actions:
                    # Positive P(s1 -a-> s1.a)
                    count = exc if a == user_profile + 1 else 1
                    new_count = args.alpha * count if not args.norm else args.alpha * count / total_count
                    s2 = get_next_state_id(s1, a)
                    f.write("%d\t%d\t%d\t%s\n" % (s1, a, s2, new_count))
                    # Negative P(s1 -a-> s1.b), b!= a
                    beta = (total_count - args.alpha * count) / (total_count - count)
                    # For every s2, sample T(s1, a, s2)
                    for s2_link in actions:
                        if (s2_link != a):
                            s2 = get_next_state_id(s1, s2_link)
                            count = exc if s2_link == user_profile + 1 else 1
                            f.write("%d\t%d\t%d\t%s\n" % (s1, a, s2, beta * count if not args.norm else beta * count / total_count))
            f.write("\n")

    ###### 6. Summary
    print "\n\n\033[92m-----> End\033[0m"
    print "   All outputs are in %s" % output_base
    with open("%s.summary" % output_base, 'w') as f:
        f.write("%d States\n%d Actions (Items)\n%d user profiles\n%d history length\n%d product clustering level\n\n%s" % (n_states, n_items, n_users, args.history, args.nactions, logger.to_string()))
    print
    # End
