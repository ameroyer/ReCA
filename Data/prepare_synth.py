from __future__ import print_function
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a synthetic recommendation task MEMDP with high discrepancy between the environments.
"""
__author__ = "Amelie Royer"
__email__ = "amelie.royer@ist.ac.at"


import sys, os
import gzip
import argparse
from random import randint
from utils import ChunkedWriter, Logger, init_base_writing, get_nstates, get_next_state_id

if sys.version_info[0] == 3:
    xrange = range

def init_output_dir(nitems, hlength):
    """
    Initializes the output directory.

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
    ###### 0. Parameters
    base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="Generate a synthetic recommendation task MEMDP with high discrepancy between the environments.")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(base_folder, "Code", "Models"), help="Output directory.")
    parser.add_argument('-n', '--nactions', type=int, default=3, help="Number of items.")
    parser.add_argument('-k', '--history', type=int, default=2, help="History length.")
    parser.add_argument('-a', '--alpha', type=float, default=1.1, help="Positive rescaling of transition probabilities matching the recommendation.")
    parser.add_argument('-t', '--test', type=int, default=2000, help="Number of test runs.")
    parser.add_argument('--norm', action='store_true', help="If present, normalize the output transition probabilities.")
    parser.add_argument('--zip', action='store_true', help="If present, the transitiosn are output in a compressed file.")
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
    n_states = int(get_nstates(n_items, args.history))
    output_base = init_output_dir(args.nactions, args.history)

    #### 2. Write .items and .profiles dummy files
    with open("%s.items" % output_base, "w") as f:
        f.write("\n".join("Item %d" % i for i in xrange(n_items)))

    with open("%s.profiles" % output_base, "w") as f:
        f.write("\n".join("%d\t1\t1" % i for i in xrange(n_users)))

    ##### 3. Create dummy test sessions
    print("\n\033[91m-----> Test sequences generation\033[0m")
    exc = 4 * (n_users - 1)  # Sample size. Ensure 0.8 probability given to action i
    with open("%s.test" % output_base, 'w') as f:
        for user in xrange(args.test):
            sys.stderr.write("       %d / %d   \r" % (user + 1, args.test))
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
    print("\n\n\033[91m-----> Rewards generation\033[0m")
    with open("%s.rewards" % output_base, 'w') as f:
        for item in actions:
            sys.stderr.write("      item: %d / %d   \r" % (item, len(actions)))
            f.write("%d\t%.5f\n" % (item, 1))

    ###### 5. Create transition function
    # Write
    f = gzip.open("%s.transitions.gz" % output_base, 'w') if args.zip else open("%s.transitions" % output_base, 'w')
    buffer_size = 2**16
    print("\n\n\033[91m-----> Probability inference\033[0m")
    total_count = exc + n_items - 1
    transitions_str = ""
    for user_profile in xrange(n_users):
        print("\n   > Profile %d / %d: \n" % (user_profile + 1, n_users), end=" ", file=sys.stderr)
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
                transitions_str += "%d\t%d\t%d\t%s\n" % (s1, a, s2, new_count)
                # Negative P(s1 -a-> s1.b), b!= a
                beta = (total_count - args.alpha * count) / (total_count - count)
                # For every s2, sample T(s1, a, s2)
                for s2_link in actions:
                    if (s2_link != a):
                        s2 = get_next_state_id(s1, s2_link)
                        count = exc if s2_link == user_profile + 1 else 1
                        transitions_str += "%d\t%d\t%d\t%s\n" % (s1, a, s2, beta * count if not args.norm else beta * count / total_count)
                        # If buffer overflows, write in the zip file
                        if len(transitions_str) > buffer_size:
                            f.write(bytes(transitions_str.encode("UTF-8")) if args.zip else transitions_str)
                            transitions_str = ""
        transitions_str += "\n"
    f.write(bytes(transitions_str.encode("UTF-8")) if args.zip else transitions_str)
    f.close()

    with open("%s.summary" % output_base, 'w') as f:
        f.write("%d States\n%d Actions (Items)\n%d user profiles\n%d history length\n%d product clustering level\n\n%s" % (n_states, n_items, n_users, args.history, args.nactions, logger.to_string()))

    ###### 6. Summary
    print("\n\n\033[92m-----> End\033[0m")
    print("   Output directory: %s" % output_base)
    # End
