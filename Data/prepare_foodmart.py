from __future__ import print_function
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess Foodmart data to extract the POMDP model.
"""
__author__ = "Amelie Royer"
__email__ = "amelie.royer@ist.ac.at"


import sys, os
import csv
import gzip
import argparse
import numpy as np
from collections import defaultdict
from random import random
from utils import Logger, line_count, iteritems, itervalues, get_nstates, assign_product_cluster, init_base_writing, get_next_state_id, state_indx, id_to_state
import tarfile as tar

# Python 2 - 3 compatibility
if sys.version_info[0] == 3:
    xrange = range

def init_output_dir(plevel, ulevel, hlength):
    """
    Initializa the output directory.

    Args:
     * ``plevel`` (*int*): level parameter for the product clustering.
     * ``ulevel`` (*int*): level parameter for the customer clustering.
     * ``hlength`` (*int*): history length.

    Returns:
     * ``output_base`` (*str*): base name for output files.
    """
    import shutil
    output_base = "foodmart_u%d_k%d_pl%d" % (ulevel, hlength, plevel)
    output_dir = os.path.join(args.output, "Foodmart%d%d%d" % (ulevel, hlength, plevel))
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return os.path.join(output_dir, output_base)


def load_datafile(base_name, target):
    """
    Load the given ``target`` file from the given input (data folder or archive).
    """
    try:
        if tar.is_tarfile(base_name):
            t = tar.open(base_name)
            res = t.extractfile(os.path.join("Foodmart", "data", target))
            if res is not None:
                return t.extractfile(os.path.join("Foodmart", "data", target))
            else:
                raise IOError
    except IOError:
        try:
            return open(os.path.join(base_name, target), "r")
        except IOError:
            print("File %s not found" % base_name, file=sys.stderr)
            raise SystemExit


def load_data(base_name, plevel, ulevel, hlength, sv=False):
    """
    Load and pre-format the Foodmart data (products, customers and user sessions).

    Args:
     * ``base_name`` (*str*): path to the main data folder.
     * ``plevel`` (*int*): level parameter for the product clustering.
     * ``hlength`` (*int*): history length.
     * ``sv`` (*bool, optional*): if True, store the computed informations in .items, .profiles, .train and .test

    Returns:
     * ``product_to_cluster`` (*ndarray*): maps a productID to a clusterID. Note 0 -> -1 is the empty selection.
     * ``customer_to_cluster`` (*ndarray*): maps a customerID to a clusterID.
    """

    # Init output folder
    if sv:
        output_base = init_output_dir(plevel, ulevel, hlength)


    ###### Load and Cluster items
    #########################################################################

    print("\n\033[92m-----> Load and Cluster products\033[0m")
    product_to_cluster = np.zeros(line_count(load_datafile(base_name, "product.csv")) + 1, dtype=int)      # Product ID -> Cluster ID
    tmp_index = {}                          # Cluster name -> Cluster ID
    tmp_clusters = defaultdict(lambda: [])  # Cluster name -> Product ID list

    # Load product list
    if plevel == 0:
        f = load_datafile(base_name, "product.csv")
        r = csv.reader(f)
        next(r)
        for product in r:
            tmp_clusters[product[3]].append(int(product[1]))
            try:
                product_to_cluster[int(product[1])] = tmp_index[product[3]]
            except KeyError:
                tmp_index[product[3]] = len(tmp_index) + 1
                product_to_cluster[int(product[1])] = tmp_index[product[3]]
        f.close()

    else:
        # Load product categories
        product_classes = {}
        f = load_datafile(base_name, "product_class.csv")
        r = csv.reader(f)
        next(r)
        for categories in r:
            product_classes[int(categories[0])] = categories[plevel]
        f.close()

        # Cluster products
        f = load_datafile(base_name, "product.csv")
        r = csv.reader(f)
        next(r)
        for product in r:
            try:
                product_to_cluster[int(product[1])] = tmp_index[product_classes[int(product[0])]]
            except KeyError:
                tmp_index[product_classes[int(product[0])]] = len(tmp_index) + 1
                product_to_cluster[int(product[1])] = tmp_index[product_classes[int(product[0])]]
            tmp_clusters[product_classes[int(product[0])]].append(int(product[1]))
        f.close()

    # Print summary
    print("   %d product profiles (%d products)" % (len(tmp_index), (len(product_to_cluster) - 1)))
    print('\n'.join("     > %s: %.2f%%" % (k, 100 * float(len(v)) / (len(product_to_cluster) - 1)) for k, v in iteritems(tmp_clusters)))
    actions = sorted(itervalues(tmp_index))
    product_to_cluster[0] = 0 # Empty selection

    # Init states
    print("\n\033[92m-----> [Optional] Export states description\033[0m")
    init_base_writing(len(actions), args.history)
    if sv:
        rv_tmp_indx = {v: k for k, v in tmp_index.items()}
        rv_tmp_indx[0] = str(chr(35))
        with open("%s.states" % output_base, 'w') as f:
            f.write('\n'.join("%f\t%s" % (x, '|'.join(rv_tmp_indx[y] for y in id_to_state(x))) for x in xrange(get_nstates(len(actions), args.history))))

    ###### Load and Store user sessions
    #########################################################################

    print("\n\033[92m-----> Load user sessions and shop profits \033[0m")
    user_sessions = defaultdict(lambda: [0] * hlength)

    # Load session
    f = load_datafile(base_name, "sales.csv")
    r = csv.reader(f)
    next(r)
    for sale in r:
        product_clusterID = product_to_cluster[int(sale[0])]
        user_sessions[int(sale[2])].append(product_clusterID)
    f.close()

    # Save product clusters information
    if sv:
        with open("%s.items" % output_base, 'w') as f:
            f.write('\n'.join("%d\t%s\t%d" %(tmp_index[k], k, len(tmp_clusters[k])) for k in sorted(tmp_index.keys(), key=lambda x: tmp_index[x])))

    # Return values
    return product_to_cluster, user_sessions, actions, output_base


def estimate_probability(seqs, n_states, n_items, epsilon=0.5, normalized=True):
    """
    Estimate the transition probabilities from a list of item sequences (<-> bigram model on the states = past histories).

    Args:
     * ``seqs`` (*dict: user -> sequence*): state sequences.
     * ``n_states`` (*int*): number of states in the model.
     * ``n_items`` (*int*): number of items/actions in the model.
     * ``epsilon`` (*float, optional*): smoothing parameter. Defaults to 0.5.

    Returns:
     * ``js_count`` (*n_states x n_items ndarray*): estimated probability transitions.

    """
    ### Count co-occurrences
    js_count = np.zeros((n_states, n_items), dtype=float) # js[s1, a] = P(s1.a | s1; cluster)
    for _, session in iteritems(seqs):
        s1 = 0
        for item in session[args.history:]:
            s2 = get_next_state_id(s1, item)
            js_count[s1, item - 1] += 1
            s1 = s2

    ### Normalize
    if normalized:
        for s1, s1_counts in enumerate(js_count[:, :]):
            print("      state: %d / %d   \r" % (s1 + 1, n_states), file=sys.stderr, end=' ')
            nrm = np.sum(s1_counts) + n_items * epsilon
            js_count[s1, :] = (s1_counts + epsilon) / nrm

    ### Return
    return js_count

def compute_perplexity(seq, js_count):
    """
    Compute the perplexity of the given sequence.

    Args:
     * ``seq``: state sequence.
     * ``js_count`` (*n_states x n_items ndarray*): estimated probability transitions.

    Returns:
     * ``perp``: information-theory perplexity.
    """
    s1 = 0; perp = 0
    for item in seq[args.history:]:
        perp += np.log2(js_count[s1, item - 1])
        s2 = get_next_state_id(s1, item)
        s1 = s2
    perp = 2**(- perp / (len(seq) - args.history))
    return perp



#####################################################   M A I N    R O U T I N E  #######
if __name__ == "__main__":

    ###### 0. Set Parameters
    base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description='Extract POMDP transition probabilities from the Foodmart dataset.')
    parser.add_argument('-in', '--data', type=str, default=os.path.join(base_folder, "Data", "Foodmart", "data"), help="Path to data directory or archive.")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(base_folder, "Code", "Models"), help="Path to output directory.")
    parser.add_argument('-u', '--ulevel', type=int, default=3, help="Number of environments to create.")
    parser.add_argument('-p', '--plevel', type=int, default=4, help="Clustering level for product categorization (0: no lumping to 4:lumping by family). See product classes hierarchy.")
    parser.add_argument('-k', '--history', type=int, default=2, help="Length of the history to consider for one state of the MEMDP.")
    parser.add_argument('-D', '--D', type=int, default=500, help="Number of sequences to keep to estimate each environment's transition function.")
    parser.add_argument('-a', '--alpha', type=float, default=1.1, help="Positive rescaling of transition probabilities matching the recommendation.")
    parser.add_argument('-t', '--test', type=int, default=2000, help="Number of test sequences to generate.")
    parser.add_argument('--norm', action='store_true', help="If present, normalize the output transition probabilities. ")
    parser.add_argument('--zip', action='store_true', help="If present, the transitiosn are output in a compressed file.")
    args = parser.parse_args()

    ###### 0-bis. Check assertions
    assert(args.plevel in [0, 1, 2, 3, 4]), "product cluster argument must be in [0, 1, 2, 3, 4]"
    assert(args.ulevel >= 0), "number of user clusters must be positive"
    assert(args.alpha >= 1), "alpha argument must be greater than 1"
    assert(args.history > 1), "history length must be strictly greater than 1"
    logger = Logger(sys.stdout)
    sys.stdout = logger
    epsilon=0.5 # smoothing parameter

    ###### 1. Load data and create product/user profile
    product_to_cluster, user_sessions, actions, output_base = load_data(args.data, args.plevel, args.ulevel, args.history, sv=True)
    n_items = len(actions)
    n_states = get_nstates(n_items, args.history)
    assert(args.D * args.ulevel < len(user_sessions)), "not enough data to fit choice of parameters D and n"

    ###### 2. Store rewards
    print("\n\033[91m-----> Reward function\033[0m")
    print("   %d States in the database" % n_states)
    print("   %d Actions in the database" % n_items)
    with open("%s.rewards" % output_base, 'w') as f:
        for item in actions:
            f.write("%d\t%s\n" % (item, 1.0))

    ###### 3. Cluster sequences with a perplexity criterion
    print("\n\033[91m-----> Clustering\033[0m")
    clusters = defaultdict(lambda: {})
    f_prop = open("%s.profiles" % output_base, 'w')
    alternate = 1
    seqs = {k:v for k, v in iteritems(user_sessions) if len(v) > (args.history + 20)} # to ensure reliable perplexity
    #seqs = dict(user_sessions)
    while len(clusters) < args.ulevel:
        # estimate probability over all sequences still available
        js_count = estimate_probability(seqs, n_states, n_items, epsilon)

        # estimate perplexity and sort sequences in decreasing order
        aux = [(user, seq, compute_perplexity(seq, js_count)) for user, seq in iteritems(seqs)]
        aux = sorted(aux, key=lambda x: - x[2])

        # form a cluster out of the sequences with the highest perplexity
        cluster_id = len(clusters)
        mean_perp = 0
        browse = aux[:args.D] if alternate else aux[-args.D:]
        for user, seq, p in browse:
            clusters[cluster_id][user] = seq
            del seqs[user]
            mean_perp += p
        alternate = 1 - alternate
        mean_perp /= len(clusters[cluster_id])
        f_prop.write("%d\t%d\t%d\t%s\n" % (cluster_id, cluster_id, len(clusters[cluster_id]), mean_perp))
        print("   Create cluster %d with mean perplexity %.3f" % (cluster_id, mean_perp))
    del seqs

    ###### 4. Compute transition probabilities per cluster
    print("\n\033[91m-----> Probability inference\033[0m")
    f = gzip.open("%s.transitions.gz" % output_base, 'w') if args.zip else open("%s.transitions" % output_base, 'w')
    buffer_size = 2**31 -1
    max_upscale = 0.95 # prevent overflow when multiplying by alpha

    ### Write MDP transition probabilities
    transitions_str = ""
    js_count = estimate_probability(user_sessions, n_states, n_items, epsilon, normalized=False)
    # For fixed s1
    print("   > MDP:", file=sys.stderr)
    for s1, s1_counts in enumerate(js_count[:, :]):
        print("      state: %d / %d   \r" % (s1 + 1, n_states), file=sys.stderr, end=' ')
        nrm = np.sum(s1_counts) + n_items * epsilon
        # For fixed a
        for a in actions:
            # Positive (s1, a, s1.a)
            s2 = get_next_state_id(s1, a)
            count = s1_counts[a - 1] + epsilon
            new_count = min(args.alpha * count, max_upscale * nrm) if not args.norm else min(max_upscale, args.alpha * count / nrm)
            assert (new_count < nrm if not args.norm else new_count < 1), "AssertionError: Probabilities out of range."
            transitions_str += "%d\t%d\t%d\t%s\n" % (s1, a, s2, new_count)
            # Negative (s1, b, s1.a)
            beta = float(nrm - min(args.alpha * count, max_upscale * nrm)) / (nrm - count)
            for s2_link, s2_count in enumerate(s1_counts):
                if s2_link != a - 1:
                    s2 = get_next_state_id(s1, s2_link + 1)
                    transitions_str += "%d\t%d\t%d\t%s\n" % (s1, a, s2, beta * (s2_count + epsilon) if not args.norm else beta * (s2_count + epsilon) / nrm)
            # If buffer overflow, write in file
            if len(transitions_str) > buffer_size:
                f.write(bytes(transitions_str.encode("UTF-8")) if args.zip else transitions_str)
                transitions_str = ""
    # Environment change
    transitions_str += "\n"
    f.write(bytes(transitions_str.encode("UTF-8")) if args.zip else transitions_str)

    ### Write each environment transition probabilties and test sequences
    transitions_str = ""
    nt = int(args.test / len(clusters))
    f_test = open("%s.test" % output_base, 'w')
    for user_profile, aux in iteritems(clusters):
        print("\n   > Profile %d / %d:" % (user_profile + 1, len(clusters)), file=sys.stderr)
        # Count
        js_count = estimate_probability(aux, n_states, n_items, epsilon, normalized=False)

        # Generate test sequences
        for n in xrange(nt):
            tst_str = "%d\t%d\t" % (n, user_profile)
            s1 = 0
            l = np.random.randint(10, 100)
            for i in xrange(l):
                item = np.random.choice(actions, p=(js_count[s1, :] + epsilon) / (np.sum(js_count[s1, :]) + len(actions) * epsilon))
                tst_str += "%d %d " % (s1, item)
                s1 = get_next_state_id(s1, item)
            f_test.write("%s\n" % tst_str)

        # Estimate and normalize probabilities
        # For fixed s1
        for s1, s1_counts in enumerate(js_count[:, :]):
            print("      state: %d / %d   \r" % (s1 + 1, n_states), file=sys.stderr, end=' ')
            nrm = np.sum(s1_counts) + len(actions) * epsilon
            # For fixed a
            for a in actions:
                # Positive (s1, a, s1.a)
                s2 = get_next_state_id(s1, a)
                count = s1_counts[a - 1] + epsilon
                new_count = min(args.alpha * count, max_upscale * nrm) if not args.norm else min(max_upscale, args.alpha * count / nrm)
                assert (new_count < nrm if not args.norm else new_count < 1), "AssertionError: Probabilities out of range."
                transitions_str += "%d\t%d\t%d\t%s\n" % (s1, a, s2, new_count)
                # Negative (s1, b, s1.a)
                beta = float(nrm - min(args.alpha * count, max_upscale * nrm)) / (nrm - count)
                for s2_link, s2_count in enumerate(s1_counts):
                    if s2_link != a - 1:
                        s2 = get_next_state_id(s1, s2_link + 1)
                        transitions_str += "%d\t%d\t%d\t%s\n" % (s1, a, s2, beta * (s2_count + epsilon) if not args.norm else beta * (s2_count + epsilon) / nrm)
                # If buffer overflow, write in file
                if len(transitions_str) > buffer_size:
                    f.write(bytes(transitions_str.encode("UTF-8")) if args.zip else transitions_str)
                    transitions_str = ""
        # Environment change
        transitions_str += "\n"
    f.write(bytes(transitions_str.encode("UTF-8")) if args.zip else transitions_str)
    f.close()
    f_test.close()

    ###### 5. Summary
    print("\n\n\033[92m-----> End\033[0m")
    print("   All outputs are in %s" % output_base)
    with open("%s.summary" % output_base, 'w') as f:
        f.write("%d States\n%d Actions (Items)\n%d user profiles\n%d history length\n%f alpha\n%d product clustering level\n\n%s" % (n_states, n_items, len(clusters), args.history, args.alpha, args.plevel, logger.to_string()))
    print()
    # End
