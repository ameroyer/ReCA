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
from utils import *
import tarfile as tar

def init_output_dir(plevel, ulevel, hlength, alpha):
    """
    Initializa the output directory.

    Args:
     * ``plevel`` (*int*): level parameter for the product clustering.
     * ``ulevel`` (*int*): level parameter for the customer clustering.
     * ``hlength`` (*int*): history length.
     * ``alpha`` (*float, optional*): positive probability scaling for the recommended action.

    Returns:
     * ``output_base`` (*str*): base name for output files.
    """
    import shutil
    output_base = "foodmart_u%d_k%d_pl%d_a%.2f" % (get_n_customer_cluster(ulevel), hlength, plevel, alpha)
    output_dir = os.path.join(args.output, "Foodmart%d%d%d" % (get_n_customer_cluster(ulevel), hlength, plevel))
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


def load_data(base_name, plevel, ulevel, hlength, alpha, trfr, sv=False):
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
        output_base = init_output_dir(plevel, ulevel, hlength, alpha)


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
    print('\n'.join("     > %s: %.2f%%" % (k, 100 * float(len(v)) / (len(product_to_cluster) - 1)) for k, v in tmp_clusters.items()))
    actions = sorted(tmp_index.values())
    product_to_cluster[0] = 0 # Empty selection

    # Init states
    print("\n\033[92m-----> [Optional] Export states description\033[0m")
    init_base_writing(len(actions), args.history)
    if sv:
        rv_tmp_indx = {v: k for k, v in tmp_index.items()}
        rv_tmp_indx[0] = str(chr(35))
        with open("%s.states" % output_base, 'w') as f:
            f.write('\n'.join("%f\t%s" % (x, '|'.join(rv_tmp_indx[y] for y in id_to_state(x))) for x in range(get_nstates(len(actions), args.history))))

    ######  Load and Cluster users by profile
    #########################################################################

    customer_to_cluster = np.zeros(line_count(load_datafile(base_name, "customer.csv")), dtype="int")  - 1                                  # Customer ID -> Cluster ID
    tmp_index_u = {}                          # Cluster name -> Cluster ID
    f = load_datafile(base_name, "customer.csv")
    r = csv.reader(f)
    next(r)
    for user in r:
        customerID = int(user[0])
        try:
            clusterID = tmp_index_u[assign_customer_cluster(user, ulevel)]
        except KeyError:
            clusterID = len(tmp_index_u)
            tmp_index_u[assign_customer_cluster(user, ulevel)] = clusterID
        customer_to_cluster[customerID] = clusterID
    f.close()


    ###### Load and Store user sessions
    #########################################################################

    print("\n\033[92m-----> Load user sessions and shop profits \033[0m")
    product_profit = np.zeros(len(actions) + 1, dtype=float)           # Product ID -> profit
    product_profit_nrm = np.zeros(len(actions) + 1, dtype=int)
    user_sessions = {k: defaultdict(lambda: [0] * hlength) for k in tmp_index_u.values()}  # Customer ID -> Session (product list)

    # Load session
    f = load_datafile(base_name, "sales.csv")
    r = csv.reader(f)
    next(r)
    for sale in r:
        product_clusterID = product_to_cluster[int(sale[0])]
        product_profit[product_clusterID] += float(sale[5]) - float(sale[6])
        product_profit_nrm[product_clusterID] += 1
        for _ in range(int(sale[7])):
            user_sessions[customer_to_cluster[int(sale[2])]][int(sale[2])].append(product_clusterID)
    f.close()

    # Summary profit
    product_profit[1:] /= product_profit_nrm[1:]
    print("   %d user sessions\n" % sum(len(x) for x in user_sessions.values()))
    print("   Average profit per product profile")
    print('\n'.join("     > %s: %.2f $" % (k, product_profit[v]) for k, v in tmp_index.items()))

    # Summary profiles
    rv_tmp_indx_u = {v: k for k, v in tmp_index_u.items()}
    print("\n\033[92m-----> Build user profiles (cluster)\033[0m")
    print("   %d User profiles (%d total users)" %(len(user_sessions), sum(len(x) for x in user_sessions.values())))
    print('\n'.join("     > %s: %.2f%%" % (print_customer_cluster(rv_tmp_indx_u[v], ulevel), 100 * float(len(user_sessions[v])) /sum(len(x) for x in user_sessions.values())) for v in user_sessions.keys()))

    # Save product clusters information
    if sv:
        with open("%s.items" % output_base, 'w') as f:
            f.write('\n'.join("%d\t%s\t%d\t%.3f" %(tmp_index[k], k, len(tmp_clusters[k]), product_profit[tmp_index[k]]) for k in sorted(tmp_index.keys(), key=lambda x: tmp_index[x])))

    # Save profiles information
        with open("%s.profiles" % output_base, 'w') as f:
            f.write('\n'.join("%d\t%d\t%.5f\t%s" %(k, rv_tmp_indx_u[k], 100 * float(len(v)) /sum(len(z) for z in user_sessions.values()), print_customer_cluster(rv_tmp_indx_u[k], ulevel)) for k, v in user_sessions.items()))

    # Return values
    return product_to_cluster, customer_to_cluster, user_sessions, product_profit, actions, output_base





#####################################################   M A I N    R O U T I N E  #######
if __name__ == "__main__":

    ###### 0. Set Parameters
    base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description='Extract POMDP transition probabilities from the Foodmart dataset.')
    parser.add_argument('-d', '--data', type=str, default=os.path.join(base_folder, "Data", "Foodmart", "data"), help="Path to data directory or archive.")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(base_folder, "Code", "Models"), help="Path to output directory.")
    parser.add_argument('-pl', '--plevel', type=int, default=4, help="Clustering level for product categorization (0: no lumping to 4:lumping by family). See product classes hierarchy.")
    parser.add_argument('-ul', '--ulevel', type=int, default=0, help="Clustering level for user categorization (0, 1 or 2: 6, or environments).")
    parser.add_argument('-k', '--history', type=int, default=2, help="Length of the history to consider for one state of the MEMDP.")
    parser.add_argument('-t', '--train', type=float, default=0.8, help="Fraction of training data to extract from the database.")
    parser.add_argument('-a', '--alpha', type=float, default=1.4, help="Positive rescaling of transition probabilities matching the recommendation.")
    parser.add_argument('--norm', action='store_true', help="If present, normalize the output transition probabilities. ")
    parser.add_argument('--draw', action='store_true', help="If present, draw the first user MDP model. For debugging purposes")
    parser.add_argument('--zip', action='store_true', help="If present, the transitiosn are output in a compressed file.")
    args = parser.parse_args()

    ###### 0-bis. Check assertions
    assert(args.train >= 0 and args.train <= 1), "Training fraction must be between 0 and 1 (included)"
    assert(args.plevel in [0, 1, 2, 3, 4]), "plevel argument must be in [0, 1, 2, 3, 4]"
    assert(args.ulevel in [0, 1, 2]), "ulevel argument must be in [0, 1, 2]"
    assert(args.alpha >= 1), "alpha argument must be greater than 1"
    assert(args.history > 1), "history length must be strictly greater than 1"
    logger = Logger(sys.stdout)
    sys.stdout = logger


    ###### 1. Load data and create product/user profile
    product_to_cluster, customer_to_cluster, user_sessions, product_profit, actions, output_base = load_data(args.data, args.plevel,  args.ulevel, args.history,args.alpha, args.train, sv=True)
    n_items = len(actions)
    n_states = get_nstates(n_items, args.history)


    ###### 2. Split training and testing database

    print("\n\033[96m-----> Split training and testing database \033[0m")
    test_sessions = [0] * sum(int((1 - args.train) * len(x)) for x in user_sessions.values())
    i = 0
    for u, sessions in user_sessions.items():
        test_users = sorted(sessions, key=lambda k: random())[:int((1 - args.train) * len(sessions))]
        for j, x in enumerate(test_users):
            assert(len(sessions[x]) > args.history), "Empty user session %d" % x
            test_sessions[i] = [x, u, sessions[x]]
            i += 1
            del user_sessions[u][x]

    # Summary
    print("   %d training sessions" % sum(len(x) for x in user_sessions.values()))
    print("   %d test sessions" % len(test_sessions))

    # Save train and test Sessions
    trn_str = '\n'.join("%d\t%d\t%s" % (u,  c, ' '.join('%d %d' % (state_indx(s[i : i + args.history]), s[i + args.history]) for i, x in enumerate(s[:-args.history]))) for u, s in sessions.items() for c, sessions in user_sessions.items())
    tst_str = '\n'.join("%d\t%d\t%s" % (u,  c, ' '.join('%d %d' % (state_indx(s[i : i + args.history]), s[i + args.history]) for i, x in enumerate(s[:-args.history]))) for u, c, s in test_sessions)

    f = gzip.open("%s.train.gz" % output_base, 'w') if args.zip else open("%s.train" % output_base, 'w')
    f.write(bytes(trn_str.encode("UTF-8")) if args.zip else trn_str)
    f.close()

    f = open("%s.test" % output_base, 'w')
    f.write(tst_str)
    f.close()


    ###### 3. Init probability counts [Smoothing] and rewards
    print("\n\033[91m-----> Reward function\033[0m")
    print("   %d States in the database" % n_states)
    print("   %d Actions in the database" % n_items)
    with open("%s.rewards" % output_base, 'w') as f:
        for item in actions:
            f.write("%d\t%.5f\n" % (item, product_profit[item]))
    del product_profit



    ###### 4. Compute joint state occurrences from the database
    print("\n\033[91m-----> Probability inference\033[0m")
    f = gzip.open("%s.transitions.gz" % output_base, 'w') if args.zip else open("%s.transitions" % output_base, 'w')
    buffer_size = 2**31 -1
    transitions_str = ""
    epsilon = 0.5   # Smoothing
    max_upscale = 0.95 # Max value of the probability after upscale by alpha
    for user_profile, aux in user_sessions.items():
        print("\n   > Profile %d / %d: \n" % (user_profile + 1, len(user_sessions)), file=sys.stderr)
        sys.stderr.flush()
        # Count
        js_count = np.zeros((n_states, n_items), dtype=int) # js[s1, a] = P(s1.a | s1; cluster)
        for _, session in aux.items():
            s1 = 0
            for item in session[args.history + 1:]:
                s2 = get_next_state_id(s1, item)
                js_count[s1, item - 1] += 1
                s1 = s2

        # Estimate and normalize probabilities
        for s1, s1_counts in enumerate(js_count[:, :]):
            sys.stderr.write("      state: %d / %d   \r" % (s1 + 1, n_states))
            sys.stderr.flush()
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

    ###### 5. Summary
    print("\n\n\033[92m-----> End\033[0m")
    print("   All outputs are in %s" % output_base)
    with open("%s.summary" % output_base, 'w') as f:
        f.write("%d States\n%d Actions (Items)\n%d user profiles\n%d history length\n%f alpha\n%d product clustering level\n\n%s" % (n_states, n_items, len(user_sessions), args.history, args.alpha, args.plevel, logger.to_string()))
    print()
    # End
