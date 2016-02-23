#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a synthetic "dummy" POMDP model based on Foodmart.
"""
__author__ = "Amelie Royer"
__email__ = "amelie.royer@ist.ac.at"


import sys, os
import csv
import argparse
import numpy as np
from collections import defaultdict
from random import randint
from utils import *



def init_output_dir(plevel, ulevel, hlength):
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
    output_base = "random_u%d_k%d_pl%d" % (ulevel, hlength, plevel)
    output_dir = os.path.join(args.output, "Random%d%d%d" % (ulevel, hlength, plevel))
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return os.path.join(output_dir, output_base)


def load_data(base_name, plevel, ulevel, hlength, trfr, sv=False):
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


    ###### Load and Cluster items
    #########################################################################
    
    print "\n\033[92m-----> Load and Cluster products\033[0m"
    tmp_index = {}             # Cluster name -> Cluster ID

    # Load product list
    if plevel == 0:
        with open(os.path.join(base_name, "product.csv"), 'r') as f:
            r = csv.reader(f)
            r.next()
            for product in r:
                try:
                    tmp_index[product[3]]
                except KeyError:
                    tmp_index[product[3]] = len(tmp_index) + 1
    else:
        # Load product categories
        product_classes = {}
        with open(os.path.join(base_name, "product_class.csv"), 'r') as f:
            r = csv.reader(f)
            r.next()
            for categories in r:
                product_classes[int(categories[0])] = categories[plevel]

        # Cluster products
        with open(os.path.join(base_name, "product.csv"), 'r') as f:
            r = csv.reader(f)
            r.next()
            for product in r:
                try:
                    tmp_index[product_classes[int(product[0])]]
                except KeyError:
                    tmp_index[product_classes[int(product[0])]] = len(tmp_index) + 1

    # Print summary
    actions = sorted(tmp_index.values())
    n_items = len(actions)
    # Init output folder
    if sv:
        output_base = init_output_dir(plevel, n_items, hlength)

    init_base_writing(len(actions), args.history)



    ######  Load and Cluster users by profile
    #########################################################################

    customer_to_cluster = np.zeros(line_count(os.path.join(base_name, "customer.csv")), dtype="int")  - 1                                  # Customer ID -> Cluster ID
    tmp_index_u = {}                          # Cluster name -> Cluster ID
    with open(os.path.join(base_name, "customer.csv"), 'r') as f:
        r = csv.reader(f)
        r.next()
        for user in r:
            customerID = int(user[0])
            try:
                clusterID = tmp_index_u[assign_customer_cluster(user)]
            except KeyError:
                clusterID = len(tmp_index_u)
                tmp_index_u[assign_customer_cluster(user)] = clusterID
            customer_to_cluster[customerID] = clusterID
    n_users = len(tmp_index_u)

    # Return values
    return actions, n_items, get_nstates(n_items, hlength), n_users, output_base





#####################################################   M A I N    R O U T I N E  #######
if __name__ == "__main__":

    ###### 0. Set Parameters

    parser = argparse.ArgumentParser(description='Extract POMDP transition probabilities from the Foodmart dataset.')
    parser.add_argument('-d', '--data', type=str, default="/home/amelie/Rotations/ChatterjeeRotation/Data/Foodmart/data", help="Path to data directory.")
    parser.add_argument('-o', '--output', type=str, default="/home/amelie/Rotations/ChatterjeeRotation/Code/Models", help="Path to output directory.")
    parser.add_argument('-pl', '--plevel', type=int, default=4, help="Clustering level for product categorization (0: no lumping to 4:lumping by family). See product classes hierarchy.")
    parser.add_argument('-ul', '--ulevel', type=int, default=0, help="Clustering level for user categorization.")
    parser.add_argument('-k', '--history', type=int, default=2, help="Length of the history to consider for one state of the MEMDP.")
    parser.add_argument('-t', '--train', type=float, default=0.8, help="Fraction of training data to extract from the database.")
    parser.add_argument('--ordered', action='store_true', help="If present, the states of the MEMDP are ordered product sequences. TODO.")
    parser.add_argument('--norm', action='store_true', help="If present, normalize the output transition probabilities.")
    parser.add_argument('--draw', action='store_true', help="If present, draw the first user MDP model.")
    args = parser.parse_args()


    ###### 1. Check assertions

    assert(args.train >= 0 and args.train <= 1), "Training fraction must be between 0 and 1 (included)"
    assert(args.plevel in [0, 1, 2, 3, 4]), "plevel argument must be in [0, 1, 2, 3, 4]"
    assert(args.ulevel == 0), "ulevel  must be in 0"
    assert(args.history > 1), "history length must be strictly greater than 1"
    logger = Logger(sys.stdout)
    sys.stdout = logger
    limit = 5000
    n_test = 2000
    

    ###### 1. Load data and create product/user profile
    actions, n_items, n_states, n_users, output_base = load_data(args.data, args.plevel,  args.ulevel, args.history, args.train, sv=True)
    n_users = n_items
    exc = 4 * (n_users -1)

    #### 2. Write dummy files
    with open("%s.items" % output_base, "w") as f:
        f.write("\n".join("Item %d" % i for i in xrange(n_items)))

    with open("%s.profiles" % output_base, "w") as f:
        f.write("\n".join("%d\t1\t1" % i for i in xrange(n_users)))


    ##### Create dummy test sessions
    with open("%s.test" % output_base, 'w') as f:
        for user in xrange(n_test):
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
    

    ###### 2. Set rewards
    with open("%s.rewards" % output_base, 'w') as f:
        for s1 in xrange(n_states):
            for item in actions:
                f.write("%d\t%d\t%d\t%.5f\n" % (s1, item, get_next_state_id(s1, item), 1))


    ###### 3. Assign random transition probabilities
    print "\n\033[91m-----> Probability inference\033[0m"
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

                    # For every s2, sample T(s1, a, s2)
                    for link in actions:
                        count = exc if link == user_profile + 1 else 1
                        s2 = get_next_state_id(s1, link)
                        f.write("%d\t%d\t%d\t%s\n" % (s1, a, s2, count))
            f.write("\n")




    print "\n\n\033[92m-----> End\033[0m"
    print "   All outputs are in %s" % output_base
    with open("%s.summary" % output_base, 'w') as f:
        f.write("%d States\n%d Actions (Items)\n%d user profiles\n%d history length\n%d product clustering level\n\n%s" % (n_states, n_items, n_users, args.history, args.plevel, logger.to_string()))
    print
    # End
