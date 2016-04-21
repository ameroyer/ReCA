#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitary functions for state indices handling for recommendation datasets.
"""
__author__ = "Amelie Royer"
__email__ = "amelie.royer@ist.ac.at"


import mmap
import numpy as np
from StringIO import StringIO


class Logger:
    """
    Capture print statments and write them to a log file while printing on the screen.
    """
    def __init__(self, stdout):
        self.stdout = stdout
        self.logfile = StringIO()

    def write(self, text):
        self.stdout.write(text)
        self.logfile.write(text)
        self.logfile.flush()

    def close(self):
        self.stdout.close()
        self.logfile.close()

    def to_string(self):
        return self.logfile.getvalue()


def line_count(f):
    """
    Returns the number of lines in the given file f.

    Args:
     * ``f`` (*File*): file.

    Returns:
     * ``nlines`` (*int*): number of lines in f.
    """
    nlines= len(f.readlines())
    f.close()
    return nlines


def get_nstates(n_items, hlength):
    """
    Returns the number of states in the MDP

    Args:
     * ``n_items`` (*int*): number of items/products available to the user.
     * ``hlength`` (*int*): history length to consider.

    Returns:
     * ``n_states`` (*str*): number of states in the corresponding MDPs.
    """
    return (n_items ** (hlength + 1) - 1) / (n_items - 1)


def assign_customer_cluster(user):
    from random import randint
    """
    Assigns user profile given customer data from the foodmart dataset.

    Args:
     * ``user`` (*list*): data for one customer extracted from the foodmart dataset.

    Returns:
     * ``cluster`` (*int*): Cluster ID
    """
    gender = int(user[19] == 'F')
    age_category = ((1997 - int(user[16].split('-', 1)[0])) / 10) / 3
    return gender * 10 + age_category


def print_customer_cluster(cluster):
    """
    Returns the string representation for a cluster ID.

    Args:
     * ``cluster`` (*int*): Cluster ID

    Returns:
     * ``cluster_str`` (*str*): String representation of the cluster ID
    """

    return "%s in the %d+ years old category" %("Female" if cluster / 10 else "Male", 30 * (cluster % 10))


def get_n_customer_cluster(ulevel):
    """
    Returns the number of user profiles for the given clustering parameter.

    Args:
     * ``ulevel`` (*int*): user clustering parameter.

    Returns:
     * ``n_clusters`` (*int*): Number of clusters that will be created.
    """
    if ulevel == 0:
        return 6
    else:
        print >> sys.stderr, "Unknown ulevel = %d option. Exit." % ulevel
        raise SystemExit


def assign_product_cluster(product, product_classes, plevel):
    """
    Assigns product profile given product data from the foodmart dataset.

    Args:
     * ``product`` (*list*): data for one product extracted from the foodmart dataset.
     * ``product_classes`` (*list*): data extracted from the product_class.csv in the dataset.
     * ``plevel`` (*int*): level of clustering in the database (0: fine grained to 4: rough)

    Returns:
     * ``cluster`` (*str*): Cluster ID
    """

    if plevel == 0:   # Product
        return product[3]
    elif plevel == 1: # Product class/subcategory
        return product_classes[int(product[0])][0]
    elif plevel == 2: # Product category
        return product_classes[int(product[0])][1]
    elif plevel == 3: # Product department
        return product_classes[int(product[0])][2]
    elif plevel == 4: # Product family
        return product_classes[int(product[0])][3]


pows = []
"""
``pows`` contains the precomputed exponents for the use of base (n_items) in decreasing order, used for state to index encryption.
"""

acpows = []
"""
``acpows`` contains the cumulative sum of  the values in ``pows``.
"""

n_states = 0
"""
``n_states`` contains the number of states in the model.
"""

def init_base_writing(n_items, hlength):
    """
    Inits the global variables ``pows`` that contains the precomputed exponents for the base (n_items + 1), and ``n_states`` that indicates the number of state in the system..

    Args:
     * ``n_items``: number of items in the task (not counting the empty selection).
     * ``hlength`` (*int*): history length.
    """
    global pows, acpows, n_states
    pows = [1] * hlength
    acpows = [1] * hlength
    for i in xrange(1, hlength):
        pows[hlength - 1 - i] = pows[hlength - i] * n_items
        acpows[hlength - 1 - i] = acpows[hlength - i] + pows[hlength - 1 - i]
    n_states = get_nstates(n_items, hlength)


def state_indx(item_list):
    """
    Returns the index for a state in the model.

    Args:
     * ``item_list`` (*int list*): input state; ordered sequence of items where the first item represented the oldest choice.
    """
    global pows, n_states
    id = sum(x * p for x, p in zip(item_list, pows))
    assert(id < n_states), "out-of-bound state index: %d" % id
    return id


def get_next_state_id(s, i):
    """
    Given the current state and the next item chosen by the user, returns the correpsonding next state.

    Args:
     * ``s`` (*int*): current state index.
     * ``i`` (*int*): next user choice.
    """
    global pows, acpows
    cd, rst = divmod(s - acpows[0], pows[0])
    if cd < -1:
        return (rst - pows[0]) * pows[-2] + (i - 1) + acpows[0]
    else:
        return rst * pows[-2] + (i - 1) + acpows[0]


def id_to_state(s):
    """
    Returns the sequence of items corresponding to a state index.

    Args:
     * ``s`` (*int*): input state index

    Returns
     * ``item_list`` (*int list*): ordered sequence of items corresponding to ``s``, where the first item represented the oldest choice.
    """
    global pows, acpows
    real = s
    output = [0] * len(pows)
    i = 0
    while real > pows[-2]:
        cd, rst = divmod(real - acpows[i], pows[i])
        if cd < -1:
            rst -= pows[i]
        else:
            output[i] = cd + 1
        real = rst + acpows[i + 1]
        i += 1
    output[-1] = real
    return output
