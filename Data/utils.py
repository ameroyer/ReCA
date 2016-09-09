#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitary functions for state indices handling for recommendation datasets.
"""
__author__ = "Amelie Royer"
__email__ = "amelie.royer@ist.ac.at"


import sys
import numpy as np
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

# Python 2 - 3 compatibility
if sys.version_info[0] == 3:
    xrange = range

try:
    dict.iteritems
except AttributeError:
    # Python 3
    def itervalues(d):
        return iter(d.values())
    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    def itervalues(d):
        return d.itervalues()
    def iteritems(d):
        return d.iteritems()
    
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

    def flush(self):
        self.logfile.flush()


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
    return (n_items ** (hlength + 1) - 1) // (n_items - 1)


def assign_customer_cluster(user, ulevel):
    from random import randint
    """
    Assigns user profile given customer data from the foodmart dataset.

    Args:
     * ``user`` (*list*): data for one customer extracted from the foodmart dataset.

    Returns:
     * ``cluster`` (*int*): Cluster ID
    """
    # Parse
    gender = int(user[19] == 'F')
    age_category = ((1997 - int(user[16].split('-', 1)[0])) // 10) // 2
    n_children = int(user[20])
    n_children_home = min(int(user[21]), 3) #clip to 3
    income = int(''.join([c for c in user[18].split('-')[-1] if c.isdigit()]))
    income = 0 if income <= 50 else 1 if income <= 90 else 2
    card = 0 if user[24] == 'Bronze' else 1 if user[24] == 'Normal' else 2
    status = int(user[17] == 'M')
    house = int(user[26] == 'Y')
    ncars = int(user[27])
    #return n_children_home
    # Ulevel 0, distinguish on 5 age category and gender
    if ulevel == 0:
        return gender + 2 * age_category
    # Ulevel 1, gender, number of children at home and income
    elif ulevel == 1:
        return gender + 2 * (n_children_home + 4 * income)
    # Ulevel 2: number of children, income, marital status and house
    else:
        return n_children_home + 4 * (income + 3 * (status + 2 * (house + 2 * card)))

def print_customer_cluster(cluster, ulevel):
    """
    Returns the string representation for a cluster ID.

    Args:
     * ``cluster`` (*int*): Cluster ID

    Returns:
     * ``cluster_str`` (*str*): String representation of the cluster ID
    """
    if ulevel == 0:
        return "%s in the %d+ years old category" %("Female" if cluster % 2 else "Male", 20 * (cluster // 2))
    elif ulevel == 1:
        return "%s, %d children at home, %d tier income" %("Female" if cluster % 2 else "Male", (cluster // 2) % 4, cluster // 8)
    elif ulevel == 2:
        return "%s, %s, %d children at home, %d tier income. %d card" %("Married" if ((cluster // 12) % 2) else "Single", "house" if cluster // 24 else "house", cluster % 4, (cluster // 4) % 3, cluster // 24)
        


def get_n_customer_cluster(ulevel):
    """
    Returns the number of user profiles for the given clustering parameter.

    Args:
     * ``ulevel`` (*int*): user clustering parameter.

    Returns:
     * ``n_clusters`` (*int*): Number of clusters that will be created.
    """
    if ulevel == 0:
        return 10
    elif ulevel == 1:
        return 24
    elif ulevel == 2:
        return 89
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

class ChunkedWriter(object):
    """
    Write chunks of data in a given file. Work around of the overflow bug when writing
    with gzip in Python 2.7
    """
    def __init__(self, file, chunksize=65536):
        self.file = file
        self.chunksize = chunksize

    def write(self, mdata):
        for i in range(0, len(mdata), self.chunksize):
            self.file.write(bytes(mdata[i:i+self.chunksize]))
