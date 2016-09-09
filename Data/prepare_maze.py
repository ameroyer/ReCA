from __future__ import print_function
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a maze model MEMDP from a given topology or with random environment initialization.
"""
__author__ = 'mchmelik, aroyer'


import os
import sys
import argparse
import numpy as np
from shutil import rmtree
from utils import iteritems

if sys.version_info[0] == 3:
    xrange = range

def isWall(i,j,direction):
    """
    Return True iff maze(i, j, direction) leads to a wall by going forward).
    """
    if(direction.__eq__('N')):
        return (maze[i-1][j] == '1')
    if(direction.__eq__('E')):
        return (maze[i][j+1] == '1')
    if(direction.__eq__('W')):
        return (maze[i][j-1] == '1')
    if(direction.__eq__('S')):
        return (maze[i+1][j] == '1')

left = {'N':'W','W':'S','S':'E','E':'N'}
def turnLeft(orient):
    """
    Return the orientation to the left of ``orient``.
    """
    global left
    return left[orient]

right = {v: k for (k, v) in iteritems(left)}
def turnRight(orient):
    """
    Returns the orientation to the right of ``orient``.
    """
    global right
    return right[orient]

def mazeBoundaries(maze):
    """
    Returns the reachable boundaries of the maze.
    """
    width, height = len(maze), len(maze[0])
    # Find min x
    for x in xrange(width):
      if not all(z == '1' for z in maze[x]):
        break
    min_x = x
    # Find max x
    for x in xrange(width - 1, -1, -1):
      if not all(z == '1' for z in maze[x]):
        break
    max_x = x

    # Find min y
    for y in xrange(height):
      if not all(maze[x][y] == '1' for x in xrange(min_x, max_x + 1)):
        break
    min_y = y
    # Find max y
    for y in xrange(height - 1, -1, -1):
      if not all(maze[x][y] == '1' for x in xrange(min_x, max_x + 1)):
        break
    max_y = y
    return min_x, max_x, min_y, max_y


if __name__ == "__main__":
    ###### 0. Parameters
    base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="Generate a maze model MEMDP from a given topology or with random environment initialization.")
    parser.add_argument("-i",  "--fin", type=str, help="If given, load the mazes from a file (takes precedence over the other parameters.")
    parser.add_argument("-n", "--size", type=int, default=5, help="size of the maze")
    parser.add_argument("-s", "--init", default=1, type=int, help="number of initial states per maze")
    parser.add_argument("-t", "--trap", default=0, type=int, help="number of trap states per maze")
    parser.add_argument("-w", "--wall", default=0, type=int, help="number of walls per maze")
    parser.add_argument("-g", "--goal", default=1, type=int, help="number of goal states per maze")
    parser.add_argument("-e", "--env", default=1, type=int, help="number of environments to generate for")
    parser.add_argument("-wf", "--wall_failure", default=0.05, type=float, help="Probability of failure when going forward at a wall")
    parser.add_argument("--rdf", action='store_true', help="each environment has randomized failure rates")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Code", "Models"), help="Path to output directory.")
    args = parser.parse_args()

    # Hyperparameters
    actions = ['F','L','R']
    failures = [0.2, 0.1, 0.1] # Probability of staying still after action forward, left and right respectively.
    wall_failure = args.wall_failure # Probability of being trapped when going forward towards a wall
    goal_reward = 1.0
    min_x, max_x, min_y, max_y = sys.maxsize, 0, sys.maxsize, 0
    changeMap = {'N':[-1,0],'S':[1,0],'E':[0,1],'W':[0,-1]}

    ###### 1. Create mazes
    print("\n\n\033[91m-----> Maze creation\033[0m")
    mazes = []
    # load from file
    if args.fin is not None:
        base_name = os.path.basename(args.fin).rsplit('.', 1)[0]
        maze = []
        with open(args.fin, 'r') as fIn:
            for line in fIn.read().splitlines():
                if not line.strip():
                    # change env
                    print("\r Maze %d" % len(mazes), end=' ')
                    mazes.append(maze)
                    x1, x2, y1, y2 = mazeBoundaries(maze)
                    min_x = min(min_x, x1); max_x = max(max_x, x2);
                    min_y = min(min_y, y1); max_y = max(max_y, y2);
                    maze = []
                else:
                    maze.append(line.split())
        # add last maze (no empty last line)
        if len(maze) > 1:
            mazes.append(maze)
            x1, x2, y1, y2 = mazeBoundaries(maze)
            min_x = min(min_x, x1); max_x = max(max_x, x2);
            min_y = min(min_y, y1); max_y = max(max_y, y2);
    # or generate mazes
    else:
        base_name = "gen_%d_%d_%d_%d_%d_%d" % (args.size, args.init, args.trap, args.goal, args.wall, args.env)
        maze = np.pad(np.zeros((args.size - 1, args.size - 1), dtype=int) + 48, 1, 'constant', constant_values=49)
        n_cases = (args.size - 1) * (args.size - 1)
        n_choices = args.goal + args.init + args.trap + args.wall
        choices = range(n_cases)
        assert(n_choices <= n_cases)
        # for each environment
        for e in xrange(args.env):
            print("\r Maze %d/%d" % (e + 1, args.env), end=' ')
            # choose cases
            current = np.array(maze)
            cases = np.random.choice(choices, n_choices, replace=False)
            # write states
            for i in xrange(n_choices):
                c = cases[i]
                current[c // (args.size - 1) + 1, c % (args.size - 1) + 1] = 60 if i < args.init else 120 if i < args.init + args.trap else 103 if i < args.init + args.trap + args.goal else 49
            # append new environment
            str_maze = [[str(chr(x)) for x in line] for line in current]
            mazes.append(str_maze)
            x1, x2, y1, y2 = mazeBoundaries(str_maze)
            min_x = min(min_x, x1); max_x = max(max_x, x2)
            min_y = min(min_y, y1); max_y = max(max_y, y2)

    # Check that mazes shape are consistent
    aux = [(len(maze), len(maze[0])) for maze in mazes]
    assert(aux.count(aux[0]) == len(aux))
    width, height = aux[0]

    # Create output dir and files
    output_dir = os.path.join(args.output, base_name)
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir)
    f_rewards = open(os.path.join(output_dir, "%s.rewards" % base_name), 'w')
    f_transitions = open(os.path.join(output_dir, "%s.transitions" % base_name), 'w')
    f_summary = open(os.path.join(output_dir, "%s.summary" % base_name), 'w')
    f_summary.write("%d min x\n%d max x\n%d min y\n%d max y\n%d environments\n" % (min_x, max_x, min_y, max_y, len(mazes)))
    f_summary.write("%d inits\n%d goals\n%d traps\n%d walls\n%.3f wall failure\n" % (args.init, args.goal, args.trap, args.wall, args.wall_failure))
    f_summary.write("\nFailure rates for each environment:\n")

    # Store mazes if not loading from file
    if args.fin is None:
        with open(os.path.join(output_dir, "%s.mazes" % base_name), 'w') as f_mazes:
            f_mazes.write('\n\n'.join('\n'.join(' '.join(line) for line in m) for m in mazes))

    ###### 2. Create transitions function
    print("\n\n\033[91m-----> Transitions generation\033[0m")
    # Parse each maze
    from collections import Counter
    for e, maze in enumerate(mazes):
        print("\n   > Maze %d/%d \n" % (e + 1, len(mazes)), end=' ')
        if args.rdf:
            failures = np.random.rand(3) / 2. # failure rates, sampled in [0; 0.5)
            f_summary.write("%s\n" % (' '.join("%.3f" % x for x in failures)))
            print("      sampled failures:", failures)
        c = Counter([x for y in maze for x in y])
        n_init = c['v'] + c['>'] + c['^'] + c['<']
        for i in range(0, width):
            for j in range(0, height):
                print("\r      state %d/%d" % (4 * (i * height + j + 1), 4 * width * height), end=' ')
                element = maze[i][j]

                # I.N.I.T
                if element in ['>', '<', 'v', '^']:
                    current_state = "%dx%dx%s" % (i, j, 'E' if (element == '>') else ('W' if (element == '<') else ('S' if (element == 'v') else 'N')))
                    for action in actions:
                        f_transitions.write("%s %s %s %f\n" % ('S', action, current_state, 1.0 / n_init))

                # other states
                for orient in ['N','E','S','W']:
                    current_state = "%dx%dx%s" % (i, j, orient)
                    # T.R.A.P
                    if element == 'x':
                        for action in actions:
                            f_transitions.write("%s %s %s %f\n" % (current_state, action, 'T', 1.0))
                    # G.O.A.L
                    elif element == 'g':
                        for action in actions:
                            f_transitions.write("%s %s %s %f\n" % (current_state, action, 'G', 1.0))
                            f_rewards.write("%s %s %s %f\n" % (current_state, action, 'G', goal_reward))
                    # E.L.S.E
                    elif element != '1':
                        # Move forward
                        target, fail = ("%dx%dx%s" % (i + changeMap[orient][0], j + changeMap[orient][1],orient), failures[0]) if not isWall(i, j, orient) else ('T', 1.0 - wall_failure)
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'F', target, 1.0 - fail))
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'F', current_state, fail))

                        # Turn left
                        target = "%dx%dx%s" % (i, j, left[orient]);
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'L', target, 1.0 - failures[1]))
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'L', current_state, failures[1]))

                        # Turn right
                        target = "%dx%dx%s" % (i, j, right[orient]);
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'R', target, 1.0 - failures[2]))
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'R', current_state, failures[2]))

        # Next environment
        f_transitions.write("\n")
        f_rewards.write("\n")

    f_transitions.close()
    f_rewards.close()
    f_summary.close()

    ###### 3. End
    print("\n\n\033[92m-----> End\033[0m")
    print("   Output directory: %s" % output_dir)
    # End
