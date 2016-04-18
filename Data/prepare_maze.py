__author__ = 'mchmelik'
import os
import sys
import shutil
import argparse

left = {'N':'W','W':'S','S':'E','E':'N'}
right = {'N':'E','E':'S','S':'W','W':'N'}
changeMap = {'N':[-1,0],'S':[1,0],'E':[0,1],'W':[0,-1]}


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

def turnLeft(orient):
    """
    Return the orientation to the left of ``orient``.
    """
    return left[orient]

def turnRight(orient):
    """
    Returns the orientation to the right of ``orient``.
    """
    return right[orient]


def mazeBoundaries(maze):
    """
    Return the reachable boundaries of the maze.
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
    ###### Parameters
    base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description='Generate Maze MEMDP.')
    parser.add_argument("fin", type=str, default=os.path.join(base_folder, "Code", "Models"), help="Path to output directory.")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Code", "Models"), help="Path to output directory.")
    args = parser.parse_args()
    base_name = os.path.basename(args.fin).rsplit('.', 1)[0]
    output_dir = os.path.join(args.output, base_name)
    # Hyperparameters
    actions = ['F','L','R']
    failures = [0.2, 0.1, 0.1]
    goal_reward = 5.0
    
    # Create output dir and files
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    f_transitions = open(os.path.join(output_dir, "%s.transitions" % base_name), 'w')
    f_rewards = open(os.path.join(output_dir, "%s.rewards" % base_name), 'w')

    # Load every maze
    mazes = []
    maze = []
    min_x, max_x, min_y, max_y = sys.maxint, 0, sys.maxint, 0
    with open(args.fin, 'r') as fIn:
        for line in fIn.readlines():
            if not line.strip():
                # Change env
                mazes.append(maze)
                x1, x2, y1, y2 = mazeBoundaries(maze)
                min_x = min(min_x, x1); max_x = max(max_x, x2);
                min_y = min(min_y, y1); max_x = max(max_y, y2);
                maze = []
            else:
                maze.append(line.split())

    # Check tha mazes shape are consisten
    aux = [(len(maze), len(maze[0])) for maze in mazes]
    assert(aux.count(aux[0]) == len(aux))
    width, height = aux[0]
    
    # Parse each maze
    from collections import Counter
    for maze in mazes:
        c = Counter([x for y in maze for x in y])
        n_init = c['v'] + c['>'] + c['^'] + c['<']
        for i in range(0, width):
            for j in range(0, height):
                element = maze[i][j]
                
                # I.N.I.T
                if element in ['>', '<', 'v', '^']:
                    current_state = "%dx%dx%s" % (i, j, 'E' if (element == '>') else ('W' if (element == '<') else ('S' if (element == 'v') else 'N')))
                    for action in actions:
                        f_transitions.write("%s %s %s %f\n" % ('S', action, current_state, 1.0 / n_init))
                
                # Other states
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
                        target = "%dx%dx%s" % (i + changeMap[orient][0], j + changeMap[orient][1],orient) if not isWall(i, j, orient) else 'T'
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'F', target, 1.0 - failures[0]))
                        f_transitions.write("%s %s %s %f\n" % (current_state, 'F', current_state, failures[0]))

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
                            
        
    # END
    f_transitions.close()
    f_rewards.close()

    # write out summary file
    f_summary = open(os.path.join(output_dir, "%s.summary" % base_name), 'w')
    f_summary.write("%d min x\n%d max x\n%d min y\n%d max y\n%d environments" % (min_x, max_x, min_y, max_y, len(mazes)))
    f_summary.close()

raise SystemExit
#### Generate some test sequences
import numpy as np
from collections import defaultdict
from random import randint
from numpy.random import choice

# parameters
n_env = len(goal_states)
n_obs = 3 + (max_x - min_x + 1) * (max_y - min_y + 1) * 4
n_links = 5
n_actions = 3
def state_to_id(line):
    global min_x, max_x, min_y, max_y
    x, y, o = line.split('x')
    x = int(x); y = int(y);
    o = 0 if o == 'N' else 1 if o == 'E' else 2 if o == 'S' else 3
    return 3 + (y - min_y) + (max_y - min_y + 1) * ((x - min_x) + (max_x - min_x + 1) * o)

def action_to_id(a):
    if a == 'L':
        return 0
    elif a == 'R':
        return 1
    else:
        return 2


def link(line1, line2):
    global n_links
    x1, y1, o1 = line1.split('x')
    x1 = int(x1); y1 = int(y1); o1 = 0 if o1 == 'N' else 1 if o1 == 'E' else 2 if o1 == 'S' else 3
    x2, y2, o2 = line2.split('x')
    x2 = int(x2); y2 = int(y2); o2 = 0 if o2 == 'N' else 1 if o2 == 'E' else 2 if o2 == 'S' else 3
    # No Move
    if x1 == x2 and y1 == y2 and o1 == o2:
        return n_links - 2
    # Change of orientation
    elif x1 == x2 and y1 == y2:
        if o2 == (o1 + 3) % 4:
            return 0
        elif o2 == (o1 + 1) % 4:
            return 1
    # Forward
    elif o1 == o2:
        if ((o1 == 0 and x2 == x1 - 1 and y1 == y2) or (o1 == 1 and x1 == x2 and y2 == y1 + 1) or (o1 == 2 and x2 == x1 + 1 and y1 == y2) or (o1 == 3 and x1 == x2 and y2 == y1 - 1)):
            return 2
    # Default: fail
    return n_links

def next_state(s, lk):
    ors = ['N', 'E', 'S', 'W']
    # Find x, y, o
    state = s - 3
    y = state % (max_y - min_y + 1) + min_y
    state = state / (max_y - min_y + 1)
    x = state % (max_x - min_x + 1) + min_x
    o = state / (max_x - min_x + 1)
    # Go
    if lk == n_links - 1:
        return 2
    elif lk == n_links - 2:
        return s
    elif lk == 2:
        if o == 0 and x > min_x:
            return state_to_id("%dx%dx%s" % (x - 1, y, ors[o]))
        elif o == 1 and y < max_y:
            return state_to_id("%dx%dx%s" % (x, y + 1, ors[o]))
        elif o == 2 and x < max_x:
            return state_to_id("%dx%dx%s" % (x + 1, y, ors[o]))
        elif o == 3 and y > min_y:
            return state_to_id("%dx%dx%s" % (x, y - 1, ors[o]))           
    # Orientation
    elif lk == 1:
        return state_to_id("%dx%dx%s" % (x, y, ors[(o + 1) % 4]))
    elif lk == 0:
        return state_to_id("%dx%dx%s" % (x, y, ors[(o + 3) % 4]))
    # Default: fail
    print x, y, o, lk
    return -1


# Load transitions
transitions = np.zeros((n_env, n_obs, n_actions, n_links))
G_states = defaultdict(lambda: [])
starting_states = defaultdict(lambda: [])
trap_states = defaultdict(lambda: [])
with open(os.path.join(output_dir, "%s.transitions" % base_name), 'r') as f:
    env = 0
    for line in f.read().splitlines():
        # Skip to next environment
        try :
            s1, a, s2, v = line.split()
        except ValueError:
            env += 1
            continue
        # States that might reach T. Strive to find t by list expansion
        if s2 == 'T' and s1 != 'T':
            trap_states[env].append(state_to_id(s1))
        # Ignore absorbing transition
        if s1 == 'T' or s1 == 'G':
            continue
        # Detect goal states
        elif s2 == 'G':
            G_states[env].append(state_to_id(s1))
        # Detect trap states
        elif s2 == 'T':
            transitions[env, state_to_id(s1), action_to_id(a), n_links - 1] = float(v)
        # Detect starting states
        elif s1 == 'S':
            starting_states[env].append(state_to_id(s2))
        # Normal case
        else:
            transitions[env, state_to_id(s1), action_to_id(a), link(s1, s2)] = float(v)

# Remove duplicates
for env, l in starting_states.iteritems():
    starting_states[env] = list(set(l))
for env, l in G_states.iteritems():
    G_states[env] = list(set(l))

# Try to find trap states by extension

#print transitions
# Generate test sessions
f_test = open(os.path.join(output_dir, "%s.test" % base_name), 'w')
for user in xrange(args.test):
    # Choose environment
    env = randint(0, n_env - 1)
    state = 0
    trace = [state]
    # While non absorbing
    while state != 1 and state != 2:
        # Choose action
        a = randint(0, n_actions - 1)
        # If start stae
        if state == 0:
            state = choice(starting_states[env])
        # If goal state
        elif state in G_states[env]:
            state = 1
        # Normal state
        else:
            old_state = state
            lk = choice(range(n_links), p=transitions[env, state, a, :])
            state = next_state(state, lk)
        # Append
        trace.append(a)        
        trace.append(state)

    # DEBUG and DIRTY
    # For debugging purpose, print the trace as a maze
    ## maze = [['' for _ in xrange(max_y - min_y + 1)] for _ in xrange(max_x - min_x + 1)]
    ## seen = []
    ## for i, s in enumerate(trace):
    ##     if i % 2 != 0 or s in [0, 2]:
    ##         continue
    ##     # Goal
    ##     if s == 1:
    ##         maze[x][y] = 'G'
    ##     # Print
    ##     elif not s in seen:
    ##         state = s - 3
    ##         y = state % (max_y - min_y + 1)
    ##         state = state / (max_y - min_y + 1)
    ##         x = state % (max_x - min_x + 1)
    ##         o = state / (max_x - min_x + 1)
    ##         seen.append(s)
    ##         maze[x][y] = maze[x][y] + ' ' + str(s) + ( '^' if o ==0 else '>' if o == 1 else 'v' if o == 2 else '<')
    ## # Trap states
    ## seen = []
    ## for s in trap_states[env]:
    ##     state = s - 3
    ##     y = state % (max_y - min_y + 1)
    ##     state = state / (max_y - min_y + 1)
    ##     x = state % (max_x - min_x + 1)
    ##     o = state / (max_x - min_x + 1)
    ##     if not (x, y) in seen:
    ##         seen.append((x, y))
    ##         maze[x][y] += ' T'        
    ## print  '\n'.join(str(line) for line in maze)
    ## END DEBUG
    
    # Write
    f_test.write("%d\t%d\t%s\n" % (user, env, ' '.join(str(x) for x in trace)))
f_test.close()








