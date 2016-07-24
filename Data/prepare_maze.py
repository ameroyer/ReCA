__authors__ = 'mchmelik, aroyer'
import os
import sys
import shutil
import argparse
import numpy as np

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
    parser.add_argument("-i",  "--fin", type=str, help="If given, load the mazes from fiels (takes precedence over the other parameters.")
    parser.add_argument("-n", "--size", type=int, default=5, help="size of the maze")
    parser.add_argument("-s", "--init", default=1, type=int, help="number of initial states per maze")
    parser.add_argument("-t", "--trap", default=1, type=int, help="number of trap states per maze")
    parser.add_argument("-g", "--goal", default=1, type=int, help="number of goal states per maze")
    parser.add_argument("-e", "--env", default=1, type=int, help="number of environments to generate for")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Code", "Models"), help="Path to output directory.")
    args = parser.parse_args()
    # Hyperparameters
    actions = ['F','L','R']
    failures = [0.2, 0.1, 0.1]
    goal_reward = 5.0

    # Load maze string representation
    mazes = []
    # from file
    if args.fin is not None:
        base_name = os.path.basename(args.fin).rsplit('.', 1)[0]
        maze = []
        min_x, max_x, min_y, max_y = sys.maxint, 0, sys.maxint, 0
        with open(args.fin, 'r') as fIn:
            for line in fIn.read().splitlines():
                if not line.strip():
                    # Change env
                    mazes.append(maze)
                    x1, x2, y1, y2 = mazeBoundaries(maze)
                    min_x = min(min_x, x1); max_x = max(max_x, x2);
                    min_y = min(min_y, y1); max_y = max(max_y, y2);
                    maze = []
                else:
                    maze.append(line.split())
        # Add last maze if not done
        if len(maze) > 1:
            mazes.append(maze)
            x1, x2, y1, y2 = mazeBoundaries(maze)
            min_x = min(min_x, x1); max_x = max(max_x, x2);
            min_y = min(min_y, y1); max_y = max(max_y, y2);
    # or generated
    else:
        base_name = "gen_%dx%d_%d%d%d_%d" % (args.size, args.size, args.init, args.trap, args.goal, args.env)
        maze = np.pad(np.zeros((args.size - 1, args.size - 1), dtype=int) + 48, 1, 'constant', constant_values=49)
        n_cases = (args.size - 1) * (args.size - 1)
        n_choices = args.goal + args.init + args.trap
        choices = range(n_cases)
        assert(n_choices <= n_cases)
        # for each environment
        for e in xrange(args.env):
            # choose cases
            current = np.array(maze)
            cases = np.random.choice(choices, n_choices, replace=False)
            # write states
            for i in xrange(n_choices):
                c = cases[i]
                current[c / (args.size - 1) + 1, c % (args.size - 1) + 1] = 60 if i < args.init else 120 if i < args.init + args.trap else 103
            # append new environment
            mazes.append([[str(unichr(x)) for x in line] for line in current])
        print '\n\n'.join('\n'.join(''.join(line) for line in maze) for maze in mazes)
        raise SystemExit
    
    # Check that mazes shape are consistent
    aux = [(len(maze), len(maze[0])) for maze in mazes]
    assert(aux.count(aux[0]) == len(aux))
    width, height = aux[0]

    # Create output dir and files
    output_dir = os.path.join(args.output, base_name)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    f_transitions = open(os.path.join(output_dir, "%s.transitions" % base_name), 'w')
    f_rewards = open(os.path.join(output_dir, "%s.rewards" % base_name), 'w')

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
