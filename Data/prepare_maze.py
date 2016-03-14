__author__ = 'mchmelik'
import os
import sys
import shutil
import argparse

left = {'N':'W','W':'S','S':'E','E':'N'}
right = {'N':'E','E':'S','S':'W','W':'N'}

changeMap = {'N':[-1,0],'S':[1,0],'E':[0,1],'W':[0,-1]}


def isWall(i,j,direction):
    if(direction.__eq__('N')):
        return (maze[i-1][j] == '1')
    if(direction.__eq__('E')):
        return (maze[i][j+1] == '1')
    if(direction.__eq__('W')):
        return (maze[i][j-1] == '1')
    if(direction.__eq__('S')):
        return (maze[i+1][j] == '1')

def turnLeft(orient):
    return left[orient]

def turnRight(orient):
    return right[orient]

###### Parameters
base_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parser = argparse.ArgumentParser(description='Generate Maze MEMDP.')
parser.add_argument("fin", type=str, default=os.path.join(base_folder, "Code", "Models"), help="Path to output directory.")
parser.add_argument('-o', '--output', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Code", "Models"), help="Path to output directory.")
args = parser.parse_args()
base_name = os.path.basename(args.fin).rsplit('.', 1)[0]
output_dir = os.path.join(args.output, base_name)
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
fIn = open(args.fin, 'r')
f_transitions = open(os.path.join(output_dir, "%s.transitions" % base_name), 'w')
f_rewards = open(os.path.join(output_dir, "%s.rewards" % base_name), 'w')

actions = {'F','L','R'}
maze = [];
for line in fIn.readlines():
    maze.append(line.split());

width = len(maze);
height = len(maze[0]);


goal_states = []
initialStates = []

for i in range(0, width):
    for j in range(0, height):
      if maze[i][j] == 'g':
        goal_states.append((i,j))
      if maze[i][j] == '+':
        initialStates.append((i,j))



#print(goal_states)




for goal_row,goal_column in goal_states:
  transitions = []
  for action in actions:
    transitions.append(('T', action, 'T',1.0))
    transitions.append(('G', action, 'G',1.0))


  for i in range(0, width):
    for j in range(0, height):
        element = maze[i][j]
        if goal_row == i and goal_column == j:
          goal = True
        else:
          goal = False

        for orient in {'N','E','S','W'}:
            current_state = "%dx%dx%s" %(i,j,orient)
            if element in {"0","+"} or (not goal and element in {"g"}):
                # Move forward
                target = "%dx%dx%s" % (i+changeMap[orient][0],j+changeMap[orient][1],orient) if not isWall(i,j,orient) else "T";
                transitions.append((current_state,'F',target,0.8))
                transitions.append((current_state,'F',current_state,0.2))

                # Turn left
                target = "%dx%dx%s" % (i,j,left[orient]);
                transitions.append((current_state,'L',target,0.9))
                transitions.append((current_state,'L',current_state,0.1))

                # Turn right
                target = "%dx%dx%s" % (i,j,right[orient]);
                transitions.append((current_state,'R',target,0.9))
                transitions.append((current_state,'R',current_state,0.1))

            if element in {"x"}:
              for action in actions:
                transitions.append((current_state,action,'T',1.0))
            if element in {"g"} and goal:
              for action in actions:
                transitions.append((current_state,action,'G',1.0))
  for initial_column, initial_row in initialStates:
    target = "%dx%dxS" % (initial_column,initial_row);
    prob = 1.0 / len(initialStates)
    for action in actions:
      transitions.append(('S',action,target,prob))



  # write out transitions for every environment
  for trans in transitions:
    line = ' '.join(map(str,trans))+"\n"
    f_transitions.write(line)
    source = trans[0]
    target = trans[2]
    if source != "G" and target == "G":
      line = ' '.join(map(str,trans[:3])) + " 5.0\n"
      f_rewards.write(line)

  f_transitions.write("\n")
  f_rewards.write("\n")


# END
f_transitions.close()
f_rewards.close()

# write out summary file
f_summary = open(os.path.join(output_dir, "%s.summary" % base_name), 'w')

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

f_summary.write("%d min x\n%d max x\n%d min y\n%d max y\n%d environments" % (min_x, max_x, min_y, max_y, len(goal_states)))
f_summary.close()







