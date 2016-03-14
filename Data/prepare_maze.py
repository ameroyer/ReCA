
__author__ = 'mchmelik'
import sys

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

fIn = open(sys.argv[1], 'r');
f_transitions = open(sys.argv[2], 'w');
f_rewards = open(sys.argv[3],'w')


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



print(goal_states)




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






