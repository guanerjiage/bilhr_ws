# Tutorial 5

## Source code
Codes related to decision tree and Reinforcement Learning algorithms are in the folder scripts/rldt. There are some simple environments and dataset to test the algorithms.  
scripts/rl_node.py is used to collect transitions and to train the agent for penalty kick.  
scripts/central_node.py automatically loads the agent data in the script/RLDT_penalty_XXX.json.  

## To test the robot's performance:

1. Open the modified nao robot environment naoqisim_goal_rl_modified.wbt.  
2. Drag the "goalkeeper" to desired position (note that the quantization of goalkeeper position has only 5 levels).
3. Launch Naoqi controller.
4. Run central_node.py, be sure the terminal path is the folder with RLDT_penalty_XXX.json, otherwise the robot will use an empty agent.
5. Run keyboard_node.py, first type "penaltyready", wait until the robot finished the pose change for penalty kick. Then type "penaltydt" to let the robot kick the ball using RL-DT agent.
6. To run the next test, you must reset the simulation in Webot and relaunch NAO controller, central_node.py, and keyboard_node.py

## Training procedure
1. Open the modified nao robot environment naoqisim_goal_rl_modified.wbt.  
2. Drag the "goalkeeper" to desired position (note that the quantization of goalkeeper position has only 5 levels).
3. Launch Naoqi controller.
4. Run central_node.py.
5. Run rl_node.py, type "start" to let the robot prepare to kick. If continue training, type "load" to load the transitions from RL_transitions.json. Type "next" to perform the next episode of learning. Watch the action, give a reward. If reward not equal to -1, the episode finishes. Type "store" to save the transition dataset. If training finishes, type "save" to save the agent.
6. To run the next episode of learning, you must reset the simulation in Webot and relaunch NAO controller, central_node.py, and rl_node.py

## Training data
- We trained agents for different goalkeeper positions separately. The transitions are stored in corresponding folders in scripts/rldt_training_data/. Then we concatenated these datasets to one and use all the transitions to train the final agent offline. The concatenated transition dataset and the final agent data are stored in the folder scripts/rldt_training_data/All. The agent data are copied to scripts/ for the central_node.py.

## Results and plots
- The video BILfHR_t5_demo.m4v shows a demo of the final agent. Due to the time limit, we can only show the goals and some typical fails. 
- The plots of cummulative rewards are placed in the folder plots/. We can see that goalkeeper at center is the most difficult case (as expected). Due to the noise and the randomness of the environment and the robot itself, the robot has a chance to score at many foot positions. The agent must try to kick at all positions many times to find the position with the highest score probability. As shown in reward_c.png, this exploration happend in the first 80 episodes. It got negative rewards in most episodes. Then the agent finally found a position with satisfactory average reward and switched to exploitation mode. Then the cummulative reward started to increase and we could stop the learning. In this case, the exploration strategy of RL-DT introduced in the given paper is very useful. The agent can have an aim to explore, which can greatly increase sample efficiency.
- reward_l.png and reward_r.png show the cummulative reward for the goalkeeper at the very left and right position. The robot can achieve a goal at all positions with very high probability. We can see the agent could already obtain positive reward in the first or second episode. Then it directly switched to exploitation mode. But the agent might get trapped into suboptimum and cannot explore better policy (for example less movements before kick). To solve this problem, we add an epsilon-greedy action choice, which is widely applied in model-free RL, such that the agent can still do some exploration even the RL-DT algorithm evaluates the value functions as "satisfied".
- reward_cl.png and reward_cr.png show the cummulative reward for the goalkeeper at center left and center right position (see the video). These are a bit more difficult than the previous cases (very left and right). In the center right case, the agent took about 25 episodes to explore the foot positions. Then it could exploit the knowledge to increase cummulative reward. In the center left case, the agent got a positive reward already in the first episode, we must manually add some randomness in the action choice.