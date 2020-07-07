#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np


class GridWorld:
    def __init__(self, num_row=3, num_col=8, initial_pos=(0, 0), terminal_pos=(-1, -1), init_reward=0, slide_prob=0.1):
        self.num_row = num_row
        self.num_col = num_col
        self.grids = np.ones((num_row, num_col)) * init_reward
        self.n_state = num_row * num_col  # number of states
        self.init_x = initial_pos[0] # initial position x
        self.init_y = initial_pos[1] # initial position y
        if terminal_pos == (-1, -1):
            self.terminal_x = self.num_col - 1
            self.terminal_y = self.num_row - 1
        else:
            self.terminal_x = terminal_pos[0]
            self.terminal_y = terminal_pos[1]
        self.current_x = initial_pos[0] # current position
        self.current_y = initial_pos[1] # current position
        self.slide_prob = slide_prob
        self.action_space = ["left", "right", "up", "down"]

    def define_reward(self, x, y, reward):
        self.grids[y, x] = reward

    def __str__(self):
        output = "    "
        for col in range(self.num_col):
            output += "%4d" % col
        output += "\n"
        for row in range(self.num_row):
            output += "%4d" % row
            for col in range(self.num_col):
                output += "%4d" % self.grids[row, col]
            output += "\n"
        output += "Agent Pos: (%d, %d)\n" % (self.current_x, self.current_y)
        output += "Init Pos: (%d, %d)\n" % (self.init_x, self.init_y)
        output += "Terminate Pos: (%d, %d)\n" % (self.terminal_x, self.terminal_y)
        return output

    def print_state(self):
        output = "   "
        for col in range(self.num_col):
            output += "%3d" % col
        output += "\n"
        for row in range(self.num_row):
            output += "%3d" % row
            for col in range(self.num_col):                
                if row == self.current_y and col == self.current_x:
                    output += "  X"
                elif row == self.init_y and col == self.init_x:
                    output += "  S"
                elif row == self.terminal_y and col == self.terminal_x:
                    output += "  T"   
                else:
                    output += "  O"
            output += "\n"
        print(output)

    def reset(self):
        self.current_x = self.init_x
        self.current_y = self.init_y
        return [self.current_x, self.current_y]

    def update(self, a):
        # left
        if a == self.action_space[0]:
            if self.current_x > 0:
                # go left
                self.current_x -= 1
                # determine slide
                random_num = np.random.rand()
                # slide up
                if random_num <= self.slide_prob:
                    if self.current_y < self.num_row - 1:
                        self.current_y += 1
                # slide down
                elif random_num <= self.slide_prob * 2:
                    if self.current_y > 0:
                        self.current_y -= 1
            
        # right
        elif a == self.action_space[1]:
            if self.current_x < self.num_col - 1:
                # go right
                self.current_x += 1
                # determine slide
                random_num = np.random.rand()
                # slide down
                if random_num <= self.slide_prob:
                    if self.current_y < self.num_row - 1:
                        self.current_y += 1
                # slide up
                elif random_num <= self.slide_prob * 2:
                    if self.current_y > 0:
                        self.current_y -= 1

        # up
        elif a == self.action_space[2]:
            if self.current_y > 0:
                # go up
                self.current_y -= 1
                # determine slide
                random_num = np.random.rand()
                # slide right
                if random_num <= self.slide_prob:
                    if self.current_x < self.num_col - 1:
                        self.current_x += 1
                # slide left
                elif random_num <= self.slide_prob * 2:
                    if self.current_x > 0:
                        self.current_x -= 1

        # down
        elif a == self.action_space[3]:
            if self.current_y < self.num_row - 1:
                # go down
                self.current_y += 1
                # determine slide
                random_num = np.random.rand()
                # slide right
                if random_num <= self.slide_prob:
                    if self.current_x < self.num_col - 1:
                        self.current_x += 1
                # slide left
                elif random_num <= self.slide_prob * 2:
                    if self.current_x > 0:
                        self.current_x -= 1
            
        else:
            print("Action Invalid")
            return None, None

        if self.current_x == self.terminal_x and self.current_y == self.terminal_y:
            terminal = True
        else:
            terminal = False

        return [self.current_x, self.current_y], self.grids[self.current_y, self.current_x], terminal


if __name__ == "__main__":
    env = GridWorld(init_reward=-1, slide_prob=0)
    env.define_reward(7, 2, 3)
    env.define_reward(1, 1, 1)
    print(env)
    action_space = env.action_space
    env.print_state()
    for i in range(10):
        a = np.random.choice(action_space)
        print(a)
        print(env.update(a))
        env.print_state()

