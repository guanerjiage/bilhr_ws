#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

class Treasure1D:
    def __init__(self, n_state=8):
        self.n_state = n_state  # number of states
        self.action_space = ["left", "right"]
        self.s = 0  # current state

    def update(self, a):
        if a == "right":
            if self.s == self.n_state - 2:
                # s_ = "terminal"
                s_ = self.s + 1
                r = 20
            else:
                s_ = self.s + 1
                r = -1
        elif a == "left":
            r = -1
            if self.s == 0:
                s_ = self.s
            else:
                s_ = self.s - 1
        else:
            print("ERROR")
            s_ = self.s
            r = -1
        self.s = s_
        return s_, r

    def reset(self):
        self.s = 0
        return self.s

    def __str__(self):
        env_list = ['-'] * (self.n_state - 1) + ['T']
        env_list[self.s] = 'o'
        return "".join(env_list)

