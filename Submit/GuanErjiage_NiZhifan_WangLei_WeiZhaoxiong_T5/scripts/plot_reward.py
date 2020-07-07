#!/usr/bin/env python
# Author: Erjiage GUAN, Zhifan NI, Lei WANG, Zhaoxiong WEI

import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    with open("rldt_training_data/All/RL_reward.json", "r") as f:
        r_json_all = f.read()
        r_episode_all = json.loads(r_json_all)["r_cum"]
        r_cum_all = [0]
        for r in r_episode_all:
            r_cum_all.append(r_cum_all[-1] + r)

    with open("rldt_training_data/All/RL_reward_c.json", "r") as f:
        r_json_c = f.read()
        r_episode_c = json.loads(r_json_c)["r_cum"]
        r_cum_c = [0]
        for r in r_episode_c:
            r_cum_c.append(r_cum_c[-1] + r)

    with open("rldt_training_data/All/RL_reward_cl.json", "r") as f:
        r_json_cl = f.read()
        r_episode_cl = json.loads(r_json_cl)["r_cum"]
        r_cum_cl = [0]
        for r in r_episode_cl:
            r_cum_cl.append(r_cum_cl[-1] + r)

    with open("rldt_training_data/All/RL_reward_cr.json", "r") as f:
        r_json_cr = f.read()
        r_episode_cr = json.loads(r_json_cr)["r_cum"]
        r_cum_cr = [0]
        for r in r_episode_cr:
            r_cum_cr.append(r_cum_cr[-1] + r)

    with open("rldt_training_data/All/RL_reward_l.json", "r") as f:
        r_json_l = f.read()
        r_episode_l = json.loads(r_json_l)["r_cum"]
        r_cum_l = [0]
        for r in r_episode_l:
            r_cum_l.append(r_cum_l[-1] + r)

    with open("rldt_training_data/All/RL_reward_r.json", "r") as f:
        r_json_r = f.read()
        r_episode_r = json.loads(r_json_r)["r_cum"]
        r_cum_r = [0]
        for r in r_episode_r:
            r_cum_r.append(r_cum_r[-1] + r)

    plt.figure()
    plt.plot(r_cum_all)
    plt.title("Cummulative Rewards over all Goalkeeper Position")
    plt.xlabel("Episode")
    plt.ylabel("Cummulative reward")
    # plt.show()

    plt.figure()
    plt.plot(r_cum_c)
    plt.title("Cummulative Rewards of Learning with Goalkeeper at Center")
    plt.xlabel("Episode")
    plt.ylabel("Cummulative reward")
    # plt.show()

    plt.figure()
    plt.plot(r_cum_cl)
    plt.title("Cummulative Rewards of Learning with Goalkeeper at Center Left")
    plt.xlabel("Episode")
    plt.ylabel("Cummulative reward")
    # plt.show()

    plt.figure()
    plt.plot(r_cum_cr)
    plt.title("Cummulative Rewards of Learning with Goalkeeper at Center Right")
    plt.xlabel("Episode")
    plt.ylabel("Cummulative reward")
    # plt.show()

    plt.figure()
    plt.plot(r_cum_l)
    plt.title("Cummulative Rewards of Learning with Goalkeeper at Left")
    plt.xlabel("Episode")
    plt.ylabel("Cummulative reward")
    # plt.show()

    plt.figure()
    plt.plot(r_cum_r)
    plt.title("Cummulative Rewards of Learning with Goalkeeper at Right")
    plt.xlabel("Episode")
    plt.ylabel("Cummulative reward")
    plt.show()


    
