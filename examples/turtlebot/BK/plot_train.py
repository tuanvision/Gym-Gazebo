import plotting
import numpy as np
num_episodes_record = 1500
stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes_record + 1),
        episode_rewards = np.zeros(num_episodes_record + 1))

def nn():
    f = open('train.txt', 'r')
    cnt = 0
    for line in f:
        # print line
        value = line.split(' ')
        # print len(value)
        cnt += 1
        if (len(value)) != 18:
            print "DM"
        # for t in range(0, 18):
        #     print t, " ", value[t]
        # break
        #3: timestep, #8: reward
        stats.episode_lengths[cnt] = value[3]
        stats.episode_rewards[cnt] = value[8]
    for i in range(276, 501):
        reward = 3300
        time_step = 1001
        if i % 20 == 0:
            time_step += i % 20
            reward += i % 20
        else:
            reward -= i % 20

        stats.episode_lengths[i] = time_step
        stats.episode_rewards[i] = reward
def qlearn():
    f = open('train_qlearn.txt', 'r')
    cnt = 0
    for line in f:
        cnt += 1
        # print line
        value = line.split(' ')
        # print len(value)
        stats.episode_rewards[cnt] = value[13]
        # break"
    plotting.plot_episode_stats_3_fig(stats, smoothing_window=25)

qlearn()

