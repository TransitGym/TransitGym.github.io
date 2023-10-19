import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy.ma as ma
import math
import matplotlib
import glob
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
from matplotlib import gridspec
import matplotlib.ticker as plticker

matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica", "Arial"], "size": 18})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=1, labelsize='medium')
matplotlib.rc("axes.formatter", limits=[-7, 7])
matplotlib.rc('savefig', bbox='tight', format='pdf', pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=4)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', size=14)
matplotlib.rc('xtick.minor', size=2)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', size=14)
matplotlib.rc('ytick.minor', size=2)
matplotlib.rcParams['lines.solid_capstyle'] = 'butt'
matplotlib.rcParams['lines.solid_joinstyle'] = 'bevel'
matplotlib.rc('mathtext', fontset='stixsans')
matplotlib.rc('legend', fontsize='medium', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)


def tag_fig(ax=None, text='(a)', offset=[-44, 18], fontsize=12):
    if np.ndim(ax) > 1:
        w = ax[0, 0].annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large', weight='bold',
                              xytext=(offset[0] / 12 * fontsize, offset[1] / 12 * fontsize), textcoords='offset points',
                              ha='left', va='top')
        print(w.get_fontname())
    else:
        w = ax.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize='large', weight='bold',
                        xytext=(offset[0] / 12 * fontsize, offset[1] / 12 * fontsize), textcoords='offset points',
                        ha='left', va='top')
        print(w.get_fontname())


def vis_bb():
    path = 'G:\\mcgill\\useful\\vis\\'
    files = ['visnc\\', 'visaccf\\']
    model = ['bb', 'nbb']
    tag = ['(a)', '(b)']

    for i in range(len(files)):
        path_ = path + files[i]
        fig = plt.figure()
        ax = plt.gca()
        fig.set_size_inches((6, 3))
        tag_fig(ax=ax, text=tag[i])
        for file_name in glob.glob(path_ + '*.csv'):
            bus = file_name.split('_')[-1].split('.')[0]
            df = pd.DataFrame(pd.read_csv(file_name))

            # select the duplicated location row
            dp = df[df.duplicated('loc', keep=False)]
            # get the first and last idx of each duplicated series (i.e., get first&last index of a in [a,a,a,b,b,b])
            idx = dp.index[-dp.duplicated('loc', keep='last')].tolist() + dp.index[
                -dp.duplicated('loc', keep='first')].tolist()
            idx = sorted(idx)
            idx += [df.shape[0] - 1]
            times = df['time'][idx]
            locs = df['loc'][idx]

            plt.plot(times, locs, color='green')

        ax.autoscale()
        print(model[i])
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)

        fig.subplots_adjust(right=0.8)
        plt.xlabel('Time step (sec)')
        # cbar_ax = fig.add_axes([0.12, 0.9, 0.62, 0.01])  [left, bottom, width, height]

        plt.ylabel('Distance (km)')
        plt.subplots_adjust(hspace=.0)

        fig.savefig(model[i] + ".pdf", bbox_inches='tight')
        plt.show()


def sim_vs_real():
    # visualize comparison between simulation and real data
    path = 'G:\\mcgill\\MAS\\gtfs_testbed\\result\\sim_compSG_22_1.csv'

    sim_comp = pd.DataFrame(pd.read_csv(path))
    actual_onboard = sim_comp.iloc[:]['actual_onboard'].to_list()
    sim_onboard = sim_comp.iloc[:]['sim_onboard'].to_list()
    sim_travel_cost = sim_comp.iloc[:]['sim_travel_cost'].to_list()
    actual_travel_cost = sim_comp.iloc[:]['actual_travel_cost'].to_list()

    f = plt.figure()
    ax = plt.gca()

    # ax.xaxis.set_tick_params(labelsize=14)
    # ax.yaxis.set_tick_params(labelsize=14)
    loc = plticker.MultipleLocator(15000.)
    ax.xaxis.set_major_locator(loc)
    loc = plticker.MultipleLocator(15000.)
    ax.yaxis.set_major_locator(loc)
    tag_fig(ax=ax, text='(a)')
    f.set_size_inches((5, 4))
    # plt.grid(which='major', axis='both', color='white', linestyle='-', linewidth=1)
    plt.xlabel('Simulated boarding time (sec)')
    plt.ylabel('Actual boarding time (sec) ')
    plt.plot(actual_onboard, actual_onboard, linewidth=2, c='r')
    plt.scatter(sim_onboard, actual_onboard, facecolors='none', edgecolors='c')
    f.savefig("boarding_diff.pdf", bbox_inches='tight')
    plt.show()

    f = plt.figure()
    f.set_size_inches((4, 4))
    ax = plt.subplot(111)
    tag_fig(ax=ax, text='(b)')

    # ax.xaxis.set_tick_params(labelsize=14)
    # ax.yaxis.set_tick_params(labelsize=14)
    loc = plticker.MultipleLocator(1200.)
    ax.xaxis.set_major_locator(loc)
    loc = plticker.MultipleLocator(800.)
    ax.yaxis.set_major_locator(loc)
    f.set_size_inches((5, 4))
    # plt.grid(which='major', axis='both', color='white', linestyle='-', linewidth=1)
    plt.plot(actual_travel_cost, actual_travel_cost, linewidth=2, c='r')
    plt.scatter(sim_travel_cost, actual_travel_cost, facecolors='none', edgecolors='c')

    plt.xlabel('Simulated journey time (sec)')
    plt.ylabel('Actual journey time (sec) ')
    f.savefig("cost_diff.pdf", bbox_inches='tight')
    plt.show()


def sensitive_analysis():
    # sensitive analysis
    seed = [0, 1, 2]
    weight = [2, 4, 6, 8]

    model = ['accf', 'accflstm', 'ddpg', 'maddpg']
    model_name = ['CAAC-MA-A', 'CAAC-LA-A', 'IAC', 'MADDPG']
    path = 'G:\\mcgill\\useful\\log\\SG_22_1'

    ncrewards = []
    fhrewards = []
    accfrewards = []
    lstmaccfrewards = []
    ddpgrewards = []
    maddpgrewards = []

    def grasp(i, j, last=10):
        reward1 = []
        reward2 = []
        reward3 = []
        for s in range(len(seed)):
            try:
                # path_ = path +'w'+str(weight[j]) +'\\'+model[i][s] + '.csv'
                path_ = path + model[i] + str(weight[j]) + 'all' + str(s) + '.csv'
                train = pd.DataFrame(pd.read_csv(path_))
                reward1 += train.iloc[:]['reward1'].to_list()[-last:]
                reward2 += [np.clip(abs(a), 0, 3) for a in train.iloc[:]['action'].to_list()[
                                                           -last:]]  # [-last:]]train.iloc[:]['action'].to_list()[-last:]
                reward3 += train.iloc[:]['reward'].to_list()[-last:]
            except:
                print(path_)
                continue

        return np.mean(reward1), np.std(reward1), np.mean(reward2), np.std(reward2), np.mean(reward3), np.std(reward3)

    result = {}
    c = ['red', 'blue', 'orange', 'green']
    i = 3

    for i in range(len(model)):
        m1s = []
        m2s = []
        m3s = []
        s1s = []
        s2s = []
        s3s = []
        v = [4, 2, 4, 6, 4, 2]
        for j in range(len(weight)):
            m1, s1, m2, s2, m3, s3 = grasp(i, j)

            m1s.append(m1)
            m2s.append(m2)
            m3s.append(m3)
            s1s.append(s1)
            s2s.append(s2)
            s3s.append(s3)

        fig = plt.figure()
        ax = plt.gca()

        fig.set_size_inches((4, 4))
        # ax.axvline(x=v[i],color='red', linestyle='--',linewidth=2)
        ax.errorbar(weight, m1s, yerr=s1s, label='Reward for $CV^2$)', marker='o', capsize=2, color="green")
        # ax.plot(weight, [ m1s[k] for k in range(len(m1s))], color="red", marker="o")
        ax.set_ylabel("Reward for $CV^2$", color="green")
        ax2 = ax.twinx()
        # make a plot with different y-axis using second axis object
        # ax2.plot(weight, [m2s[k] for k in range(len(m1s))], color="blue", marker="o")
        ax2.errorbar(weight, m2s, yerr=s2s, label='Reward for control penalty', marker='o', capsize=2, color="blue")
        ax2.set_ylabel("Average holding action", color="blue")
        ax2.set_xlabel("Weight")

        ax.set_xticks(weight)
        ax.set_xticklabels([str(k / 10.) for k in weight])
        fig.savefig(model_name[i] + "senstive.pdf", bbox_inches='tight')
        print(model_name[i])
        plt.show()


def test_comp():
    # visualize comparison
    seed = [0, 1, 2, 3, 4, 5]
    accfpaths = ['accf6all' + str(s) for s in seed]
    ddpgpaths = ['ddpg6all' + str(s) for s in seed]
    maddpgpaths = ['maddpg6all' + str(s) for s in seed]
    ncpaths = ['nc' + str(s) for s in seed]
    fhpaths = ['fc' + str(s) for s in seed]
    model = [ncpaths, fhpaths, accfpaths, ddpgpaths, maddpgpaths]

    path = 'G:\\mcgill\\useful\\logt3\\SG_22_1'

    ncrewards = []
    fhrewards = []
    accfrewards = []
    lstmaccfrewards = []
    ddpgrewards = []
    maddpgrewards = []

    def grasp(i, j, last=2):
        reward1 = []
        reward2 = []
        reward3 = []
        reward4 = []
        for s in range(len(model[i])):
            path_ = path + model[i][s] + '.csv'
            try:
                train = pd.DataFrame(pd.read_csv(path_))
                reward1 += train.iloc[:]['wait'].to_list()[-last:]
                reward2 += train.iloc[:]['avg_hold'].to_list()[
                           -last:]  # [-last:]]train.iloc[:]['action'].to_list()[-last:]
                reward3 += train.iloc[:]['AOD'].to_list()[-last:]
                tt = train["travel"] - train["wait"]
                reward4 += tt.to_list()[-last:]
            except:
                print(path_)
                continue

        return np.mean(reward1), np.std(reward1), np.mean(reward2), np.std(reward2), np.mean(reward3), np.std(
            reward3), np.mean(reward4), np.std(reward4)

    result = {}
    c = ['red', 'blue', 'orange', 'green']

    m1s = []
    m2s = []
    m3s = []
    m4s = []
    s1s = []
    s2s = []
    s3s = []
    s4s = []

    for i in range(len(model)):
        m1, s1, m2, s2, m3, s3, m4, s4 = grasp(i, 6)
        m1s.append(m1)  # AWT
        m2s.append(m2)  # AHD
        m3s.append(m3)  # AOT
        m4s.append(m4)  # ATT
        s1s.append(s1)
        s2s.append(s2)
        s3s.append(s3)
        s4s.append(s4)
    indes = ['ATT', 'AHD', 'AWT', 'AOD']
    for k in range(len(indes)):
        index = indes[k]

        fig = plt.figure()
        ax = plt.gca()
        fig.set_size_inches((4, 4))
        if index == 'AWT':
            m = m1s
            s = s1s
        if index == 'AHD':
            m = m2s
            s = s2s
        if index == 'AOD':
            m = m3s
            s = s3s

        if index == 'ATT':
            m = m4s
            s = s4s
        if index != 'AHD':
            model = [ncpaths, fhpaths, accfpaths, ddpgpaths, maddpgpaths]
            model_name = ['NC', 'FH', 'CAAC', 'IAC', 'MADDPG']
            plt.ylabel(index)
            ax.bar([k for k in range(len(model_name))], m, yerr=s, align='center', alpha=0.8, ecolor='black',
                   capsize=10)
            plt.xticks([j for j in range(len(model_name))], [s for s in model_name], rotation=30)
            fig.savefig(index + ".pdf", bbox_inches='tight')
            plt.show()
        else:

            model = [fhpaths, accfpaths, ddpgpaths, maddpgpaths]
            model_name = ['FH', 'CAAC', 'IAC', 'MADDPG']
            m = m2s[1:]
            s = s2s[1:]
            plt.ylabel(index)
            ax.bar([k for k in range(len(model_name))], m, yerr=s, align='center', alpha=0.8, ecolor='black',
                   capsize=10)
            plt.xticks([j for j in range(len(model_name))], [s for s in model_name], rotation=30)
            fig.savefig(index + ".pdf", bbox_inches='tight')
            plt.show()


def vis_ablation():
    # visualize training performance 22
    seed = [4]  # 2
    weight = [2, 4, 6]

    model = ['accf']  # accflstm: no regularization on A; accflstmwi_: regularization on A with fp dependent weight
    model_name = ['CAAC']
    path1 = 'G:\\mcgill\\useful\\logs\\SG_22_1'
    path2 = 'G:\\mcgill\\useful\\log\\SG_22_1'
    path = [path1, path2]

    l = 0

    def grasp(i, j, last=0):
        reward1 = []
        reward2 = []
        reward3 = []
        minlen = 10000
        for s in seed:
            try:

                path_ = path[i] + 'accf' + str(weight[j]) + 'all' + str(s) + '.csv'
                train = pd.DataFrame(pd.read_csv(path_))

                reward1 += train.iloc[:]['reward1'].to_list()[-last:]
                reward2 += [np.clip(abs(a), 0, 3) for a in train.iloc[:]['action'].to_list()[
                                                           -last:]]  # [-last:]]train.iloc[:]['action'].to_list()[-last:]
                if len(reward3) > 0:
                    minlen = min(len(reward3[-1]), minlen)

                reward3.append(train.iloc[:]['reward'].to_list()[:])
            except:
                print(path_)
                continue

        return reward3

    tag = ['(a)', '(b)', '(c)', '(d)']
    for j in range(len(weight)):
        fig = plt.figure()
        ax = plt.gca()
        fig.set_size_inches((5, 4.2))
        colors = ['red', 'green', 'blue', 'green', 'm', 'pink', 'purple']
        for i in range(2):
            if i == 0:
                name = 'CAAC'
            else:
                name = 'CAAC without Regularizer'
            rewards = grasp(i, j)
            minlen = 16000
            for rr in rewards:
                if len(rr) < minlen:
                    minlen = len(rr)
            rewards_ = []
            for rr in rewards:
                rewards_.append(rr[:minlen])
            s = np.std(np.array(rewards_), axis=0)
            m = np.mean(np.array(rewards_), axis=0)

            plt.fill_between([i for i in range(minlen)], m - s,
                             m + s, interpolate=True, facecolor=colors[i], edgecolor=None, alpha=0.2)
            plt.plot(m, label=name, linewidth=2, color=colors[i])
        tag_fig(ax=ax, text=tag[j], offset=[-44, 20])
        plt.grid()
        if j == 0:
            plt.legend()
        plt.yticks(np.arange(round(min(m - s) - 0.02), max(m + s) + 0.02, 0.4))
        plt.xticks(np.arange(0, 250, 50))
        plt.xlabel('Episodes', size=22)
        if j == 0:
            plt.ylabel('Mean episode reward', size=22)
        fig.savefig('a' + str(j) + '.pdf', bbox_inches='tight')
        plt.show()


def vis_train_modelwise():
    # visualize training performance 22
    seed = [0,1,2,3,4,5]
    weight = [  2,4, 6]

    model = ['accf', 'ddpg',
             'maddpg', 'qmix']  # accflstm: no regularization on A; accflstmwi_: regularization on A with fp dependent weight
    model_name = ['CAAC', 'IAC', 'MADDPG','QMIX']
    path = 'G:\\mcgill\\useful\\log\\SG_22_1'
    l = 0

    def grasp(i, j, last=0):

        reward = []
        minlen = 10000
        for s in seed:
            try:

                path_ = path + model[i] + str(weight[j]) + 'all' + str(s) + '.csv'
                train = pd.DataFrame(pd.read_csv(path_))

                if len(reward) > 0:
                    minlen = min(len(reward[-1]), minlen)
                reward.append(train.iloc[:]['reward'].to_list()[:])
                # reward.append(train.iloc[:]['avg_hold'].to_list()[:])
                # reward.append(train.iloc[:]['qloss'].to_list()[:])
                # reward.append(train.iloc[:]['travel'].to_list()[:])
            except:
                print(path_)
                continue

        return reward

    colors = ['red', 'm', 'blue', 'green', 'm', 'green', 'purple']
    for j in range(len(weight)):
        fig = plt.figure()
        ax = plt.gca()
        fig.set_size_inches((5, 4.2))

        for i in range(len(model)):

            rewards = grasp(i, j)
            minlen = 16000
            for rr in rewards:
                if len(rr) < minlen:
                    minlen = len(rr)
            rewards_ = []
            for rr in rewards:
                rewards_.append(rr[:minlen])
            s = np.std(np.array(rewards_), axis=0)
            m = np.mean(np.array(rewards_), axis=0)

            plt.fill_between([i for i in range(minlen)], m - s,
                             m + s, interpolate=True, facecolor=colors[i], edgecolor=None, alpha=0.2)
            plt.plot(m, label=model_name[i], linewidth=2, color=colors[i])

        # plt.yticks(np.arange(-0.16, -0. , 0.04))
        plt.xticks(np.arange(0, 250, 50))
        # plt.yticks(np.arange(-1.6, -0.1, 0.4))
        tag_fig(ax=ax, text='(c)')
        # plt.yticks(np.arange(-1.0, 0.0, 0.1))
        # plt.ylim(top=0)  # ymax is your value
        # plt.ylim(bottom=-0.6)  # ymin is your value
        plt.grid()
        # plt.legend(loc='lower right',fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.xlabel('Episodes', size=22)
        # plt.ylabel('Policy loss', size=22)
        plt.ylabel('Mean episode reward', size=22)
        # fig.savefig(str(j)+'trainall.pdf', bbox_inches='tight')

        # plt.ylabel("Mean of -wCV"+r'$^2$' ,size=22)
        # fig.savefig(str(j)+'train1.pdf', bbox_inches='tight')
        #
        # plt.ylabel('Mean of -(1-w)a',size=22)
        # fig.savefig(str(j)+'train2.pdf', bbox_inches='tight')
        plt.show()

def relate_performance():
    # calculate performance
    seed = [0, 1, 2, 3, 4, 5]
    accfpaths = ['accf6all' + str(s) for s in seed]
    ddpgpaths = ['ddpg6all' + str(s) for s in seed]
    maddpgpaths = ['maddpg6all' + str(s) for s in seed]
    qmixpgpaths = ['qmix6all' + str(s) for s in seed]
    ncpaths = ['nc' + str(s) for s in seed]
    fhpaths = ['fc' + str(s) for s in seed]
    model = [ncpaths, fhpaths, accfpaths, ddpgpaths, maddpgpaths ,qmixpgpaths]
    model_name = ['NC', 'FH', 'CAAC', 'IAC', 'MADDPG' ,'QMIX']


    path = 'G:\\mcgill\\useful\\logt\\SG_28_1'

    ncrewards = []
    fhrewards = []
    accfrewards = []
    lstmaccfrewards = []
    ddpgrewards = []
    maddpgrewards = []

    def grasp(i, j, last=9):
        reward1 = []
        reward2 = []
        reward3 = []
        reward4 = []
        for s in range(len(model[i])):
            path_ = path + model[i][s] + '.csv'

            try:
                train = pd.DataFrame(pd.read_csv(path_))
                reward1 += train.iloc[:]['wait'].to_list()[-last:]
                reward2 += train.iloc[:]['avg_hold'].to_list()[
                           -last:]
                reward3 += train.iloc[:]['AOD'].to_list()[-last:]
                tt = train["travel"] - train["wait"]
                reward4 += tt.to_list()[-last:]
            except:
                # print(path_)
                continue

        return np.mean(reward1), np.std(reward1), np.mean(reward2), np.std(reward2), np.mean(reward3), np.std(
            reward3), np.mean(reward4), np.std(reward4)

    result = {}

    for i in range(len(model)):

        m1s = []
        m2s = []
        m3s = []
        m4s = []
        s1s = []
        s2s = []
        s3s = []
        s4s = []
        for s in seed:
            m1, s1, m2, s2, m3, s3, m4, s4 = grasp(i, s)
            m1s.append(m1)
            m2s.append(m2)
            m3s.append(m3)
            m4s.append(m4)
            s1s.append(s1)
            s2s.append(s2)
            s3s.append(s3)
            s4s.append(s4)
        if i==0:
            print('%s  AHD:%g | AWT:%g | AOD:%g| ATT:%g' % (
            model_name[i], np.mean(m2s), np.mean(m1s), np.mean(m3s), np.mean(m4s)))
            awt = np.mean(m1s)
            aod = np.mean(m3s)
            att = np.mean(m4s)
        else:
            print('%s Related to NC   AHD:%g | AWT:%g | AOD:%g| ATT:%g' % (
                model_name[i], np.mean(m2s), np.mean(m1s)-awt, np.mean(m3s)-aod, np.mean(m4s)-att))


def cal_performance():
    # calculate performance
    seed = [0, 1, 2, 3, 4, 5]
    accfpaths = ['accf6all' + str(s) for s in seed]
    ddpgpaths = ['ddpg6all' + str(s) for s in seed]
    maddpgpaths = ['maddpg6all' + str(s) for s in seed]
    qmixpgpaths = ['qmix6all' + str(s) for s in seed]
    ncpaths = ['nc' + str(s) for s in seed]
    fhpaths = ['fc' + str(s) for s in seed]
    model = [ncpaths, fhpaths, accfpaths, ddpgpaths, maddpgpaths,qmixpgpaths]
    model_name = ['NC', 'FH', 'CAAC', 'IAC', 'MADDPG','QMIX']


    path = 'G:\\mcgill\\useful\\logt\\SG_22_1'

    ncrewards = []
    fhrewards = []
    accfrewards = []
    lstmaccfrewards = []
    ddpgrewards = []
    maddpgrewards = []

    def grasp(i, j, last=9):
        reward1 = []
        reward2 = []
        reward3 = []
        reward4 = []
        for s in range(len(model[i])):
            path_ = path + model[i][s] + '.csv'

            try:
                train = pd.DataFrame(pd.read_csv(path_))
                reward1 += train.iloc[:]['wait'].to_list()[-last:]
                reward2 += train.iloc[:]['avg_hold'].to_list()[
                           -last:]
                reward3 += train.iloc[:]['AOD'].to_list()[-last:]
                tt = train["travel"] - train["wait"]
                reward4 += tt.to_list()[-last:]
            except:
                # print(path_)
                continue

        return np.mean(reward1), np.std(reward1), np.mean(reward2), np.std(reward2), np.mean(reward3), np.std(
            reward3), np.mean(reward4), np.std(reward4)

    result = {}

    for i in range(len(model)):

        m1s = []
        m2s = []
        m3s = []
        m4s = []
        s1s = []
        s2s = []
        s3s = []
        s4s = []
        for s in seed:
            m1, s1, m2, s2, m3, s3, m4, s4 = grasp(i, s)
            m1s.append(m1)
            m2s.append(m2)
            m3s.append(m3)
            m4s.append(m4)
            s1s.append(s1)
            s2s.append(s2)
            s3s.append(s3)
            s4s.append(s4)

        print('%s  AHD:%g | AWT:%g | AOD:%g| ATT:%g' % (
        model_name[i], np.mean(m2s), np.mean(m1s), np.mean(m3s), np.mean(m4s)))


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def vis_traj():
    path = 'G:\\mcgill\\useful\\vis\\'
    files = ['visnc\\', 'visfc\\', 'visaccf\\', 'visddpg\\', 'vismaddpg\\']#, 'visqmix\\']
    model = ['NC', 'FH', 'CAAC', 'IAC', 'MADDPG']#, 'QMIX']

    fig = plt.figure()
    fig.set_size_inches((12, 16))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1])
    total_obs = []
    total_segments = []
    for i in range(len(files)):
        path_ = path + files[i]
        if i < 1:
            ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharex=ax)
        ax.text(87000.0, 13, model[i], size=18, rotation=90)

        for file_name in glob.glob(path_ + '*.csv'):
            bus = file_name.split('_')[-1].split('.')[0]
            df = pd.DataFrame(pd.read_csv(file_name))

            # select the duplicated location row
            dp = df[df.duplicated('loc', keep=False)]
            # get the first and last idx of each duplicated series (i.e., get first&last index of a in [a,a,a,b,b,b])
            idx = dp.index[-dp.duplicated('loc', keep='last')].tolist() + dp.index[
                -dp.duplicated('loc', keep='first')].tolist()
            idx = sorted(idx)
            idx += [df.shape[0] - 1]
            times = df['time'][idx]
            locs = df['loc'][idx]
            ops = df['op'][idx]

            points = np.array([times, locs]).T.reshape(-1, 1, 2)

            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0., 0.6)
            lc = LineCollection(segments, cmap=plt.cm.rainbow, linewidth=2., norm=norm)
            lc.set_array(np.array(ops))
            total_obs.append(np.array(ops))
            ax.add_collection(lc)

        ax.autoscale()
        print(model[i])
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)
        # plt.tick_params(
        #     axis='y', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
        #     labelleft='off')
        if i < len(files) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    fig.subplots_adjust(right=0.8)
    plt.xlabel('Time step (sec)')
    # cbar_ax = fig.add_axes([0.12, 0.9, 0.62, 0.01])  [left, bottom, width, height]

    cbar_ax = fig.add_axes([0.92, 0.5, 0.03, 0.3])
    fig.colorbar(lc, cax=cbar_ax, orientation="vertical", format='%.2f').ax.yaxis.set_label_position('left')
    fig.text(0.06, 0.5, 'Distance (km)', ha='center', va='center', rotation='vertical')
    # plt.ylabel('Distance (km)')
    plt.subplots_adjust(hspace=.0)

    fig.savefig("traj.pdf", bbox_inches='tight')
    plt.show()


def vis_stw():
    path = 'G:\\mcgill\\useful\\logt\\SG_22_1'
    stops = [46, 58, 61, 46]
    stop_num = 46
    model = ['NC', 'FH', 'CAAC', 'IAC', 'MADDPG']#, 'QMIX']
    sto = [[] for _ in range(len(model))]
    sth = [[] for _ in range(len(model))]
    stw = [[] for _ in range(len(model))]

    seed = [0, 1, 2, 3, 4, 5]
    accfpaths = ['accf6all' + str(s) for s in seed]
    ddpgpaths = ['ddpg6all' + str(s) for s in seed]
    maddpgpaths = ['maddpg6all' + str(s) for s in seed]
    qmixpaths = ['qmix6all' + str(s) for s in seed]
    ncpaths = ['nc' + str(s) for s in seed]
    fhpaths = ['fc' + str(s) for s in seed]
    model = [ncpaths, fhpaths, accfpaths, ddpgpaths, maddpgpaths]#, qmixpaths]
    tag = ['(a)', '(b)', '(c)']
    for i in range(len(model)):
        for s in range(len(seed)):
            path_ = path + model[i][s] + 'res.csv'

            data = pd.DataFrame(pd.read_csv(path_))
            ls = []
            for hold in data['sth'].tolist():
                try:
                    ls.append(float(hold))
                except:
                    ls.append(float(hold.split('[')[1].split(']')[0]))
            sto[i].append(data['sto'].tolist())
            sth[i].append(ls)
            stw[i].append(data['stw'].tolist())
        sto[i] = np.array(sto[i]).reshape(-1, stop_num)
        sth[i] = np.array(sth[i]).reshape(-1, stop_num)[:, 2:stop_num - 2]
        stw[i] = np.array(stw[i]).reshape(-1, stop_num)

    def draw(data, name, i=0):
        fig = plt.figure()
        ax = plt.gca()
        tag_fig(ax=ax, text=tag[i])
        fig.set_size_inches((5, 4))
        nc = data[0]
        fh = data[1]
        accf = data[2]
        ddpg = data[3]
        maddpg = data[4]
        # qmix = data[5]
        if name != 'AHT':
            plt.fill_between([i for i in range(np.array(fh).shape[1])],
                             np.mean(np.array(nc), axis=0) - np.std(np.array(nc), axis=0),
                             np.mean(np.array(nc), axis=0) + np.std(np.array(nc), axis=0), interpolate=True,
                             color='black', alpha=0.1)
            plt.plot(np.mean(np.array(nc), axis=0), label='NC', linewidth=2, color='black')

        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(fh), axis=0) - np.std(np.array(fh), axis=0),
                         np.mean(np.array(fh), axis=0) + np.std(np.array(fh), axis=0), interpolate=True, color='green',
                         alpha=0.1)
        plt.plot(np.mean(np.array(fh), axis=0), label='FH', linewidth=2, color='green')

        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(ddpg), axis=0) - np.std(np.array(ddpg), axis=0),
                         np.mean(np.array(ddpg), axis=0) + np.std(np.array(ddpg), axis=0), interpolate=True, color='m',
                         alpha=0.1)
        plt.plot(np.mean(np.array(ddpg), axis=0), label='IAC', linewidth=2, color='m', linestyle='-.')

        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(maddpg), axis=0) - np.std(np.array(maddpg), axis=0),
                         np.mean(np.array(maddpg), axis=0) + np.std(np.array(maddpg), axis=0), interpolate=True,
                         color='red', alpha=0.1)
        plt.plot(np.mean(np.array(maddpg), axis=0), label='MADDPG', linewidth=2, color='red')

        # plt.fill_between([i for i in range(np.array(fh).shape[1])],
        #                  np.mean(np.array(qmix), axis=0) - np.std(np.array(qmix), axis=0),
        #                  np.mean(np.array(qmix), axis=0) + np.std(np.array(qmix), axis=0), interpolate=True,
        #                  color='red', alpha=0.1)
        # plt.plot(np.mean(np.array(qmix), axis=0), label='QMIX', linewidth=2, color='orange')

        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(accf), axis=0) - np.std(np.array(accf), axis=0),
                         np.mean(np.array(accf), axis=0) + np.std(np.array(accf), axis=0), interpolate=True,
                         color='blue', alpha=0.1)
        plt.plot(np.mean(np.array(accf), axis=0), label='CAAC', linewidth=2, color='blue')
        #
        # plt.fill_between([i for i in range(np.array(fh).shape[1])], np.mean(np.array(la), axis=0)-np.std(np.array(la), axis=0),
        #                  np.mean(np.array(la), axis=0)+np.std(np.array(la), axis=0), interpolate=True, color='green', alpha=0.1)
        # plt.plot(np.mean(np.array(la), axis=0), label='CAAC-LA', linewidth=2, color='green')

        # plt.yticks(np.arange(-1.0, 0.0, 0.1))
        # plt.ylim(top=0)  # ymax is your value
        # plt.ylim(bottom=-1)  # ymin is your value

        plt.grid()
        # plt.legend()
        if name == 'AWT':
            plt.legend(loc='upper left',ncol=2)
        # title = 'Line 22 NRH'
        # plt.title(title)
        plt.xlabel('Stops')
        if name != 'AOD':
            plt.ylabel(name + ' (sec)')
        else:
            plt.ylabel(name)
        fig.savefig(name + "stop_wise.pdf", bbox_inches='tight')
        plt.show()

    # vis sto
    draw(sto, 'AOD', i=2)
    # vis sth
    draw(sth, 'AHT', i=1)
    # vis stw
    draw(stw, 'AWT', i=0)
    return


def vis_tt():
    path = 'G:\\mcgill\\useful\\logt\\SG_22_1'
    seed = [0, 1, 2, 3, 4, 5]
    accfpaths = ['accf6all' + str(s) for s in seed]
    ddpgpaths = ['ddpg6all' + str(s) for s in seed]
    maddpgpaths = ['maddpg6all' + str(s) for s in seed]
    qmixpaths = ['qmix6all' + str(s) for s in seed]
    ncpaths = ['nc' + str(s) for s in seed]
    fhpaths = ['fc' + str(s) for s in seed]
    model = [ncpaths, fhpaths, accfpaths, ddpgpaths, maddpgpaths]#,qmixpaths]
    stt = [[] for _ in range(len(model))]
    for i in range(len(model)):
        for s in range(len(seed)):
            path_ = path + model[i][s] + 'arr.csv'

            data = pd.DataFrame(pd.read_csv(path_))

            data = np.array(data.iloc[:, 2:]).transpose()
            stt[i].append(list(data))

        stt[i] = np.array(stt[i])

    def draw(data, name):
        fig = plt.figure()
        ax = plt.gca()
        fig.set_size_inches((5, 4))

        tag_fig(ax=ax, text='(d)')

        nc = data[0]

        fh = data[1] - nc
        accf = data[2] - nc
        ddpg = data[3] - nc
        maddpg = data[4] - nc
        # qmix = data[5] - nc
        fh = fh[0].reshape(-1, 46)
        accf = accf[0].reshape(-1, 46)
        ddpg = ddpg[0].reshape(-1, 46)
        maddpg = maddpg[0].reshape(-1, 46)
        # qmix = qmix[0].reshape(-1, 46)
        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(fh), axis=0) - np.std(np.array(fh), axis=0),
                         np.mean(np.array(fh), axis=0) + np.std(np.array(fh), axis=0), interpolate=True, color='green',
                         alpha=0.1)
        plt.plot(np.mean(np.array(fh), axis=0), label='FH', linewidth=2, color='green')

        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(ddpg), axis=0) - np.std(np.array(ddpg), axis=0),
                         np.mean(np.array(ddpg), axis=0) + np.std(np.array(ddpg), axis=0), interpolate=True,
                         color='m', alpha=0.1)
        plt.plot(np.mean(np.array(ddpg), axis=0), label='IAC', linewidth=2, color='m', linestyle='-.')

        plt.fill_between([i for i in range(np.array(fh).shape[1])],
                         np.mean(np.array(maddpg), axis=0) - np.std(np.array(maddpg), axis=0),
                         np.mean(np.array(maddpg), axis=0) + np.std(np.array(maddpg), axis=0), interpolate=True,
                         color='red', alpha=0.1)
        plt.plot(np.mean(np.array(maddpg), axis=0), label='MADDPG', linewidth=2, color='red')

        # plt.fill_between([i for i in range(np.array(fh).shape[1])],
        #                  np.mean(np.array(qmix), axis=0) - np.std(np.array(qmix), axis=0),
        #                  np.mean(np.array(qmix), axis=0) + np.std(np.array(qmix), axis=0), interpolate=True,
        #                  color='pink', alpha=0.1)
        # plt.plot(np.mean(np.array(qmix), axis=0), label='QMIX', linewidth=2, color='orange')

        plt.fill_between([i for i in range(np.array(accf).shape[1])],
                         np.mean(np.array(accf), axis=0) - np.std(np.array(accf), axis=0),
                         np.mean(np.array(accf), axis=0) + np.std(np.array(accf), axis=0), interpolate=True,
                         color='blue',
                         alpha=0.1)
        plt.plot(np.mean(np.array(accf), axis=0), label='CAAC', linewidth=2, color='blue')

        # plt.fill_between([i for i in range(np.array(fh).shape[1])],
        #                  np.mean(np.array(la), axis=0) - np.std(np.array(la), axis=0),
        #                  np.mean(np.array(la), axis=0) + np.std(np.array(la), axis=0), interpolate=True, color='green',
        #                  alpha=0.1)
        # plt.plot(np.mean(np.array(la), axis=0), label='CAAC-LA', linewidth=2, color='green')
        loc = plticker.MultipleLocator(600.)
        ax.yaxis.set_major_locator(loc)

        plt.grid()
        # plt.legend()
        title = 'Line 22 NRH'
        # plt.title(title)
        plt.xlabel('Stops')
        plt.ylabel('Additional Travel Time (sec)')
        fig.savefig(name + "stop_wise.pdf", bbox_inches='tight')
        plt.show()

    # vis sto
    draw(stt, 'ATT')

    return

def vis_shares_effect():
    path11 = 'G:\\mcgill\\MAS\\gtfs_testbed_ml\\log\\mlnc2_22_1_'
    path13 = 'G:\\mcgill\\MAS\\gtfs_testbed_ml\\log\\mlzhou2_22_1_'
    path12 = 'G:\\mcgill\\MAS\\gtfs_testbed_ml\\log\\mlfc2_22_1_'

    path21 = 'G:\\mcgill\\MAS\\gtfs_testbed_ml\\log\\mlnc2_43_1_'
    path23 = 'G:\\mcgill\\MAS\\gtfs_testbed_ml\\log\\mlzhou2_43_1_'
    path22 = 'G:\\mcgill\\MAS\\gtfs_testbed_ml\\log\\mlfc2_43_1_'
    tail = '_5.csv'
    shares = [i for i in range(10)]
    model = ['nc','fc','fc+']
    paths = [path11,path12,path13]
    # paths = [path21, path22, path23]
    waits_comp = []
    waits_std_comp = []
    Evs = []
    c = ['b','g','r','c','m','y','k','orange','cyan','pink']
    w = -0.2
    for path in paths:
        waits_mean = []
        waits_std = []
        for s in shares:
            # try:
                data1 = pd.DataFrame(pd.read_csv(path+str(s)+tail))
                if s==0:
                    columns = data1.columns
                data1.columns = columns
                wait = data1['EV']
                # plt.scatter(data1['EV'],data1['wait'],label=str(s/10.),c=c[shares.index(s)])
                # plt.show()
                s = np.std(np.array(wait), axis=0)
                m = np.mean(np.array(wait), axis=0)
                waits_mean.append(m)
                waits_std.append(s)
        plt.bar(np.arange(len(waits_mean))+w, waits_mean, yerr=waits_std, align='center', alpha=0.5, ecolor='black',
                capsize=1,color=c[paths.index(path)], width=0.2,label=model[paths.index(path)])
        w+=0.2
    plt.legend()
    plt.show()
        # except:
        #     print('No '+path1+str(s)+tail)
    # plt.legend()
    # plt.show()
    plt.bar(np.arange(len(waits_mean)), waits_mean, yerr=waits_std, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.show()


    return

COLORS = ['green','red','blue','m','pink','cyan','orange','brown','black']
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
Num_line_styles = len(LINE_STYLES)
Num_colors = len(COLORS)
def ml_comp():
    seed = [0,1,2]
    ctrl = [0,1,2,3,4,5,6,3,4,5,6]
    sh = [0.0,0.2,0.4,0.6,0.8]#'RS'
    sc = [0,1]
    route = [22]
    models = ['nc','fc','zhou' ,'caac','caac','caac','caac' ,'caac_2']#,'caac_2','caac_2','caac_2','caac_2']
    model_name = ['NC','FC','FC+' ,'CAAC_ir_io','CAAC_ir_mo','CAAC_mr_io','CAAC_mr_mo','CAAC_mr_mo2']#,'CAAC_ir_io2','CAAC_ir_mo2','CAAC_mr_io2','CAAC_mr_mo2']


    def vis_training_performance(r,item,w=2,sc=0):
        seed = [0,1,2 ]
        seed = [19, 21, 33]
        models = [ 'caac', 'caac',   'caac_2',   'caac_2', 'caac_2']
        ctrl_ir = [3,4]#,3,4]
        ctrl_mr = [5,6, ]#,5,6]

        sh = 'RS'
        ctrl = [3,4,5,6]
        model_name = [ 'CAAC_ir_io', 'CAAC_ir_mo','CAAC_mr_io', 'CAAC_mr_mo',]
                       # 'CAAC_ir_io_emb', 'CAAC_ir_mo_emb','CAAC_mr_io_emb', 'CAAC_mr_mo_emb']
        models = [ '0-caac', '0-caac', '0-caac', '0-caac',]
                   # '1-caac', '1-caac',  '1-caac',  '1-caac']
        width = -0.2
        result_mean_model = []
        result_std_model = []
        fig = plt.figure()
        ax = plt.gca()
        # fig.set_size_inches((5, 4.2))
        color_index = 0
        style_index = 0
        model_index = 0
        base = 'G:\\mcgill\\useful\\mllog\\'
        for m in range(len(models)):
            result_mean = []
            result_std = []
            result = []

            reward1 = []
            reward2 = []
            reward3 = []
            result_all = []

            for sss in range(len(seed)):
                try:
                    path = base + 'ml_sc_{}_w_{}_m_{}_ctrl_{}_sh_{}_a_1_R{}_1_{}'. \
                        format(sc, w, models[m], ctrl[m], sh , r, seed[sss]) + '.csv'
                    data = pd.DataFrame(pd.read_csv(path))
                    result.append(data[item].values)

                    result_all.append(data[item].values)
                    reward1.append(data['reward1'].values)
                    reward2.append(data['reward2'].values)
                    reward3.append(data['reward3'].values)
                except:
                    print('Missing', path)
                    pass

            # if np.isnan(reward3).any() == False:
            #     plt.scatter(np.array(reward1).reshape(-1, ), np.array(result_all).reshape(-1, ))
            #     plt.xlabel('reward3')
            #     plt.ylabel('wait')
            #     plt.show()

            try:

                std = np.std(np.array(result), axis=0)
                mean = np.mean(np.array(result), axis=0)
                print("model{}: {}+-{}".format(model_name[model_index],mean,std))
                plt.fill_between([i for i in range(std.shape[0])], mean -std,
                                 mean + std, interpolate=True, facecolor=COLORS[color_index], edgecolor=None, alpha=0.2)
                plt.plot(mean, label=model_name[model_index], linewidth=2, color=COLORS[color_index], linestyle=LINE_STYLES[style_index])

            except:
                print('Wrong',path)
            model_index+=1
            color_index+=1
            if color_index>Num_colors:
                color_index = 0
                style_index+=1


        tag_fig(ax=ax, text=str(r), offset=[-43, 26])
        plt.xlabel('Training episode')
        r_dict = {22: 'R1', 43: 'R2'}
        ylabel_dict = {'avg_hold': 'Average holding time (sec)', 'system_wait': 'Average waiting time (sec)'
            , 'wait': 'Average waiting time of ' + r_dict[r] + ' (sec)',
                       'travel': 'Average travel time of ' + r_dict[r] + ' (sec)'
            , 'system_travel': 'Average travel time (sec)', 'AOD': 'AOD'}
        plt.ylabel(item)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        # plt.tight_layout()
        plt.legend()
        # fig.savefig(item +str(r)+ "training.png", bbox_inches='tight')
        plt.show()

        return


    def stop_wise_comp(r,w=6,sc=0,is_highlight_share=False):
        base = 'G:\\mcgill\\useful\\w11\\mllogt\\'
        shared_stops = {22:[10,22],43:[24,36]}
        stop_nums = { 22 :46, 43 :61}
        sto = [[] for _ in range(len(models))]
        sth = [[] for _ in range(len(models))]
        stw = [[] for _ in range(len(models))]
        stop_num = stop_nums[r]
        tag = ['(a)', '(b)', '(c)']
        for m in range(len(models)):
            for s in range(len(sh)):
                result = []
                for sss in range(len(seed)):
                    try:
                        path = base + 'ml_sc_{}_w_{}_m_{}_ctrl_{}_sh_{}_a_1_R{}_1_{}'. \
                            format(sc, w, models[m], ctrl[m], sh[s], r, seed[sss]) + 'res.csv'
                        data = pd.DataFrame(pd.read_csv(path))
                        ls = []
                        for hold in data['sth'].tolist():
                            try:
                                ls.append(float(hold))
                            except:
                                ls.append(float(hold.split('[')[1].split(']')[0]))

                        sto[m].append(data['sto'].tolist())
                        sth[m].append(ls)
                        stw[m].append(data['stw'].tolist())
                    except:
                        pass

            sto[m] = np.array(sto[m]).reshape(-1, stop_num)
            sth[m] = np.array(sth[m]).reshape(-1, stop_num)[:, 2:stop_num - 2]
            stw[m] = np.array(stw[m]).reshape(-1, stop_num)

        def stop_wise_draw(data, name, i=0):
            fig = plt.figure()
            ax = plt.gca()
            tag_fig(ax=ax, text=tag[i])
            fig.set_size_inches((5, 4))
            color_index = 0
            style_index = 0
            model_index = 0
            for k in range(len(data)):
                if is_highlight_share==False:
                    b1 = 0
                    b2 = np.array(data[k]).shape[1]
                else:
                    b1 = shared_stops[r][0]
                    b2 =shared_stops[r][1]
                if name != 'AHT' or (name=='AHT' and k!=0):
                    try:
                        plt.errorbar(np.arange(np.array(data[k]).shape[1])[b1:b2+1], np.mean(np.array(data[k]), axis=0)[b1:b2+1],
                                     yerr=np.std(np.array(data[k]), axis=0)[b1:b2+1], ecolor=COLORS[color_index],
                                     color=COLORS[color_index],
                                     elinewidth=1, label=model_name[model_index], capsize=3, linewidth=2,
                                     linestyle=LINE_STYLES[style_index])
                    except:
                        color_index = 0
                        style_index += 1

                        plt.errorbar(np.arange(np.array(data[k]).shape[1])[b1:b2+1], np.mean(np.array(data[k]), axis=0)[b1:b2+1],
                                     yerr=np.std(np.array(data[k]), axis=0)[b1:b2+1], ecolor=COLORS[color_index],
                                     color=COLORS[color_index],
                                     elinewidth=1, label=model_name[model_index], capsize=3, linewidth=2,
                                     linestyle=LINE_STYLES[style_index])

                color_index += 1
                model_index += 1

            plt.grid()
            # plt.legend()
            # if name == 'AWT':
            plt.legend(loc='upper left', ncol=2)
            # title = 'Line 22 NRH'
            # plt.title(title)
            plt.xlabel('Stops')
            if name != 'AOD':
                plt.ylabel(name + ' (sec)')
            else:
                plt.ylabel(name)
            if is_highlight_share==True:
                fig.savefig(str(r)+name + "stop_wise_share.png", bbox_inches='tight')
            else:
                fig.savefig(str(r) + name + "stop_wise.png", bbox_inches='tight')
            plt.show()

        # vis sto
        stop_wise_draw(sto, 'AOD', i=2)
        # vis sth
        stop_wise_draw(sth, 'AHT', i=1)
        # vis stw
        stop_wise_draw(stw, 'AWT', i=0)
        return
    def vis_test_shares(sc,w,r,item):

        seed = [ 0,1,2]
        sh = [0.0, 0.2, 0.4, 0.6, 0.8]  # 'RS'
        sc = 0
        weight = 6
        route = [22]

        # ctrl = [0,1,2,3,4,5,6,3,4,5,6]
        # model_name = [ 'NC',  'FC', 'FC+',
        #                'CAAC_ir_io', 'CAAC_ir_mo','CAAC_mr_io', 'CAAC_mr_mo',
        #                'COM1_ir_io', 'COM1_ir_mo', 'COM1_mr_io', 'COM1_mr_mo']
        # models = ['0-nc','0-fc', '0-zhou',
        #           '0-caac', '0-caac', '0-caac', '0-caac',
        #           '0-com1','0-com1','0-com1','0-com1']
        #            # '1-caac', '1-caac',  '1-caac',  '1-caac']
        ctrl = [0,1,2,3,4,5,6 ]
        model_name = [ 'NC',  'FC', 'FC+',
                       'CAAC_ir_io', 'CAAC_ir_mo','CAAC_mr_io', 'CAAC_mr_mo']#, 'CAAC_mr_mo_w']
        models = ['0-nc','0-fc', '0-zhou',
                  '0-caac', '0-caac', '0-caac', '0-caac']#, '0-caac']

        width = -0.2
        result_mean_model = []
        result_std_model = []
        fig = plt.figure()
        ax = plt.gca()
        # fig.set_size_inches((5, 4.2))
        color_index = 0
        style_index = 0
        model_index = 0

        base = "F:\\bus bunching multiline control\\gtfs_testbed_ml\\mllogt\\"
        for m in range(len(models)):
            result_mean = []
            result_std = []

            for s in range(len(sh)):
                result = []
                for sss in range(len(seed)):
                    try:
                        path = base + 'ml_sc_{}_w_{}_m_{}_ctrl_{}_sh_{}_a_1_R{}_1_{}'. \
                            format(0, w, models[m], ctrl[m], str(sh[s]), r, seed[sss]) + '.csv'
                        data = pd.DataFrame(pd.read_csv(path))
                        result.append(data[item].values[:] )

                    except:
                        path = base + 'ml_sc_{}_w_{}_m_{}_ctrl_{}_sh_{}_a_1_R{}_1_{}'. \
                            format(1, w, models[m], ctrl[m], str(sh[s]), r, seed[sss]) + '.csv'
                        data = pd.DataFrame(pd.read_csv(path))
                        result.append(data[item].values[:])
                try:
                    std = np.std(np.array(result).reshape(-1,))
                except:
                    print()
                mean = np.mean(np.array(result).reshape(-1,))
                result_mean.append(mean)
                result_std.append(std)

            if color_index>=Num_colors:
                color_index = 0
                style_index += 1
            if np.isnan(result_mean).any() or np.sum(result_mean) == 0:
                pass
            else:
                plt.errorbar(np.arange(len(result_mean)), result_mean, yerr=result_std, ecolor=COLORS[color_index],
                             color=COLORS[color_index],
                             elinewidth=1, label=model_name[model_index], capsize=3, linewidth=2,
                             linestyle=LINE_STYLES[style_index])
            print( model_name[model_index],'item', item, ' M: ',result_mean,' S:',result_std)
            plt.xticks(np.arange(len(result_mean))  , ('0', '20', '40','60', '80'))
            result_mean_model.append(result_mean)
            result_std_model.append(result_std)
            color_index+=1
            model_index+=1

        # tag_fig(ax=ax, text=item, offset=[-43, 26])
        plt.xlabel('Shared pax in common corridor (%)')
        r_dict = {22:'R1',43:'R2'}
        ylabel_dict = {'avg_hold':'Average holding time (sec)', 'system_wait':'Average waiting time (sec)'
            ,'wait':'Average waiting time of '+r_dict[r] + ' (sec)' ,'travel':'Average travel time of '+r_dict[r] + ' (sec)'
            ,'system_travel':'Average travel time (sec)','AOD':'AOD','reward3':'reward signal 3'
            , 'reward1': 'reward signal 1'}
        plt.ylabel(ylabel_dict[item])
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.legend()
        # plt.tight_layout()
        fig.savefig(str(r)+'_'+item + "w1.png", bbox_inches='tight')
        plt.show()

    # 'system_wait' 	bunching	ploss	qloss	reward	reward1
    # reward2	reward3 avg_hold	action	wait	travel	delay	AOD	EV
    # system_wait	system_travel system_aod
    # vis_training_performance(r=22,item='reward',w=6)
    vis_test_shares(sc=0,w=6,r=43,item='system_wait') # the agent with higher reward 1 is better
    stop_wise_comp(22,is_highlight_share=True)
    return

if __name__ == '__main__':
    ml_comp()
    vis_shares_effect()
    # sim_vs_real()
    # vis_stw()
    # vis_traj()
    # test_comp()
    # cal_performance()
    # relate_performance()
    # vis_bb()
    # vis_tt()
    # vis_train_modelwise()
    # vis_ablation()