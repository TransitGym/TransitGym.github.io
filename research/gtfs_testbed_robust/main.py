import argparse
import os, sys
from sim import Sim_Engine
from sim import util as U
import numpy as np
import pandas as pd
import copy
import time
from random import gauss,randint
from random import seed
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
parser = argparse.ArgumentParser(description='param')
parser.add_argument("--seed", type=int, default=200)             # random seet
parser.add_argument("--model", type=str, default='iqncmadm')         # ddpg, iqnc,iqncm,iqncmadm
parser.add_argument("--st", type=int, default=0)               # st=1 : spatial-temporal event order graph
parser.add_argument("--data", type=str, default='SG_22_1')       # used data prefix
parser.add_argument("--para_flag", type=str, default='SG_22_1')  # stored parameter prefix
parser.add_argument("--episode", type=int, default=5)          # training episode
parser.add_argument("--overtake", type=int, default=0)           # overtake=0: not allow overtaking between buses
parser.add_argument("--arr_hold", type=int, default=1)           # arr_hold=1: determine holding once bus arriving bus stop
parser.add_argument("--train", type=int, default=0)              # train=1: train the model
parser.add_argument("--restore", type=int, default=0)            # restore=1: restore the model
parser.add_argument("--all", type=int, default=1)                # all=0 for considering only forward/backward buses' all=1 for all buses
parser.add_argument("--vis", type=int, default=0)                # vis=1 to visualize bus trajectory in test phase
parser.add_argument("--weight", type=int, default=20)             # /10 weight for action penalty
parser.add_argument("--control", type=int, default=2)            # 0 for no control;  1 for FH; 2 for RL
parser.add_argument("--share_scale", type=int, default=1)        # 0 non-share, 1 route-share
parser.add_argument("--mode", type=str, default='n')             # confidence on self contribution: "n" for think nothing,"cf" for confidence,ncf for "unconfidence
parser.add_argument("--accident_rate", type=int, default=1)      # /100 control the serverity of traffic state anomaly
parser.add_argument("--demand_impulse_rate", type=int, default=20)       # /10 control the serverity of demand anomaly
parser.add_argument("--sensor_error_rate", type=int, default=1)        # /10 control the serverity of commnuication anomaly
parser.add_argument("--demand_abnormal", type=int, default=0)
parser.add_argument("--state_abnormal", type=int, default=0)
args = parser.parse_args()
if args.model == 'iqncmadm':
    from model.IQNCM_adam import Agent
if args.model=='accf':
    from model.ACCF import Agent
if args.model=='ddpg':
    from model.DDPG import Agent
if args.model=='maddpg':
    from model.MADDPG import Agent
if args.model=='coma':
    from model.COMA import Agent
if args.model=='qmix':
    from model.QMIX import Agent
if args.model=='sdpg':
    from model.SDPG import Agent
if args.model=='iqnc':
    from model.IQNC import Agent
if args.model=='iqncm':
    from model.IQNCM import Agent
print('model',args.model)
model = args.model
if args.all==1:
    model = args.model +'all'
if args.share_scale==0:
    model+='0'


def train(args ):
    stop_list, pax_num = U.getStopList(args.data)
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(args.data)
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))

    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    stop_list_ = copy.deepcopy(stop_list)
    bus_routes_ = copy.deepcopy(bus_routes)
    bus_list_ = copy.deepcopy(bus_list)
    print('init...')
    agents = {}
    eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                            dispatch_times=dispatch_times,
                            demand=0, simulation_step=simulation_step, route_list=route_list,
                            hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                            share_scale=args.share_scale, weight=args.weight)

    if model != '':
        # bus_list = eng.bus_list
        # bus_stop_list = eng.busstop_list
        # U.demand_analysis(eng)
        # non share
        if args.share_scale == 0:
            for k, v in eng.bus_list.items():
                state_dim = 4

                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                                  seed=args.seed,mode=args.mode)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in eng.route_list.items():
                state_dim = 4
                agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=bus_list,
                                  seed=args.seed,mode=args.mode)
                agents[k] = agent
    now_ = time.time()
    for ep in range(args.episode):
        now_ep = time.time()

        stop_list_ = copy.deepcopy(stop_list)
        bus_routes_ = copy.deepcopy(bus_routes)
        bus_list_= copy.deepcopy(bus_list)

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight)

        eng.sensor_error_rate = np.random.randint(0,20) /100. # control the serverity of commnuication anomaly
        eng.demand_impulse_rate = np.random.randint(0,30)/10. # control the serverity of demand anomaly
        eng.accident_rate = np.random.randint(0,30) / 100.  # control the serverity of traffic state anomaly

        # Pareto sampling scheme

        for _, stop in stop_list_.items():
            stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate)
            # stop.demand_impulse_rate = np.random.randint(10,50)/10. # simulate occuring rate of demand impulse

        eng.busstop_list = stop_list_
        eng.agents = agents
        if ep > 0:
            if memory_copy != None:
                eng.GM = memory_copy
            for bid, b in eng.bus_list.items():
                eng.GM.temp_memory[bid] = {'s': [], 'a': [], 'fp': [], 'r': [], 'd': []}

        if args.restore == 1 and args.control > 1:
            for k, v in agents.items():
                print(str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(model) + str('_'))
                v.load(str(args.para_flag) + str('_') + str(args.share_scale)+ str('_') + str(args.weight) + str('_') + str(model) + str('_') + str(args.mode) + str('_') )

        Flag = True
        begin = time.time()
        while Flag:
            if eng.weight>1:
                eng.weight = np.random.randint(10.,50.)/10.
            Flag = eng.sim()
        #################### set meta output ##########################
        # if ep in [1, 2,3,100, 200, 300,400,450,500 ]:
        #     for k, v in agents.items():
        #         abspath = os.path.abspath(os.path.dirname(__file__))
        #         name = abspath + "/meta"+str(args.seed)+"/"
        #         eng.record_dt()
        #         for kk,vv in v.meta_weight_record.items():
        #             record = vv
        #             np.save(name+'{}_{}'.format(kk,ep), record)
        #         break
        for k, v in agents.items():
            v.meta_weight_record = {}
        ################################################################
        ploss_log = [0]
        qloss_log = [0]
        mloss_log = [0]
        if args.control > 1 and args.restore == 0:
            for _ in range(3):
                if ep >= 0:
                    ploss, qloss,mloss, l_f = eng.learn(episode=ep)
                    if l_f == True:
                        qloss_log.append(qloss)
                        ploss_log.append(ploss)
                        if mloss!=None:
                            mloss_log.append(mloss)

            if  ep==args.episode - 2 and args.restore == 0:
                # store model
                for k, v in agents.items():
                    v.save(str(args.para_flag) + str('_') + str(args.share_scale)+ str('_') + str(args.weight) + str('_') + str(model) + str('_') + str(args.mode) + str('_') )

        if args.control > 1:
            memory_copy = eng.GM
        else:
            memory_copy = None
        try:
            log = eng.cal_statistic(name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(model) + str('_'),
                                    train=args.train)

            abspath = os.path.abspath(os.path.dirname(__file__))
            name = abspath + "/rolog/" + args.data + args.model + str('_') + args.mode
            name += str(int(args.weight))
            if args.all == 1:
                name += 'all'

            U.train_result_track(eng=eng, ep=ep, qloss_log=qloss_log, ploss_log=ploss_log, log=log, name=name, seed=args.seed,mloss_log=mloss_log)
            # U.visualize_trajectory(engine=eng, name=name + '_' + str(args.para_flag) + str('_'))

            print('Total time cost:%g sec Episode time cost:%g' % (time.time() - now_,time.time() - now_ep))
            print('')
        except Exception as e:
            print(e)
        eng.close()

def evaluate(args):
    stop_list, pax_num = U.getStopList(args.data)
    print('Stops prepared, total bus stops: %g' % (len(stop_list)))
    bus_routes = U.getBusRoute(args.data)
    print('Bus routes prepared, total routes :%g' % (len(bus_routes)))
    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    agents = {}
    if model != '':

        stop_list_ = copy.deepcopy(stop_list)
        bus_routes_ = copy.deepcopy(bus_routes)
        bus_list_= copy.deepcopy(bus_list)
        eng = Sim_Engine.Engine(bus_list = bus_list, busstop_list=stop_list_, control_type=args.control,dispatch_times=dispatch_times,
                                demand=0 ,simulation_step=simulation_step,route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight, demand_abnormal = args.demand_abnormal,
                                state_abnormal = args.state_abnormal)

        bus_list = eng.bus_list
        bus_stop_list = eng.busstop_list
        # demand_analysis(eng)
        # non share
        if args.share_scale == 0:
            for k, v in eng.bus_list.items():
                state_dim = 4
                agent = Agent(state_dim=state_dim, name=k, n_stops=len(eng.busstop_list), buslist=eng.bus_list, seed=args.seed,mode=args.mode)
                agents[k] = agent

        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in eng.route_list.items():
                state_dim = 4
                if args.model == 'accf':
                    agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                                  seed=args.seed ,mode=args.mode)
                else:
                    agent = Agent(state_dim=state_dim, name='', n_stops=len(eng.busstop_list), buslist=eng.bus_list,
                                  seed=args.seed,mode=args.mode)
                agents[k] = agent

    now_ = time.time()

    eng.demand_impulse_rate = (args.demand_impulse_rate) / 10.  # np.random.randint(0, 60) / 10.  # control the serverity of demand anomaly

    stop_lists = []
    for ep in range(args.episode):
        stop_list_ = copy.deepcopy(stop_list)
        for _, stop in stop_list_.items():
            stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate)
        stop_lists.append(stop_list_)
    for ep in range(args.episode):
        stop_list_ = stop_lists[ep]#copy.deepcopy(stop_list)
        bus_routes_ = copy.deepcopy(bus_routes)
        bus_list_= copy.deepcopy(bus_list)
        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                share_scale=args.share_scale, weight=args.weight,demand_abnormal = args.demand_abnormal,
                                state_abnormal = args.state_abnormal)
        eng.accident_rate = (args.accident_rate ) / 10.   # np.random.randint(0, 4) / 1000.  # control the serverity of traffic state anomaly
        eng.sensor_error_rate = args.sensor_error_rate / 100.  # np.random.randint(0, 20) / 100.  # control the serverity of commnuication anomaly
        eng.agents = agents
        pax_num = [ ]
        for _, stop in stop_list_.items():
            # stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate)
            # print(len(stop.dest))
            # print('read pax time cost:%g sec' % (time.time() - now_))
            pax_num.append(len(stop.pre_gen_pax_list))
        print('pax num',sum(pax_num))
        # plt.plot(pax_num)
        # plt.show()
        eng.busstop_list = stop_list_
        # print('read pax time cost:%g sec' % (time.time() - now_))
        if ep > 0:
            if memory_copy != None:
                eng.GM = memory_copy
            for bid, b in eng.bus_list.items():
                eng.GM.temp_memory[bid] = {'s': [], 'a': [], 'fp': [], 'r': [], 'd': []}
        s = str(args.para_flag) + str('_') + str(args.share_scale)+ str('_') + str(args.weight) + str('_') + str(model)  + str('_')+args.mode+ str('_')
        if args.restore == 1 and args.control > 1 :
            for k, v in agents.items():
                v.load(s)

        Flag = True
        begin = time.time()
        while Flag:
            Flag = eng.sim()

        if args.control == 1:
            memory_copy = eng.GM
        else:
            memory_copy = None

        log = eng.cal_statistic(
            name=str(args.para_flag) + str('_') + str(args.share_scale) + str('_') + str(model) + str('_'),
            train=args.train)

        abspath = os.path.abspath(os.path.dirname(__file__))
        path = abspath+ "/rologt/{}_{}_{}/".format(str(args.sensor_error_rate),str(args.demand_impulse_rate),str(args.accident_rate))

        if args.demand_abnormal>=1:
            path = abspath + "/rologt/{}_{}_{}_".format(str(args.sensor_error_rate), str(args.demand_impulse_rate),
                                                        str(args.accident_rate))

            path+='D'+str(args.demand_abnormal)+'/'
        if args.state_abnormal >= 1:
            path = abspath + "/rologt/{}_{}_{}_".format(str(args.sensor_error_rate), str(args.demand_impulse_rate),
                                                        str(args.accident_rate))

            path+='S'+str(args.state_abnormal)+'/'
        try:
            os.makedirs(path)
        except:
            pass
        if args.control == 0:
            name = path + args.data + 'nc'

        if args.control == 2:
            print('MODEL', args.model)
            name =path + args.data + args.model + str('_')+args.mode
            name += str(int(args.weight))
            if args.all == 1:
                name += 'all'

        if args.control == 1:
            name = path + args.data + 'fc'
            print('MODEL','fc')

        U.train_result_track(eng=eng, ep=ep, qloss_log=[0], ploss_log=[0], log=log, name=name,
                             seed=args.seed)

        # if args.vis == 1 and ep==len(rs)-1 and args.seed==1 and args.data=='SG_22_1':
        print('time cost:%g sec' % (time.time() - now_))
        if args.vis == 1:
            if args.control == 0:
                folder = abspath + "/vis/visnc/"
            if args.control == 1:
                folder = abspath + "/vis/visfc/"
            if args.control == 2:
                if args.model=='ddpg':
                    folder = abspath + "/vis/vis" + args.model+'/'
                else:
                    folder = abspath + "/vis/vis" + args.model+args.mode+'/'
            try:
                os.makedirs(folder)
            except:
                print(folder, ' has existed')
            U.visualize_trajectory(engine=eng, name=folder + '/' + str(args.data) + str('_'))
            break
        print(ep,'-----------------------------------------------------------------------------')



        # break
        print(' ')
        eng.close()
if __name__ == '__main__':

    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train==1:
        train(args)

    else:
        evaluate(args)






