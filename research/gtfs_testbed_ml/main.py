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
from functools import reduce
import logging

parser = argparse.ArgumentParser(description='param')
parser.add_argument("--seed", type=int, default=42)              # random seed
parser.add_argument("--control", type=int, default=0)           # [0] no control {nc}
                                                                # [1] FH {fc}
                                                                # [2] FH for multiple line {zhou}
                                                                # [3] RL in iso reward x iso obs.
                                                                # [4] RL in iso reward x ml obs.
                                                                # [5] RL in ml reward x iso obs.
                                                                # [6] RL in ml reward x ml obs.
                                                                # [7] RL in ml reward x ml obs x adpt w
parser.add_argument("--model", type=str, default='caac')        # nc,fc,zhou,caac ddpg maddpg
parser.add_argument("--data",   default=['SG_22_1','SG_43_1'])       # used data prefix
parser.add_argument("--para_flag", type=str, default='ml')       # stored parameter prefix
parser.add_argument("--episode", type=int, default=500)          # training episode
parser.add_argument("--overtake", type=int, default=0)           # overtake=0: not allow overtaking between buses
parser.add_argument("--arr_hold", type=int, default=1)           # arr_hold=1: determine holding once bus arriving bus stop
parser.add_argument("--train", type=int, default=1)              # train=1: train the model
parser.add_argument("--restore", type=int, default=0)            # 1: restore the model
parser.add_argument("--all", type=int, default=1)                # 0: for considering only forward/backward buses. 1: for all buses
parser.add_argument("--vis", type=int, default=0)                # 1: to visualize bus trajectory in test phase
parser.add_argument("--weight", type=int, default=2)             # weight for action penalty
parser.add_argument("--share_scale", type=int, default=0)        # 0: share. 1: route-share
parser.add_argument("--shares", type=int, default=-2)             # share rate of demand in common corridor
parser.add_argument("--stop_embedding", type=int, default=0)
args = parser.parse_args()

if args.model in ['caac','caac_2',]:
    from model.CAAC import Agent
if args.model=='ddpg':
    from model.DDPG import Agent
if args.model=='maddpg':
    from model.MADDPG import Agent
if args.model=='coma':
    from model.COMA import Agent
if args.model=='qmix':
    from model.QMIX import Agent
if args.model == 'com1':
    from model.CAAC_COM1 import Agent

args.model = str(args.stop_embedding)+'-'+args.model
abspath = os.path.abspath(os.path.dirname(__file__))


if args.shares < 0:
    s = 'RS'
else:
    s = str(args.shares)



def train(args ):
    stop_list, route_list  = {},{}
    num_stops = []
    save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                     + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' + s \
                     + '_a_' + str(args.all) + '_'
    bus_routes = {}
    for data in args.data:
        d = data.split('_')
        r = d[1]+'_'+d[2]

        new_stop_list, pax_num = U.getStopList(data)
        new_bus_routes = U.getBusRoute(data)
        bus_routes[r] = new_bus_routes[r]
        route_list = reduce(lambda x, y: dict(x, **y), (route_list, new_bus_routes))
        num_stops.append(len(new_stop_list))

        print('%s buses prepared :%g' % (data,len(route_list[r])))
        print('%s stops prepared :%g' % (data, len(new_stop_list)))
        for k, v in new_stop_list.items():
            if k not in stop_list:
                stop_list[k] = v

            stop_list[k].dest_route[r] = v.dest
            stop_list[k].routes.append(r)
            stop_list[k].loc_route[r] = v.loc
            stop_list[k].prev_stop_route[r] = v.prev_stop
            stop_list[k].next_stop_route[r] = v.next_stop
            stop_list[k].serving_period = [0 for _ in range(86400)]

    print('Shared stops: %d'%(np.sum(num_stops)-len(stop_list)))

    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    stop_list_ = copy.deepcopy(stop_list)
    bus_routes_ = copy.deepcopy(bus_routes)
    bus_list_ = copy.deepcopy(bus_list)
    agents = {}

    stop_list_ = copy.deepcopy(stop_list)
    bus_routes_ = copy.deepcopy(bus_routes)
    bus_list_ = copy.deepcopy(bus_list)

    if args.control > 2:


        bus_list = bus_list_
        bus_stop_list = stop_list_
        # U.demand_analysis(eng)

        if args.control % 2 == 0:
            state_dim = 8
        else:
            state_dim = 6
        if args.control==7:
            state_dim = 11
        # share all
        if args.share_scale == 0:
            agents[0] = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed,is_stop_embedding=args.stop_embedding)
            print('agents[0] state dim:', agents[0].state_dim)


        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in bus_routes_.items():

                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                                  seed=args.seed, is_stop_embedding=args.stop_embedding)
                agents[k] = agent
    now_ = time.time()
    for ep in range(args.episode):

        if args.shares < 0:
            shares = np.random.randint(0, 90)
        else:
            shares = args.shares

        stop_list_ = copy.deepcopy(stop_list)
        bus_routes_ = copy.deepcopy(bus_routes)
        bus_list_ = copy.deepcopy(bus_list)

    # stop.demand_impulse_rate = np.random.randint(10,50)/10. # simulate occuring rate of demand impulse

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,  policy_types=len(args.model.split('_')),
                                share_scale=args.share_scale, weight=args.weight,shares=shares)
        eng.demand_impulse_rate = np.random.randint(0,40)/10. # control the serverity of demand anomaly
        eng.accident_rate =  np.random.randint(0,40) / 100.  # control the serverity of traffic state anomaly
        for _, stop in stop_list_.items():
            for r in stop.routes:
                stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate,route=r,shares=eng.shares,share_stops=eng.shared_stops)
        print('------------------------  Epoch: %d ----------------------'%ep)
        now_ep = time.time()

        eng.busstop_list = stop_list_
        eng.agents = agents

        if ep > 0:
            if memory_copy != None:
                eng.GM = memory_copy
            if PW_copy != None:
                eng.PW = PW_copy
            for bid, b in eng.bus_list.items():
                eng.GM.temp_memory[bid] = {'s':[],'a':[],'fp':[],'r':[],'stop_embed':[] }

        if args.restore == 1 and args.control > 2:
            for k, v in agents.items():
                print('Restore from:',save_para_path  + str(k))
                v.load(save_para_path + str(k))

        Flag = True
        begin = time.time()
        if ep==0:
            logging.basicConfig(filename='w{}.txt'.format(args.seed), filemode='w', format='%(message)s')
            logging.warning('Begin' + str(np.array(eng.PW.w).reshape(-1)))

        while Flag:
            Flag = eng.sim()
        print("simulation done with {} steps".format(eng.simulation_step))


        # 20220714 visualize serving conflict
        U.analyze_queueing(eng)



        ploss_log = [0]
        qloss_log = [0]
        if args.control > 2 and args.restore == 0:
            update_iterations = 3
            for _ in range(update_iterations):
                ploss, qloss, l_f = eng.learn()
                if l_f == True:
                    qloss_log.append(qloss)
                    ploss_log.append(ploss)

            if ep % 20 == 0 and ep > 10 and args.restore == 0:
                # store model
                for k, v in agents.items():
                    v.save(save_para_path + str(k))

        if args.control > 2:
            memory_copy = eng.GM
        else:
            memory_copy = None

        log_route = eng.cal_statistic(
            name=save_para_path  ,
            train=args.train)
        ## trajectory evaluation
        system_wait = log_route['22_1']["system_wait"]
        # system_travel = log_route["system_travel"]
        # system_aod = log_route["system_aod"]
        # eng.PW.hist_eva.append(system_travel)
        # eng.PW.hist_w.append(eng.PW.w)
        eval = system_wait

        for busid, bus in eng.bus_list.items():
            eng.PW.hist_eva[busid].append(eval)

        if args.control==7:
            PW_copy = eng.PW

            if eng.PW.num_traj == eng.PW.total_traj-1:
                lines = {}
                ids = []
                memory_copy.update()
                for busid, bus in eng.bus_list.items():
                    if len(eng.PW.traj[busid][0])==0:
                        continue
                    if bus.route_id not in lines:
                        ids.append(bus.route_id)
                        lines[bus.route_id] = []
                        lines[bus.route_id] = [eng.PW.traj[busid]]
                    else:
                        lines[bus.route_id].append(eng.PW.traj[busid])

                # eval, w = eng.PW.updatew(lines[ids[0]], lines[ids[1]], ep=ep)
                eval, w = eng.PW.learnw()
                logging.basicConfig(filename='w{}.txt'.format(args.seed), filemode='a', format='%(message)s')
                logging.warning(str(eval) + ' ' + str(np.array(w).reshape(-1)))
                eng.PW.reset()
            else:
                eng.PW.num_traj+=1

        name = abspath + "/mllog/"  + save_para_path

        # U.visualize_trajectory(engine=eng, name=name + '_' + str(args.para_flag) + str('_'))
        # U.vis_stop_record(engine=eng)
        print('Training>>> Control type',args.control,'Model', args.model, 'Seed',str(args.seed))
        for k, v in eng.route_list.items():
            print('Route: ', k)
            U.train_result_track(eng=eng, ep=ep, qloss_log=qloss_log, ploss_log=ploss_log, log=log_route[k],
                                 name=name + 'R' + k + '_',
                                 route = k,
                                 seed=args.seed)
        print('Total time cost:%g sec Episode time cost:%g' % (time.time() - now_,time.time() - now_ep))
        print('')
        # except Exception as e:
        #     print(e)
        eng.close()

def evaluate(args):
    stop_list, route_list = {}, {}
    num_stops = []
    save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                     + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' +   \
                      'RS_a_' + str(args.all) + '_'
    bus_routes = {}
    for data in args.data:
        d = data.split('_')
        r = d[1] + '_' + d[2]

        new_stop_list, pax_num = U.getStopList(data)
        new_bus_routes = U.getBusRoute(data)
        bus_routes[r] = new_bus_routes[r]
        route_list = reduce(lambda x, y: dict(x, **y), (route_list, new_bus_routes))
        num_stops.append(len(new_stop_list))

        print('%s buses prepared :%g' % (data, len(route_list[r])))
        print('%s stops prepared :%g' % (data, len(new_stop_list)))
        for k, v in new_stop_list.items():
            if k not in stop_list:
                stop_list[k] = v

            stop_list[k].dest_route[r] = v.dest
            stop_list[k].routes.append(r)
            stop_list[k].loc_route[r] = v.loc
            stop_list[k].prev_stop_route[r] = v.prev_stop
            stop_list[k].next_stop_route[r] = v.next_stop

    print('Shared stops: %d' % (np.sum(num_stops) - len(stop_list)))

    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    stop_list_ = copy.deepcopy(stop_list)
    bus_routes_ = copy.deepcopy(bus_routes)
    bus_list_ = copy.deepcopy(bus_list)
    agents = {}
    shares = args.shares
    if args.shares < 0:
        shares = np.random.randint(0, 10)
    else:
        shares = args.shares/10.
    if args.control > 2:
        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                policy_types=len(args.model.split('_')),
                                share_scale=args.share_scale, weight=args.weight, shares=shares)

        bus_list = eng.bus_list
        bus_stop_list = eng.busstop_list
        # U.demand_analysis(eng)

        if args.control % 2 == 0:
            state_dim = 8
        else:
            state_dim = 6
        if args.control==7:
            state_dim = 11
        # share all
        if args.share_scale == 0:
            agents[0] = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed,is_stop_embedding=args.stop_embedding)
            print('agents[0] state dim:', agents[0].state_dim)
            if len(args.model.split('_')) > 1:
                if args.control == 7:
                    state_dim = 6
                agents[1] = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                                  seed=args.seed,is_stop_embedding=args.stop_embedding)
                print('agents[1] state dim:', agents[1].state_dim)

        # share in route
        if args.share_scale == 1:
            agents = {}
            for k, v in eng.route_list.items():
                agent = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed,is_stop_embedding=args.stop_embedding)
                agents[k] = agent
    now_ = time.time()
    for ep in range(args.episode):
        stop_list_ = copy.deepcopy(stop_list)
        bus_routes_ = copy.deepcopy(bus_routes)
        bus_list_ = copy.deepcopy(bus_list)

        # stop.demand_impulse_rate = np.random.randint(10,50)/10. # simulate occuring rate of demand impulse

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                policy_types=len(args.model.split('_')),
                                share_scale=args.share_scale, weight=args.weight, shares=shares)
        eng.demand_impulse_rate = 0.2#np.random.randint(0,20)/10. # control the serverity of demand anomaly
        eng.accident_rate = 0.2#np.random.randint(0,20) / 100.  # control the serverity of traffic state anomaly

        for _, stop in stop_list_.items():
            for r in stop.routes:
                stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate, route=r, shares=eng.shares,
                                 share_stops=eng.shared_stops)
        print('------------------------  Epoch: %d ----------------------' % ep)
        now_ep = time.time()

        eng.busstop_list = stop_list_

        eng.agents = agents

        if args.restore == 1 and args.control > 2:
            save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                             + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_RS'  \
                             + '_a_' + str(args.all) + '_'
            for k, v in agents.items():
                print('Restore from:',save_para_path + str(k))
                v.load(save_para_path +  str(k))

        Flag = True
        begin = time.time()
        while Flag:
            Flag = eng.sim()

        if args.control > 2:
            memory_copy = eng.GM
        else:
            memory_copy = None


        save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                         + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' + str(shares) \
                         + '_a_' + str(args.all) + '_'

        log_route = eng.cal_statistic(
            name=save_para_path,
            train=args.train)

        name = abspath + "/mllogt/" +  save_para_path

        # U.visualize_trajectory(engine=eng, name=name + '_' + str(args.para_flag) + str('_'))
        # U.vis_stop_record(engine=eng)
        for k, v in eng.route_list.items():

            U.train_result_track(eng=eng, ep=ep, qloss_log=[0], ploss_log=[0], log=log_route[k],
                                 name=name + 'R' + k + '_' ,
                                 route=k,
                                 seed=args.seed)
        print('Total time cost:%g sec Episode time cost:%g' % (time.time() - now_, time.time() - now_ep))
        print('')
        # except Exception as e:
        #     print(e)
        eng.close()
if __name__ == '__main__':


    seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.train==1:
        train(args)

    else:
        evaluate(args)






