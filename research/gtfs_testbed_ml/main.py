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
parser.add_argument("--seed", type=int, default=9)              # random seed
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
parser.add_argument("--train", type=int, default=0)              # train=1: train the model
parser.add_argument("--restore", type=int, default=1)            # 1: restore the model
parser.add_argument("--all", type=int, default=1)                # 0: for considering only forward/backward buses. 1: for all buses
parser.add_argument("--vis", type=int, default=0)                # 1: to visualize bus trajectory in test phase
parser.add_argument("--weight", type=int, default=6)             # weight for action penalty
parser.add_argument("--share_scale", type=int, default=0)        # 0: share. 1: route-share
parser.add_argument("--shares", type=int, default=-2)             # share rate of demand in common corridor
parser.add_argument("--stop_embedding", type=int, default=0)
parser.add_argument("--query_freq", type=int, default=16)
parser.add_argument("--eval", type=int, default=0)
parser.add_argument("--test_d", type=int, default=2)
args = parser.parse_args()

if args.model in ['caac','caac_2',]:
    from model.CAAC import Agent
    if args.control == 7:
        from model.CAAC_AW import Agent
        import sim.Sim_Engine_AW as Sim_Engine
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



def train_old(args ):
    stop_list, route_list  = {},{}
    num_stops = []
    save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                     + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' + s \
                     + '_a_' + str(args.all) + '_'
    if args.eval>0:
        save_para_path+='{}_'.format(args.eval)
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

    print('Shared stops: %d'%(np.sum(num_stops)-len(stop_list)))

    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
    stop_list_ = copy.deepcopy(stop_list)
    bus_routes_ = copy.deepcopy(bus_routes)
    bus_list_ = copy.deepcopy(bus_list)
    agents = {}


    if args.control > 2:

        bus_list = bus_list_
        bus_stop_list = stop_list_
        # U.demand_analysis(eng)

        if args.control % 2 == 0:
            state_dim = 8
        else:
            state_dim = 6
        if args.control == 7:
            state_dim = 11
        # share all
        if args.share_scale == 0:
            agents[0] = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed,is_stop_embedding=args.stop_embedding)
            print('agents[0] state dim:', agents[0].state_dim)
            # if len(args.model.split('_')) > 1:
            #     agents[1] = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
            #                       seed=args.seed,is_stop_embedding=args.stop_embedding)
            #     print('agents[1] state dim:', agents[1].state_dim)

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

        eng = Sim_Engine.Engine(bus_list=bus_list_,
                                busstop_list=stop_list_,
                                control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0,
                                simulation_step=simulation_step,
                                route_list=route_list,
                                hold_once_arr=args.arr_hold,
                                is_allow_overtake=args.overtake,
                                policy_types=len(args.model.split('_')),
                                seed = args.seed,
                                share_scale=args.share_scale,
                                weight=args.weight,
                                shares=shares)

        # eng.demand_impulse_rate = np.random.randint(0,40)/100. # control the serverity of demand anomaly
        # eng.accident_rate =  np.random.randint(0,40) / 100.  # control the serverity of traffic state anomaly
        eng.demand_impulse_rate = np.random.randint(0, 40) / 10.  # control the serverity of demand anomaly
        eng.accident_rate = np.random.randint(0, 40) / 100.  # control the serverity of traffic state anomaly
        begin = time.time()
        now_ep = time.time()
        for _, stop in stop_list_.items():
            for r in stop.routes:
                stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate,
                                 route=r,
                                 shares=eng.shares,
                                 share_stops=eng.shared_stops)

        print('------------------------  Epoch: %d ----------------------'%ep)
        print("create pax cost {}".format(time.time() - begin))

        eng.busstop_list = stop_list_
        eng.agents = agents

        if ep > 0:
            if memory_copy != None:
                eng.GM = memory_copy
            if args.control==7 and PW_copy != None:
                eng.PW = PW_copy
            for bid, b in eng.bus_list.items():
                for k, v in eng.GM.temp_memory[bid].items():
                    eng.GM.temp_memory[bid][k] = []

        if args.restore == 1 and args.control > 2:
            for k, v in agents.items():
                print('Restore from:',save_para_path  + str(k))
                v.load(save_para_path + str(k))

        Flag = True
        begin = time.time()

        while Flag:
            Flag = eng.sim()
        ploss_log = [0]
        qloss_log = [0]
        if args.control > 2 and args.restore == 0:
            update_iterations = 3
            for _ in range(update_iterations):
                ploss, qloss, _ = eng.learn()
                if ploss != 0:
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
        system_travel = log_route['22_1']["system_travel"]
        system_aod = log_route['22_1']["system_aod"]

        name = abspath + "/mllog/"  + save_para_path
        print('Training>>> Control type',args.control,'Model', args.model, 'Seed',str(args.seed))
        for k, v in eng.route_list.items():
            print('Route: ', k)
            U.train_result_track(eng=eng,
                                 ep=ep,
                                 qloss_log=qloss_log,
                                 ploss_log=ploss_log,
                                 log=log_route[k],
                                 name=name + 'R' + k + '_',
                                 route = k,
                                 seed=args.seed)
        print('Total time cost:%g sec Episode time cost:%g' % (time.time() - now_, time.time() - now_ep))
        print('')
        # except Exception as e:
        #     print(e)
        eng.close()
def train(args):
    stop_list, route_list = {}, {}
    num_stops = []
    save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                     + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' + s \
                     + '_a_' + str(args.all) + '_'
    if args.eval>0:
        save_para_path+='e{}_'.format(args.eval)
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
            stop_list[k].serving_period = [0 for _ in range(120000)]

    print('Shared stops: %d' % (np.sum(num_stops) - len(stop_list)))

    dispatch_times, bus_list, route_list, simulation_step = U.init_bus_list(bus_routes)
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
        if args.control == 7:
            state_dim = 11
        # share all
        if args.share_scale == 0:
            agents[0] = Agent(state_dim=state_dim, name='', n_stops=len(bus_stop_list), buslist=bus_list,
                              seed=args.seed, is_stop_embedding=args.stop_embedding)
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
        bus_list_ = copy.deepcopy(bus_list)

        # stop.demand_impulse_rate = np.random.randint(10,50)/10. # simulate occuring rate of demand impulse

        eng = Sim_Engine.Engine(bus_list=bus_list_, busstop_list=stop_list_, control_type=args.control,
                                dispatch_times=dispatch_times,
                                demand=0, simulation_step=simulation_step, route_list=route_list,
                                hold_once_arr=args.arr_hold, is_allow_overtake=args.overtake,
                                policy_types=len(args.model.split('_')),
                                share_scale=args.share_scale, weight=args.weight, shares=shares)
        eng.demand_impulse_rate = np.random.randint(0, 40) / 10.  # control the serverity of demand anomaly
        eng.accident_rate = np.random.randint(0, 40) / 100.  # control the serverity of traffic state anomaly
        # eng.demand_impulse_rate = 0.2
        # eng.accident_rate = 0.2
        total_pax = 0.
        begin = time.time()
        for _, stop in stop_list_.items():
            for r in stop.routes:
                stop.pre_pax_gen(demand_impulse_rate=eng.demand_impulse_rate, route=r, shares=eng.shares,
                                 share_stops=eng.shared_stops)
                total_pax += len(stop.pre_gen_pax_list)
        print('------------------------  Epoch: %d ----------------------' % ep)
        print("pax {}".format(total_pax))

        print("time cost {}".format(time.time() - begin))
        now_ep = time.time()

        eng.busstop_list = stop_list_
        eng.agents = agents

        if ep > 0:
            if memory_copy is not None:
                eng.GM = memory_copy
            # if args.control == 7 and PW_copy != None:
            #     eng.PW = PW_copy
            if args.control == 7 and APL_copy is not None:
                eng.APL = APL_copy
                eng.APL2 = APL_copy2
            for bid, b in eng.bus_list.items():
                for k, v in eng.GM.temp_memory[bid].items():
                    eng.GM.temp_memory[bid][k] = []
                # eng.GM.temp_memory[bid] = {'s': [], 'a': [], 'fp': [], 'r': [], 'stop_embed': []}

        if args.restore == 1 and args.control > 2:
            for k, v in agents.items():
                print('Restore from:', save_para_path + str(k) + 'seed ' + str(args.seed))
                v.load(save_para_path + str(k))

        Flag = True
        begin = time.time()
        if ep == 0 and args.control == 7:
            if args.eval>0:
                logging.basicConfig(filename='w{}_e{}.txt'.format(args.seed, args.eval), filemode='w', format='%(message)s')
            else:
                logging.basicConfig(filename='w{}.txt'.format(args.seed), filemode='w', format='%(message)s')
            logging.warning('Begin ')

        while Flag:
            Flag = eng.sim()

        if Flag is False and args.control == 7:
            if args.eval > 0:
                file = "w{}e1.csv".format(args.seed)
            else:
                file = "w{}.csv".format(args.seed)
            with open(file, 'a') as f:
                eng.debug["1"] = [np.mean(eng.debug["1"])]
                eng.debug["2"] = [np.mean(eng.debug["2"])]
                eng.debug["3"] = [np.mean(eng.debug["3"])]
                eng.debug["4"] = [np.mean(eng.debug["4"])]
                eng.debug["5"] = [np.mean(eng.debug["5"])]
                eng.debug["6"] = [np.mean(eng.debug["6"])]
                df = pd.DataFrame(eng.debug)
                df.to_csv(file, mode='a', header=f.tell() == 0)

        log_route = eng.cal_statistic(
            name=save_para_path,
            train=args.train)
        # trajectory evaluation
        system_wait = log_route['22_1']["system_wait"]
        system_travel = log_route['22_1']["system_travel"]
        system_aod = log_route['22_1']["system_aod"]
        # eng.PW.hist_eva.append(system_travel)
        # eng.PW.hist_w.append(eng.PW.w)
        print("===" * 20)
        print(system_wait, system_aod, system_travel)
        print("===" * 20)
        eval = system_wait
        if args.eval==1:
            eval = system_travel
        if args.control == 7:
            eng.APL.num_traj += 1
            eng.APL2.num_traj += 1
            for busid, bus in eng.bus_list.items():
                eng.APL.hist_eva[busid].append(eval)
                eng.APL2.hist_eva[busid].append(eval)
        print("simulation done with {} steps".format(eng.simulation_step))
        print("simulation time cost {}".format(time.time() - begin))
        # df = pd.DataFrame(eng.debug)
        # df.to_csv("reward3.csv")
        # # 20220714 visualize serving conflict
        # U.analyze_queueing(eng)
        # # 20220725 visualize arriving curve
        # U.cumulative_arr_curve(eng, name=str(args.model) + "_" + str(args.seed))
        # continue
        ploss_log = [0]
        qloss_log = [0]
        if args.control > 2 and args.restore == 0:
            update_iterations = 3
            for _ in range(update_iterations):
                ploss, qloss, istrain = eng.learn()
                if istrain:
                    qloss_log.append(qloss)
                    ploss_log.append(ploss)
                    if args.control == 7 and eng.APL.num_traj >= 2:
                        eng.APL.update_volume()
                        eng.APL2.update_volume()

            if ep % 20 == 0 and ep > 10 and args.restore == 0:
                # store model
                for k, v in agents.items():
                    v.save(save_para_path + str(k))

        if args.control > 2:
            memory_copy = eng.GM
        else:
            memory_copy = None



        if args.control == 7:
            if eng.APL.num_traj > 30 and eng.APL.num_traj % args.query_freq == 0:
                eng.APL.log_preferences(eng.APL.volume_buffer.best_delta, eng.APL.volume_buffer.preference)
                eng.APL2.log_preferences(eng.APL2.volume_buffer.best_delta, eng.APL2.volume_buffer.preference)
                w, accept_rate = eng.APL.mcmc_vanilla()
                w2, accept_rate2 = eng.APL2.mcmc_vanilla()
                # print("best delta {} preference {}".format(eng.APL.volume_buffer.best_delta,
                #                                            eng.APL.volume_buffer.preference))
                # print(eng.APL.volume_buffer.objective_logs[-1])
                logging.basicConfig(filename='w{}.txt'.format(args.seed), filemode='a', format='%(message)s')
                logging.warning('Episode ' + str(ep))
                logging.warning('w ' + str(np.array(w).reshape(-1)) + ' a ' + str(accept_rate))
                logging.warning('w2 ' + str(np.array(w2).reshape(-1)) + ' a2 ' + str(accept_rate2))
                # logging.warning("traj eval compare {}".format(str(eng.APL.volume_buffer.objective_logs[-1])))
                logging.warning("")
                eng.APL.reset()
                eng.APL2.reset()
                # memory_copy = None

            APL_copy = eng.APL
            APL_copy2 = eng.APL2


        name = abspath + "/mllog/" + save_para_path

        # U.visualize_trajectory(engine=eng, name=name + '_' + str(args.para_flag) + str('_'))
        # U.vis_stop_record(engine=eng)
        print('Training>>> Control type', args.control, 'Model', args.model, 'Seed', str(args.seed))
        for k, v in eng.route_list.items():
            print('Route: ', k)
            U.train_result_track(eng=eng, ep=ep, qloss_log=qloss_log, ploss_log=ploss_log, log=log_route[k],
                                 name=name + 'R' + k + '_',
                                 route=k,
                                 seed=args.seed)
        print('Total time cost:%g sec Episode time cost:%g' % (time.time() - now_, time.time() - now_ep))
        print('')
        # except Exception as e:
        #     print(e)
        eng.close()
def  evaluate(args):
    stop_list, route_list = {}, {}
    num_stops = []
    save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                     + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' +   \
                      'RS_a_' + str(args.all) + '_'
    if args.eval>0:
        save_para_path+='e{}_'.format(args.eval)
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
                                seed=args.seed,
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
                                seed=args.seed,
                                share_scale=args.share_scale, weight=args.weight, shares=shares)
        # eng.demand_impulse_rate = 0.2#np.random.randint(0,20)/10. # control the serverity of demand anomaly
        # eng.accident_rate = 0.2#np.random.randint(0,20) / 100.  # control the serverity of traffic state anomaly
        eng.demand_impulse_rate = args.test_d/10.  # np.random.randint(0,20)/10. # control the serverity of demand anomaly
        eng.accident_rate = 0.1
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
            if args.eval > 0:
                save_para_path += 'e{}_'.format(args.eval)
            for k, v in agents.items():
                print('Restore from:',save_para_path + str(k))
                v.load(save_para_path +  str(k))

        Flag = True
        begin = time.time()
        if args.control == 7:
            weights = np.array(pd.DataFrame(pd.read_csv("w{}.csv".format(args.seed))).iloc[-1,1:])
            if args.eval > 0:
                weights = np.array(pd.DataFrame(pd.read_csv("w{}e{}.csv".format(args.seed, args.eval))).iloc[-1,1:])
            eng.APL2.w_curr_mean = weights[3:5]
            eng.APL.w_curr_mean = weights[:3]
            print("Load wieght setting of seed {}:".format(args.seed))
            print( eng.APL2.w_curr_mean)
            print(eng.APL.w_curr_mean)
        while Flag:
            Flag = eng.sim()


        if args.control > 2:
            memory_copy = eng.GM
        else:
            memory_copy = None


        save_para_path = str(args.para_flag) + str('_sc_') + str(args.share_scale) + str('_w_') + str(args.weight) \
                         + str('_m_') + str(args.model) + str('_ctrl_') + str(args.control) + '_sh_' + str(shares) \
                         + '_a_' + str(args.all) + '_'
        if args.eval > 0:
            save_para_path += 'e{}_'.format(args.eval)
        log_route = eng.cal_statistic(
            name=save_para_path,
            train=args.train)

        name = abspath + "/mllogt{}/".format(args.test_d) +  save_para_path
        try:
            os.makedirs(abspath + "/mllogt{}/".format(args.test_d))
        except:
            print(abspath + "/mllogt{}/".format(args.test_d), ' has existed')
        # U.visualize_trajectory(engine=eng, name=name + '_' + str(args.para_flag) + str('_'))
        if args.vis==1:
            print('time cost:%g sec' % (time.time() - now_))

            if args.control == 0:
                folder = abspath + "/vis/vis_nc_{}/".format(str(args.shares))
            if args.control == 1:
                folder = abspath + "/vis/vis_fc_{}/".format(str(args.shares))
            if args.control == 2:
                folder = abspath + "/vis/vis_fczhou_{}/".format(str(args.shares))
            if args.control> 2:
                folder = abspath + "/vis/vis_{}{}".format(args.model , str(args.control)) +  '_{}/'.format(str(args.shares))
            try:
                os.makedirs(folder)
            except:
                print(folder, ' has existed')
            U.stop_record(engine=eng,folder=folder)
            # U.cum_arr_curve(eng, name)
            exit()
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
        if args.control==7:
            train(args)
        else:
            train_old(args)

    else:
        evaluate(args)






