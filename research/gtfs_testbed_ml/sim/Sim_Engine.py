import numpy as np
import torch

from sim.Passenger import Passenger
from model.Group_MemoryC import Memory

import pandas as pd
import math
import matplotlib.pyplot as plt
from model.PreferenceWorker import PWorker
from model.ActivePreferenceLearner import PreferenceLearner


def softmax(probs):
    ratio = 1.5
    probs_ = [p * ratio for p in probs]
    return [np.exp(p * ratio) / np.sum(np.exp(probs_)) for p in probs_]


def H(prob):
    e_x = [-p_x * np.math.log(p_x, 2) for p_x in prob]
    return np.sum(e_x)


class Engine():
    def __init__(self, bus_list, busstop_list, route_list, simulation_step, dispatch_times, demand=0, agents=None,
                 policy_types=0,
                 share_scale=0, is_allow_overtake=0, hold_once_arr=1, control_type=1, seed=1, all=0, weight=0,
                 shares=0):
        self.debug = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": []}
        self.all = all
        self.busstop_list = busstop_list
        self.simulation_step = simulation_step
        self.pax_list = {}  # passenger on road
        self.arr_pax_list = {}  # passenger finihsed trip
        self.dispatch_buslist = {}
        self.agents = {}
        self.route_list = route_list
        self.is_allow_overtake = is_allow_overtake
        self.hold_once_arr = hold_once_arr
        self.control_type = control_type

        self.agents = agents
        self.bus_list = bus_list
        self.bunching_times = {}
        self.max_bus_queue = 0

        self.reward_signal = {}
        self.reward_signalp1 = {}
        self.reward_signalp2 = {}
        self.reward_signalp3 = {}
        self.qloss = {}
        self.weight = weight / 10.
        self.demand = demand
        self.records = []
        self.share_scale = share_scale
        self.dispatch_times = dispatch_times
        self.cvlog = []
        self.shares = shares
        self.policy_types = policy_types

        members = list(self.bus_list.keys())
        self.GM = Memory(members)
        self.PW = PWorker(members)
        self.APL = PreferenceLearner(members=members, d=3)
        self.APL2 = PreferenceLearner(members=members, d=2)
        self.bay_capacity = 1

        for b_id, b in self.bus_list.items():
            self.reward_signal[b_id] = []
            self.reward_signalp1[b_id] = []
            self.reward_signalp2[b_id] = []
            self.reward_signalp3[b_id] = []

        self.arrivals = {}

        shared_stops = []
        self.pax_route = {}
        self.route_list = {}
        cs = ['red', 'blue']
        rc = 0

        for k, v in route_list.items():
            self.pax_route[k] = 0
            self.bunching_times[k] = 0
            self.route_list[k] = v
            self.route_list[k].c = cs[rc]
            rc += 1
            if len(shared_stops) == 0:
                shared_stops = v.stop_list
            else:
                shared_stops = list(set(shared_stops) & set(v.stop_list))

        # stop hash
        self.stop_hash = {}
        k = 0
        for bus_stop_id, bus_stop in self.busstop_list.items():
            self.stop_hash[bus_stop_id] = k
            k += 1

        self.bus_hash = {}
        k = 0
        for bus_id, bus in self.bus_list.items():
            self.bus_hash[bus_id] = k
            k += 1
        self.shared_stops = list(set(shared_stops))
        self.shared_stops_locs = {}
        self.merge_stop = {}
        self.diverge_stop = {}
        for bus_stop_id, bus_stop in self.busstop_list.items():
            if bus_stop_id in self.shared_stops:
                bus_stop.is_in_shared_corridor = 1
                self.shared_stops_locs[bus_stop_id] = bus_stop.loc
                try:

                    for rid in list(bus_stop.loc_route.keys()):
                        if rid not in self.merge_stop:
                            self.merge_stop[rid] = [bus_stop.id, bus_stop.loc_route[rid]]
                        else:
                            if self.merge_stop[rid][1] < bus_stop.loc_route[rid]:
                                self.merge_stop[rid] = [bus_stop.id, bus_stop.loc_route[rid]]
                        if rid not in self.diverge_stop:
                            self.diverge_stop[rid] = [bus_stop.id, bus_stop.loc_route[rid]]
                        else:
                            if self.diverge_stop[rid][1] > bus_stop.loc_route[rid]:
                                self.diverge_stop[rid] = [bus_stop.id, bus_stop.loc_route[rid]]
                except:
                    print()
        self.action_record = []
        self.reward_record = []
        self.state_record = []
        self.accident_rate = 0.
        self.sensor_error_rate = 0.
        self.demand_impulse_rate = 0.

        self.sim_actual_begin = -1

    def cal_statistic(self, name, train=1):
        print('total pax:%d' % (len(self.pax_list)))
        shared_pax = []
        original_pax_in_shared = {}
        in_share = 0
        for k, p in self.pax_list.items():
            if p.share == 1:
                shared_pax.append(p)
            if p.origin in self.shared_stops and p.dest in self.shared_stops:
                in_share += 1
                if p.route in original_pax_in_shared:
                    original_pax_in_shared[p.route] += 1
                else:
                    original_pax_in_shared[p.route] = 1

        for kk, v in original_pax_in_shared.items():
            print('%s: %g' % (kk, v * 1.0 / in_share))

        print('%s, share %g' % (k, (len(shared_pax) * 1.0 / len(self.pax_list))))
        logs = {}
        system_waitcost = []
        system_travelcost = []
        system_AOD = []
        headways_var_sharestops = []
        headways_mean_sharestops = []
        for routeid, values in self.route_list.items():
            wait_cost = []
            travel_cost = []
            headways_var = {}
            headways_mean = {}
            boards = []
            arrs = []
            origins = []
            dests = []
            still_wait = 0
            stop_wise_wait = {}
            stop_wise_hold = {}
            delay = []
            miss = []
            for pax_id, pax in self.pax_list.items():
                if self.bus_list[pax.took_bus].route_id != routeid:
                    continue
                w = min(pax.onboard_time - pax.arr_time, self.simulation_step - pax.arr_time)
                miss.append(pax.miss)
                wait_cost.append(w)
                if pax.origin in stop_wise_wait:
                    stop_wise_wait[pax.origin].append(w)
                else:
                    stop_wise_wait[pax.origin] = [w]
                if pax.onboard_time < 99999999:
                    boards.append(pax.onboard_time)
                    if pax.alight_time < 999999:
                        travel_cost.append(pax.alight_time - pax.arr_time)
                        delay.append(pax.alight_time - pax.arr_time - pax.onroad_cost)
                else:
                    still_wait += 1

            # print('MISS:%g'%(np.max(miss)))
            hold_cost = []
            demo_bus = ''
            for bus_id, bus in self.bus_list.items():
                if bus.route_id != routeid:
                    continue
                demo_bus = bus
                for k, v in bus.stay.items():
                    if v > 0:
                        hold_cost.append(bus.hold_cost[k])
                        if k in stop_wise_hold:
                            stop_wise_hold[k].append(bus.hold_cost[k])
                        else:
                            stop_wise_hold[k] = [bus.hold_cost[k]]
                    else:
                        print('v==0', bus.id)

            stop_wise_wait_order = []
            stop_wise_hold_order = []

            arr_times = []
            buslog = pd.DataFrame()
            for bus_stop_id in demo_bus.pass_stop:
                buslog[bus_stop_id] = self.busstop_list[bus_stop_id].arr_log[routeid]

                if bus_stop_id in self.shared_stops:
                    # plt.scatter([self.stop_hash[bus_stop_id] for _ in range(len(buslog[bus_stop_id]))], buslog[bus_stop_id],c=self.route_list[demo_bus.route_id].c)

                    h = (np.array(buslog[bus_stop_id])[1:] - np.array(buslog[bus_stop_id])[:-1])
                    headways_mean_sharestops.append(np.mean(h))
                    headways_var_sharestops.append(np.var(h))

                arr_times.append([bus_stop_id] + self.busstop_list[bus_stop_id].arr_log[routeid])
                try:
                    stop_wise_wait_order.append(np.mean(stop_wise_wait[bus_stop_id]))
                except:
                    stop_wise_wait_order.append(0)
                try:
                    stop_wise_hold_order.append(np.mean(stop_wise_hold[bus_stop_id]))
                except:
                    stop_wise_hold_order.append(0)

                for k, v in self.busstop_list[bus_stop_id].arr_log.items():
                    if k != routeid:
                        continue
                    h = (np.array(v)[1:] - np.array(v)[:-1])
                    if np.isnan(np.var(h)):
                        continue
                    try:
                        headways_var[bus_stop_id].append(np.var(h))
                        headways_mean[bus_stop_id].append(np.mean(h))
                    except:
                        headways_var[bus_stop_id] = [np.var(h)]
                        headways_mean[bus_stop_id] = [np.mean(h)]

            log = {}
            system_travelcost.append(travel_cost)
            system_waitcost.append(wait_cost)

            log['wait_cost'] = wait_cost
            log['travel_cost'] = travel_cost
            log['hold_cost'] = hold_cost
            log['headways_var'] = headways_var
            log['headways_mean'] = headways_mean
            log['stw'] = stop_wise_wait_order
            log['sth'] = stop_wise_hold_order
            log['bunching'] = self.bunching_times[routeid]
            log['delay'] = delay
            log['EV'] = (np.mean(list(headways_var.values())) / (np.mean(list(headways_mean.values())) ** 2))
            print('[%s] bunching times:%g wait:%g travel:%g EV:%g pax:%d' % (
                routeid, self.bunching_times[routeid], np.mean(wait_cost), np.mean(travel_cost),
                (np.mean(list(headways_var.values())) / (np.mean(list(headways_mean.values())) ** 2)),
                self.pax_route[routeid]))
            print('Mean of  var of headway', np.mean(list(headways_var.values())))
            print('Mean of headway', np.mean(list(headways_mean.values())))
            AWT = []
            AHD = []
            AOD = []
            for k in demo_bus.pass_stop:
                AHD.append(np.mean(stop_wise_hold[k]))
                try:
                    AWT.append(np.mean(stop_wise_wait[k]))
                except:
                    AWT.append(0.)
                if np.mean(self.busstop_list[k].arr_bus_load) == 0:
                    AOD.append(0)
                else:
                    AOD.append(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load))
            log['sto'] = AOD
            log['AOD'] = np.mean(AOD)
            system_AOD.append(AOD)
            if train == 0:
                print('AWT:%g' % (np.mean(wait_cost)))
                print('AHD:%g' % (np.mean(AHD)))
                print('AOD:%g' % (np.mean(AOD)))
                print('headways_var:%g' % (np.sqrt(np.mean(list(headways_var.values())))))

            log['arr_times'] = arr_times
            logs[routeid] = log

        # plt.show()
        print('system wait cost', (np.mean(system_waitcost[0]) + np.mean(system_waitcost[1])) / 2.)
        print('system travel cost', (np.mean(system_travelcost[0]) + np.mean(system_travelcost[1])) / 2.)
        print('Mean of headway at shared stops', np.mean(headways_mean_sharestops))
        print('Mean of variation of headway at shared stops', np.mean(headways_var_sharestops))
        for k, v in logs.items():
            logs[k]['system_wait'] = (np.mean(system_waitcost[0]) + np.mean(system_waitcost[1])) / 2.
            logs[k]['system_travel'] = (np.mean(system_travelcost[0]) + np.mean(system_travelcost[1])) / 2.
            logs[k]['system_aod'] = (np.mean(system_AOD[0]) + np.mean(system_AOD[1])) / 2.
        # plt.scatter([ss for ss in range(len(headways_var_sharestops))], headways_var_sharestops)
        # plt.show()
        return logs

    def close(self):
        return

    # update passengers when bus arriving at stops
    def serve(self, bus, stop):
        board_cost = 0
        alight_cost = 0
        board_pax = []
        alight_pax = []
        if bus != None:
            # alighting procedure
            if self.demand == 0:
                alight_pax = bus.pax_alight()
            else:
                alight_pax = bus.pax_alight_fix(stop, self.pax_list)
            for p in alight_pax:
                self.pax_list[p].alight_time = self.simulation_step
                bus.onboard_list.remove(p)
                self.arr_pax_list[p] = self.pax_list[p]

            alight_cost = len(alight_pax) * bus.alight_period

            # boarding procedure
            if self.demand == 0:
                new_arr = stop.pax_gen_sp(bus, sim_step=self.simulation_step, shares=self.shares)
                # new_arr = stop.pax_gen(bus, sim_step=self.simulation_step)
                self.pax_route[bus.route_id] += len(new_arr)
                num = len(self.pax_list) + 1
                for t in new_arr:
                    self.pax_list[num] = Passenger(id=num, origin=stop.id, arr_time=t)
                    self.pax_list[num].took_bus = bus.id
                    self.pax_list[num].route = bus.route_id
                    self.busstop_list[stop.id].waiting_list.append(num)
                    num += 1
                pax_leave_stop = []
                waitinglist = sorted(self.busstop_list[stop.id].waiting_list)[:]
                for num in waitinglist:
                    if self.pax_list[num].route == bus.route_id:
                        self.pax_list[num].miss += 1

                    if bus.capacity > len(bus.onboard_list) and (self.pax_list[num].route == bus.route_id or (
                            self.busstop_list[stop.id].is_in_shared_corridor == 1 and np.random.randint(0,
                                                                                                        10) / 10. < self.shares)):
                        self.pax_list[num].onboard_time = self.simulation_step
                        bus.onboard_list.append(num)
                        board_cost += bus.board_period
                        pax_leave_stop.append(num)

                for num in pax_leave_stop:
                    self.busstop_list[stop.id].waiting_list.remove(num)
            else:
                new_arr = stop.pax_read(bus, sim_step=self.simulation_step)
                for p in new_arr:
                    self.pax_list[p.id] = p
                    self.busstop_list[stop.id].waiting_list.append(p.id)

                pax_leave_stop = []
                for pid in self.busstop_list[stop.id].waiting_list:
                    # add logic to consider multiline impact (i.e. the passenger can not board bus this time can board the bus with same destination later?)
                    if bus != None and bus.capacity - len(bus.onboard_list) > 0 and self.pax_list[
                        pid].route == bus.route_id:
                        self.pax_list[pid].onboard_time = self.simulation_step
                        bus.onboard_list.append(pid)
                        board_cost += bus.board_period
                        pax_leave_stop.append(pid)

                for pid in pax_leave_stop:
                    self.busstop_list[stop.id].waiting_list.remove(pid)

        else:
            new_arr = stop.pax_gen(bus, simulation_step=self.simulation_step)
            num = len(self.pax_list) + 1
            for t in new_arr:
                self.pax_list[num] = Passenger(id=num, origin=stop.id, arr_time=t)
                self.pax_list[num].route = ''
                self.busstop_list[stop.id].waiting_list.append(num)
                num += 1

        return alight_cost, board_cost

    def serve_by_presetOD(self, bus, stop):
        board_cost = 0
        alight_cost = 0
        board_pax = []
        alight_pax = []

        if bus != None:
            alight_pax = bus.pax_alight_fix(stop, self.pax_list)
            for p in alight_pax:
                self.pax_list[p].alight_time = self.simulation_step
                bus.onboard_list.remove(p)
                self.arr_pax_list[p] = self.pax_list[p]

            alight_cost = len(alight_pax) * bus.alight_period

            # boarding procedure

            new_pax = stop.get_pax(bus, sim_step=self.simulation_step)
            self.pax_route[bus.route_id] += len(new_pax)

            num = len(self.pax_list) + 1
            for k in range(len(new_pax)):
                p = new_pax[k]
                p.id = num
                self.pax_list[num] = p
                self.pax_list[num].took_bus = bus.id
                self.pax_list[num].route = bus.route_id
                self.busstop_list[stop.id].waiting_list.append(num)
                num += 1
            self.busstop_list[stop.id].cum_arr += len(new_pax)

            pax_leave_stop = []
            waiting_list = sorted(self.busstop_list[stop.id].waiting_list)[:]
            wait_num = len(waiting_list)
            for num in waiting_list:
                # add logic to consider multiline impact (i.e. the passenger can not board bus this time can board the bus with same destination later?)
                if bus != None and self.pax_list[num].route == bus.route_id:
                    self.pax_list[num].miss += 1
                if bus != None and bus.capacity > len(bus.onboard_list) and self.pax_list[
                    num].route == bus.route_id:
                    self.pax_list[num].onboard_time = self.simulation_step
                    bus.onboard_list.append(num)
                    board_cost += bus.board_period
                    pax_leave_stop.append(num)
                    self.busstop_list[stop.id].cum_dep += 1

            for num in pax_leave_stop:
                self.busstop_list[stop.id].waiting_list.remove(num)
                # 714
                self.busstop_list[stop.id].pax_actual_done.append(num)

        self.busstop_list[stop.id].actual_departures[self.simulation_step] = self.busstop_list[stop.id].cum_dep
        self.busstop_list[stop.id].actual_arrivals[self.simulation_step] = self.busstop_list[stop.id].cum_arr
        return alight_cost, board_cost, wait_num

    def sim(self):
        # update bus state

        if self.simulation_step > 180000 and self.simulation_step % 120 == 0:
            for r_id, r in self.route_list.items():
                trajectory = pd.DataFrame()
                mint = 999999999999999999
                maxt = -1
                for b_id in r.bus_list:
                    df = pd.DataFrame()

                    b = self.bus_list[b_id]

                    y = np.array(b.loc)
                    plt.plot(b.time_step, b.loc, c='blue')
                    mint = min(mint, b.time_step[0])
                    maxt = max(maxt, b.time_step[-1])
                vmax = -1
                vmin = 9999999999999
                for k, v in self.shared_stops_locs.items():
                    vmax = max(vmax, v)
                    vmin = min(vmin, v)
                plt.hlines(y=vmax, xmin=mint, xmax=maxt, colors='orange', linestyles='-', lw=3)
                plt.hlines(y=vmin, xmin=mint, xmax=maxt, colors='orange', linestyles='-', lw=3)
                plt.show()
        for bus_id, bus in self.bus_list.items():

            if bus.is_dispatch == 0 and bus.dispatch_time <= self.simulation_step:
                bus.is_dispatch = 1
                bus.last_action_tag = self.simulation_step
                bus.current_speed = bus.speed * np.random.randint(60., 120.) / 100.
                self.dispatch_buslist[bus_id] = bus

                if self.sim_actual_begin == -1:
                    self.sim_actual_begin = self.simulation_step

            if bus.is_dispatch == 1 and len(self.dispatch_buslist[bus_id].left_stop) <= 0:
                bus.is_dispatch = -1

                self.dispatch_buslist.pop(bus_id, None)

        for bus_id, bus in self.dispatch_buslist.items():
            if bus.backward_bus != None and self.bus_list[bus.backward_bus].is_dispatch == -1:
                bus.backward_bus = None
            if bus.forward_bus != None and self.bus_list[bus.forward_bus].is_dispatch == -1:
                bus.forward_bus = None

        ## bus dynamic

        for bus_id, bus in self.dispatch_buslist.items():
            bus.serve_remain = max(bus.serve_remain - 1, 0)
            bus.hold_remain = max(bus.hold_remain - 1, 0)

            ### on-arrival
            if bus.arr == 0 and abs(bus.loc[-1] - bus.stop_dist[bus.left_stop[0]]) < bus.speed:
                curr_stop = self.busstop_list[bus.left_stop[0]]
                # 20220714
                if self.busstop_list[curr_stop.id].serving_period[self.simulation_step] == 0:
                    self.busstop_list[curr_stop.id].serving_period[self.simulation_step] = bus.route_id
                else:
                    if self.busstop_list[curr_stop.id].serving_period[self.simulation_step] == bus.route_id:
                        self.busstop_list[curr_stop.id].serving_period[self.simulation_step] = "bunching"
                    else:
                        self.busstop_list[curr_stop.id].serving_period[self.simulation_step] = "conflict"

                if bus.forward_bus != None and bus.forward_bus not in self.busstop_list[bus.left_stop[0]].bus_queue \
                        and curr_stop.id not in self.bus_list[bus.forward_bus].pass_stop:
                    bus.stop(curr_stop=curr_stop)
                    continue

                if bus.id not in self.busstop_list[bus.left_stop[0]].bus_queue:
                    self.busstop_list[bus.left_stop[0]].bus_queue.append(bus.id)

                if self.max_bus_queue < len(self.busstop_list[bus.left_stop[0]].bus_queue):
                    self.max_bus_queue = len(self.busstop_list[bus.left_stop[0]].bus_queue)
                    # print(self.max_bus_queue)

                if bus.id in self.busstop_list[bus.left_stop[0]].bus_queue[:self.bay_capacity] or len(
                        bus.left_stop) == 1:
                    pass
                else:
                    bus.stop(curr_stop=curr_stop)

                    continue
                # Train & Test uncertainty
                r = np.clip(np.random.normal(1., self.accident_rate), a_min=0.1, a_max=1.5)
                bus.current_speed = r * bus.speed

                self.busstop_list[bus.left_stop[0]].arr_bus_load.append(len(bus.onboard_list))
                if bus.route_id in self.busstop_list[curr_stop.id].arr_log:
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id].append(
                        self.simulation_step)  # ([bus.id, self.simulation_step])
                else:
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id] = [
                        self.simulation_step]  # [[bus.id, self.simulation_step]]
                # 714
                self.busstop_list[curr_stop.id].uni_arr_log.append([bus.id, self.simulation_step])
                bus.arr = 1

                board_cost, alight_cost, wait_num = self.serve_by_presetOD(bus, curr_stop)

                bus.serve_remain = max(board_cost, alight_cost)

                # self.busstop_list[curr_stop.id].bus_queue.append(bus.id)

                bus.stay[curr_stop.id] = 1
                bus.cost[curr_stop.id] = bus.serve_remain
                bus.pass_stop.append(curr_stop.id)
                if bus.pass_stop[-1] != bus.stop_list[len(bus.pass_stop) - 1]:
                    print(bus.pass_stop)
                    print(bus.stop_list)
                bus.left_stop = bus.left_stop[1:]

                ## if determine holding once arriving
                if self.hold_once_arr == 1 and len(bus.pass_stop) > 1 and self.dispatch_times[bus.route_id].index(
                        bus.dispatch_time) > 0 and len(bus.left_stop) > 2:
                    if self.simulation_step in self.arrivals:
                        self.arrivals[self.simulation_step].append([curr_stop.id, bus_id, len(bus.onboard_list)])
                    else:
                        self.arrivals[self.simulation_step] = [[curr_stop.id, bus_id, len(bus.onboard_list)]]

                    bus.hold_remain = self.control(bus, curr_stop, wait_num)
                    bus.last_action_tag = self.simulation_step
                    if bus.hold_remain > 0:
                        bus.stay[curr_stop.id] = 1

                    if bus.hold_remain < 10:
                        bus.hold_remain = 0

                    bus.hold_cost[curr_stop.id] = bus.hold_remain
                    bus.is_hold = 1
            # do not operatae at last stop
            if len(bus.left_stop) == 0:
                bus.hold_remain = 0
                bus.serve_remain = 0

            if bus.hold_remain > 0 or bus.serve_remain > 0:
                # 20220714
                if self.busstop_list[bus.pass_stop[-1]].serving_period[self.simulation_step] == 0:
                    self.busstop_list[bus.pass_stop[-1]].serving_period[self.simulation_step] = bus.route_id
                else:
                    if self.busstop_list[curr_stop.id].serving_period[self.simulation_step] == bus.route_id:
                        self.busstop_list[curr_stop.id].serving_period[self.simulation_step] = "bunching"
                    else:
                        self.busstop_list[curr_stop.id].serving_period[self.simulation_step] = "conflict"
                bus.stop()
            else:
                if self.is_allow_overtake == 1:
                    bus.dep()
                else:
                    if bus.forward_bus in self.dispatch_buslist and bus.speed + bus.loc[-1] >= \
                            self.dispatch_buslist[bus.forward_bus].loc[-1]:
                        bus.stop()
                        if bus.b == 0:
                            self.bunching_times[bus.route_id] += 1
                            bus.b = 1
                    else:
                        bus.b = 0
                        if bus.arr == 1:
                            self.busstop_list[bus.pass_stop[-1]].bus_queue.remove(bus.id)

                        for p in bus.onboard_list:
                            self.pax_list[p].onroad_cost += 1
                        if len(bus.pass_stop) > 0 and bus.arr == 1:
                            if bus.route_id in self.busstop_list[bus.pass_stop[-1]].dep_log:
                                self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id].append(
                                    [bus.id, self.simulation_step])
                            else:
                                self.busstop_list[bus.pass_stop[-1]].dep_log[bus.route_id] = [
                                    [bus.id, self.simulation_step]]
                            self.busstop_list[bus.pass_stop[-1]].uni_dep_log.append(self.simulation_step)
                        bus.dep(bus.current_speed)

        self.simulation_step += 1
        # for debug cumulative arrival
        # if False and self.simulation_step % 10 == 0:
        #     for stop_id, stop in self.busstop_list.items():
        #         stop.debug_waitting(self.simulation_step)
        Flag = False
        for bus_id, bus in self.bus_list.items():

            if bus.is_dispatch != -1:
                Flag = True
        if Flag is False and self.control_type == 7:
            file = "w{}.csv".format(self.agents[0].seed)
            with open(file, 'a') as f:
                self.debug["1"] = [np.mean(self.debug["1"])]
                self.debug["2"] = [np.mean(self.debug["2"])]
                self.debug["3"] = [np.mean(self.debug["3"])]

                self.debug["4"] = [np.mean(self.debug["4"])]
                self.debug["5"] = [np.mean(self.debug["5"])]
                self.debug["6"] = [np.mean(self.debug["6"])]
                df = pd.DataFrame(self.debug)
                df.to_csv(file, mode='a', header=f.tell() == 0)
        return Flag

    def control(self, bus, bus_stop, wait_num):
        if self.control_type == 0:
            # self.rl_control(bus, bus_stop)
            return 0
        if self.control_type == 1:
            fh, bh = self.cal_headway(bus)
            if bus.forward_bus is None:
                return 0
            else:
                return min(max(0, abs(bus.dispatch_time - self.bus_list[bus.forward_bus].dispatch_time) - fh), 90.)
                # return max(0, 58  + 0.05 * (abs(bus.dispatch_time-self.bus_list[bus.forward_bus].dispatch_time) - fh))

        # Coordinated control strategy for multi-line bus bunching in common corridors [Zhou et al]
        if self.control_type == 2:
            if bus.forward_bus is None:
                return 0
            fh, bh = self.cal_headway(bus)
            delta = self.simulation_step + bus.serve_remain - bus_stop.dep_log[bus.route_id][-1][1]
            t1 = min(max(0, 1.4 * abs(bus.dispatch_time - self.bus_list[bus.forward_bus].dispatch_time) - delta), 90.)
            if bus_stop.is_in_shared_corridor == 0:
                return t1
            else:
                t2 = min(max(0, abs(bus.dispatch_time - self.bus_list[bus.forward_bus].dispatch_time) - delta), 90.)
                # find the bus from other route which is the nearest to the current stop
                for bus_route, log in bus_stop.arr_log.items():
                    if bus_route != bus.route_id:
                        break
                if bus_route == bus.route_id:
                    return t1
                t3 = max(0, 60 - (self.simulation_step + bus.serve_remain - bus_stop.dep_log[bus_route][-1][-1]))

                return min(t2, max(t1, t3))

        if self.control_type > 2:
            return self.rl_control(bus, bus_stop, wait_num)

        return 0

    def rl_control(self, bus, bus_stop, wait_num):
        # retrive historical state

        if self.share_scale == 0:
            if len(self.agents) > 0:
                is_stop_embedding = self.agents[0].is_stop_embedding
            else:
                is_stop_embedding = 0
        if self.share_scale == 1:
            is_stop_embedding = self.agents[bus.route_id].is_stop_embedding
        is_com = 0

        try:
            is_com = self.agents[0].is_com
        except:
            pass
        if bus.route_id == '22_1':
            state = [0, 1]
        else:
            state = [1, 0]
        current_interval = self.simulation_step
        if self.control_type % 2 == 0 or self.control_type == 7:
            state += self.augment_state_on_ml(bus, bus_stop)
            # print(state)
        for record in self.arrivals[current_interval]:
            bus_stop_id_ = record[0]
            bus_id_ = record[1]
            onboard = record[2]
            if bus_id_ == bus.id:
                state += [onboard / bus.capacity]
                break
        state += [wait_num / bus.capacity]

        fh, bh = self.cal_headway(bus)
        state += [min(fh / 600., 2.), min(bh / 600., 2.)]
        stop_embedding = np.array([])
        com_oneway = [0., 0., 0., 0., 0.]
        if is_com == 1 and False:
            for otherbus_id, otherbus in self.dispatch_buslist.items():
                if otherbus_id == bus.id:
                    continue
                if bus.route_id == otherbus.route_id and abs(len(bus.pass_stop) - len(otherbus.pass_stop)) <= 4:

                    com_oneway += [float(len(otherbus.pass_stop)) / float(len(otherbus.stop_list)),
                                   float(otherbus.stop_list.index(bus_stop.id)) / float(len(otherbus.stop_list)),
                                   float(len(otherbus.onboard_list)) / otherbus.capacity, 1., 0.]

                elif (bus_stop.id in otherbus.stop_list and abs(
                        len(bus.pass_stop) - otherbus.stop_list.index(bus_stop.id)) <= 4):

                    com_oneway += [float(len(otherbus.pass_stop)) / float(len(otherbus.stop_list)),
                                   float(otherbus.stop_list.index(bus_stop.id)) / float(len(otherbus.stop_list)),
                                   float(len(otherbus.onboard_list)) / otherbus.capacity, 0., 1.]

            state = [state, com_oneway]
        if is_stop_embedding == 1:
            stop_embedding = self.stop_embedding(bus, bus_stop, d_model=len(state))
            state = (np.array(state) + stop_embedding * is_stop_embedding).reshape(-1, ).tolist()
        if self.control_type == 7:
            # if len(self.GM.temp_memory[bus.id]['s']) > 2:
            #     s = torch.tensor(self.GM.temp_memory[bus.id]['s'][-3], dtype=torch.float32)
            #     w = self.PW.weight_nn.get_weights(s).detach().numpy()
            #
            # else:
            #     w = [0.4, 0.6, 1.]
            # w = self.PW.init
            w = self.APL.w_curr_mean
            if bus_stop.is_in_shared_corridor != 1:
                w = self.APL2.w_curr_mean
                w += [0.]
            try:
                state += [w[0], w[1], w[2]]
            except:
                state += [w[0], w[1], 0.]

        if self.share_scale == 0 and is_com == 0:
            if self.policy_types > 1:
                if is_stop_embedding == 1:
                    action = np.array(
                        self.agents[bus_stop.is_in_shared_corridor].choose_action(
                            [np.array(state).reshape(-1, ), stop_embedding]))
                else:
                    action = np.array(
                        self.agents[bus_stop.is_in_shared_corridor].choose_action(np.array(state).reshape(-1, )))
            else:
                if is_stop_embedding == 1:
                    action = np.array(self.agents[0].choose_action([np.array(state).reshape(-1, ), stop_embedding]))
                else:
                    action = np.array(self.agents[0].choose_action(np.array(state).reshape(-1, )))

        if self.share_scale == 1 and self.agents[bus.route_id].is_com < 1:
            if is_stop_embedding == 1:
                action = np.array(
                    self.agents[bus.route_id].choose_action([np.array(state).reshape(-1, ), stop_embedding]))
            else:
                action = np.array(self.agents[bus.route_id].choose_action(np.array(state).reshape(-1, )))
        if is_com == 1:
            action = np.array(self.agents[0].choose_action(state))
        # action = action + np.random.normal(0,0.02)

        mark = list(np.array(state + list(action)).reshape(-1, ))
        self.bus_list[bus.id].his[current_interval] = mark

        record_id = bus.id
        if len(self.GM.temp_memory[record_id]['a']) > 0:
            # organize fingerprint: consider impact of other agent between two consecutive control of the ego agent
            stop_dist = [0.]
            bus_dist = [0.]
            if is_com == 1:
                fp = [self.GM.temp_memory[record_id]['s'][-1][0] + self.GM.temp_memory[record_id]['a'][
                    -1].tolist() + stop_dist + bus_dist + [0.] + [bus.id]]
            else:
                fp = [self.GM.temp_memory[record_id]['s'][-1] + self.GM.temp_memory[record_id]['a'][
                    -1].tolist() + stop_dist + bus_dist + [0.] + [bus.id]]
            temp = bus.last_vist_interval

            while temp <= current_interval:
                if temp in self.arrivals:
                    for record in self.arrivals[temp]:
                        bus_stop_id_ = record[0]
                        bus_id_ = record[1]
                        onboard = record[2]
                        if bus_id_ == bus.id or self.bus_list[bus_id_].route_id != bus.route_id:
                            continue

                        if (bus_id_ == bus.forward_bus or bus_id_ == bus.backward_bus) or (self.all == 1):
                            curr_bus = self.dispatch_times[bus.route_id].index(bus.dispatch_time)
                            neigh_bus = self.dispatch_times[bus.route_id].index(self.bus_list[bus_id_].dispatch_time)
                            bus_dist = [(curr_bus - neigh_bus) / len(self.bus_list)]
                            stop_dist = [
                                (bus.stop_list.index(bus.pass_stop[-2]) - bus.stop_list.index(bus_stop_id_)) / len(
                                    self.busstop_list)]
                            if is_com == 1:
                                fp.append(self.bus_list[bus_id_].his[temp][0] + [
                                    self.bus_list[bus_id_].his[temp][-1]] + stop_dist + bus_dist + [
                                              abs(temp - current_interval)] + [bus_id_])
                            else:
                                fp.append(self.bus_list[bus_id_].his[temp] + stop_dist + bus_dist + [
                                    abs(temp - current_interval)] + [bus_id_])

                temp += 1

            reward1, reward2, reward3, reward = self.reward_func(bus, bus_stop)

            self.reward_record.append(reward)
            self.reward_signal[bus.id].append(reward)
            self.reward_signalp1[bus.id].append(reward1)
            self.reward_signalp2[bus.id].append(reward2)
            try:
                self.reward_signalp3[bus.id].append(reward3)
            except:
                pass
            if self.control_type == 7:
                self.GM.temp_memory[record_id]['r1'].append(reward1)
                self.GM.temp_memory[record_id]['r2'].append(reward2)
                self.GM.temp_memory[record_id]['r3'].append(reward3)
            else:
                self.GM.temp_memory[record_id]['r'].append(reward)
            self.GM.temp_memory[record_id]['fp'].append(fp)

        ## update temporal memory with current state and action and mark
        self.GM.temp_memory[record_id]['s'].append(state)
        self.GM.temp_memory[record_id]['a'].append(action)
        self.GM.temp_memory[record_id]['stop_embed'].append(stop_embedding.tolist())
        if len(self.GM.temp_memory[record_id]['s']) > 2:
            s = self.GM.temp_memory[record_id]['s'][-3]
            ns = self.GM.temp_memory[record_id]['s'][-2]
            fp = self.GM.temp_memory[record_id]['fp'][-2]
            nfp = self.GM.temp_memory[record_id]['fp'][-1]
            a = self.GM.temp_memory[record_id]['a'][-3]
            if self.control_type == 7:
                r = [self.GM.temp_memory[record_id]['r1'][-2],
                     self.GM.temp_memory[record_id]['r2'][-2],
                     self.GM.temp_memory[record_id]['r3'][-2]]
            else:
                r = self.GM.temp_memory[record_id]['r'][-2]
            stop_embed = self.GM.temp_memory[record_id]['stop_embed'][-3]
            next_stop_embed = self.GM.temp_memory[record_id]['stop_embed'][-2]
            self.GM.remember(s, fp, a, r, ns, nfp, stop_embed, next_stop_embed, record_id)
            if self.control_type == 7:
                if bus_stop.is_in_shared_corridor != 1:
                    self.APL2.update(member_id=record_id, r1=np.array(reward1).reshape(-1, ),
                                    r2=np.array(reward2).reshape(-1, ), r3=np.array(reward3).reshape(-1, ),
                                    s=np.array(s).reshape(-1, ))
                else:
                    self.APL.update(member_id=record_id, r1=np.array(reward1).reshape(-1, ),
                                    r2=np.array(reward2).reshape(-1, ), r3=np.array(reward3).reshape(-1, ),
                                    s=np.array(s).reshape(-1, ))
        self.action_record.append(action)
        action = np.clip(abs(action), 0., 3.)
        self.bus_list[bus.id].last_vist_interval = current_interval
        return 180. * action

    def cal_headway(self, bus):

        if bus.forward_bus != None and bus.forward_bus in self.dispatch_buslist:
            fh = abs(bus.loc[-1] - self.bus_list[bus.forward_bus].loc[-1]) / bus.c_speed

        else:
            fh = abs(bus.loc[-1] - 0.) / bus.c_speed

        if bus.backward_bus != None and bus.backward_bus in self.dispatch_buslist:
            bh = abs(bus.loc[-1] - self.bus_list[bus.backward_bus].loc[-1]) / bus.c_speed

        else:
            bh = abs(bus.loc[-1] - 0.) / bus.c_speed
            if self.dispatch_times[bus.route_id].index(bus.dispatch_time) == len(self.dispatch_times[bus.route_id]) - 1:
                bh = 0.

        return fh, bh

    def route_info(self, bus, is_this_route=1):
        # TO update
        fh = [500 for _ in range(50)]
        bh = [500 for _ in range(50)]
        fh = []
        bh = []
        for bus_id, bus_ in self.dispatch_buslist.items():
            if is_this_route == 1:
                if bus_.route_id == bus.route_id:
                    if bus_.forward_bus != None:
                        fh.append(abs(bus_.loc[-1] - self.bus_list[bus_.forward_bus].loc[-1]) / bus_.speed)
                    if bus_.backward_bus != None:
                        bh.append(abs(bus_.loc[-1] - self.bus_list[bus_.backward_bus].loc[-1]) / bus_.speed)
            else:
                if bus_.route_id != bus.route_id:
                    if bus_.forward_bus != None:
                        fh.append(abs(bus_.loc[-1] - self.bus_list[bus_.forward_bus].loc[-1]) / bus_.speed)
                    if bus_.backward_bus != None:
                        bh.append(abs(bus_.loc[-1] - self.bus_list[bus_.backward_bus].loc[-1]) / bus_.speed)
        # TO MODIFY
        if len(bh) < 2:
            return 999999, 999999

        return np.var(bh), np.mean(bh)

    def share_route_info(self, ):
        bus_in_share_corridor = []
        for bus_id, bus in self.dispatch_buslist.items():
            if len(bus.left_stop) >= 1 and bus.left_stop[0] in self.shared_stops and bus.pass_stop[
                -1] in self.shared_stops:
                bus_in_share_corridor.append(bus.loc[-1] / bus.speed)
        bus_in_share_corridor = np.array(sorted(bus_in_share_corridor)).reshape(-1, )
        headways = list(bus_in_share_corridor[1:] - bus_in_share_corridor[:-1])
        headways = np.array(headways)
        print(headways.shape)
        print(np.var(headways), np.mean(headways))
        return np.var(headways) + np.mean(headways)

    def learn(self):
        ploss_set = []
        qloss_set = []
        if self.policy_types <= 1:
            if self.share_scale == 0:
                for rid, r in self.route_list.items():
                    b = np.random.randint(1, len(r.bus_list) - 1)
                    bus_id = r.bus_list[b]
                    if self.control_type == 7:
                        ploss, qloss = self.agents[0].learn(self.GM.memory[bus_id], bus_id=bus_id,
                                                            weight_leaner=[self.APL2, self.APL])
                    else:
                        ploss, qloss = self.agents[0].learn(self.GM.memory[bus_id], bus_id=bus_id)
                    # self.GM.memory[bus_id].clear()
                    try:
                        self.qloss[bus_id].append(np.mean(qloss))
                    except:
                        self.qloss[bus_id] = [np.mean(qloss)]
                    ploss_set.append(ploss)
                    qloss_set.append(qloss)

            if self.share_scale == 1:
                for rid, r in self.route_list.items():
                    b = np.random.randint(0, len(r.bus_list))
                    bus_id = r.bus_list[b]
                    while self.bus_list[bus_id].forward_bus == None or len(self.GM.memory[bus_id]) <= 0 or \
                            self.bus_list[
                                bus_id].backward_bus == None:
                        if b + np.random.randint(0, len(r.bus_list) - b) >= len(r.bus_list):
                            b = 0
                        bus_id = r.bus_list[b + np.random.randint(0, len(r.bus_list) - b)]

                    ploss, qloss = self.agents[rid].learn(self.GM.memory[bus_id], bus_id=bus_id)
                    # self.GM.memory[bus_id].clear()
                    try:
                        self.qloss[bus_id].append(np.mean(qloss))
                    except:
                        self.qloss[bus_id] = [np.mean(qloss)]
                    ploss_set.append(ploss)
                    qloss_set.append(qloss)

        else:
            if self.share_scale == 0:
                for rid, r in self.route_list.items():
                    for i in range(2):
                        b = np.random.randint(1, len(r.bus_list) - 1)
                        bus_id = r.bus_list[b]
                        record_id = str(bus_id) + str('_') + str(i)

                        ploss, qloss = self.agents[i].learn(self.GM.memory[record_id], bus_id=bus_id)
                    # self.GM.memory[bus_id].clear()
                    try:
                        self.qloss[bus_id].append(np.mean(qloss))
                    except:
                        self.qloss[bus_id] = [np.mean(qloss)]
                    ploss_set.append(ploss)
                    qloss_set.append(qloss)

            if self.share_scale == 1:
                for rid, r in self.route_list.items():
                    b = np.random.randint(1, len(r.bus_list) - 1)
                    bus_id = r.bus_list[b]
                    ploss, qloss = self.agents[rid].learn(self.GM.memory[bus_id], bus_id=bus_id)

                    try:
                        self.qloss[bus_id].append(np.mean(qloss))
                    except:
                        self.qloss[bus_id] = [np.mean(qloss)]
                    ploss_set.append(ploss)
                    qloss_set.append(qloss)

        if np.mean(ploss_set) != 0 and len(self.reward_signal) > 0:

            return np.mean(ploss_set), np.mean(qloss_set), True
        else:
            return 0, 0, False

    def augment_state_on_ml(self, bus, bus_stop):
        ag_state = [2. for _ in range(2)]
        if bus_stop.is_in_shared_corridor == 1:
            rid = [rr for rr in list(bus_stop.loc_route.keys()) if rr != bus.route_id][0]
            if rid in bus_stop.dep_log and len(bus_stop.dep_log[rid]) > 0:
                prebus = bus_stop.dep_log[rid][-1][0]
                while prebus != None:
                    prebus = self.bus_list[prebus]
                    if prebus.loc[-1] < bus_stop.loc_route[rid]:
                        ag_state[0] = min(ag_state[0], (bus_stop.loc_route[rid] - prebus.loc[-1]) / bus.c_speed / 600.)
                        break
                    prebus = prebus.backward_bus
            if rid in bus_stop.dep_log and len(bus_stop.dep_log[rid]) > 0:
                ag_state[1] = min(ag_state[1], (
                        self.bus_list[bus_stop.dep_log[rid][-1][0]].loc[-1] - bus_stop.loc_route[
                    rid]) / bus.c_speed / 600.)
        # print(ag_state)
        return ag_state

    def reward_func(self, bus, bus_stop):
        # if self.policy_types > 1:
        #     record_id = str(bus.id) + '_' + str(bus_stop.is_in_shared_corridor)
        # else:
        record_id = bus.id

        reward3 = 0.
        var, mean = self.route_info(bus)
        reward1 = (-var / mean / mean) * 5
        reward2 = (-abs(self.GM.temp_memory[record_id]['a'][-1]))

        if bus_stop.is_in_shared_corridor == 1:
            # 714
            nearest_front = 1000000000
            nearest_back = 1000000000
            other_route = None
            for r in bus_stop.routes:
                if r != bus.route_id:
                    other_route = r
            for other_bus_id, other_bus in self.bus_list.items():
                if other_bus.route_id == other_route:
                    if other_bus.loc[-1] > bus.loc[-1] and (other_bus.loc[-1] - bus.loc[-1]) < nearest_front:
                        nearest_front = (other_bus.loc[-1] - bus.loc[-1])
                    if other_bus.loc[-1] < bus.loc[-1] and (bus.loc[-1] - other_bus.loc[-1]) < nearest_back:
                        nearest_back = (bus.loc[-1] - other_bus.loc[-1])
            if nearest_front == 1000000000 or nearest_back == 1000000000:
                reward3 = 0.
            else:
                reward3 = np.exp((-abs(nearest_front - nearest_back)))
            # middles = []
            # for rid in list(bus_stop.arr_log.keys()):
            #     if len(bus_stop.arr_log[rid]) > 1:
            #         for m in bus_stop.arr_log[rid]:
            #             if m <= bus_stop.arr_log[bus.route_id][-1] and m >= bus_stop.arr_log[bus.route_id][-2]:
            #                 middles.append(m)
            # # middles += [bus_stop.arr_log[bus.route_id][-2], bus_stop.arr_log[bus.route_id][-1]]
            # middles = np.array(sorted(middles)).reshape(-1, ) / 3600.
            # intervals = list(middles[1:] - middles[:-1])
            # reward3 = H(softmax(intervals))
        # self.debug["1"].append(reward1)
        # self.debug["2"].append(reward2)
        # self.debug["3"].append(reward3)

        if self.control_type < 5:
            reward = reward1 * (1 - self.weight) + reward2 * self.weight
        elif self.control_type < 7:
            reward = reward1 * (1 - self.weight) + reward2 * self.weight + reward3
        else:
            # if len(self.GM.temp_memory[bus.id]['s']) > 2:
            #     s = torch.tensor(self.GM.temp_memory[bus.id]['s'][-3], dtype=torch.float32)
            #     w = self.PW.weight_nn.get_weights(s).detach().numpy()
            #     self.debug["1"].append(w[0])
            #     self.debug["2"].append(w[1])
            #     self.debug["3"].append(w[2])
            # else:
            #     w = [0.4, 0.6, 1.]
            # w = self.PW.init
            if bus_stop.is_in_shared_corridor != 1:
                w = self.APL2.w_curr_mean
                self.debug["4"].append(w[0])
                self.debug["5"].append(w[1])
                self.debug["6"].append(0.)
            else:
                w = self.APL.w_curr_mean
                self.debug["1"].append(w[0])
                self.debug["2"].append(w[1])
                self.debug["3"].append(w[2])
            try:
                reward = w[0] * reward1 + reward2 * w[1] + reward3 * w[2]
            except:
                reward = w[0] * reward1 + reward2 * w[1] + reward3 * 0.
        return reward1, reward2, reward3, reward

    def stop_embedding(self, bus, stop, d_model):
        position = np.arange(0, len(bus.stop_list))[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        v1 = np.sin(position * div_term)[bus.stop_list.index(stop.id), :]
        v2 = np.cos(position * div_term)[bus.stop_list.index(stop.id), :]
        return np.concatenate([v1, v2])
