import numpy as np
from sim.Passenger import Passenger
from sim.Route import Route
import matplotlib.pyplot as plt
from model.Group_MemoryC import Memory
import pandas as pd
import time
import math
import random
class Engine():
    def __init__(self, bus_list, busstop_list, route_list, simulation_step, dispatch_times, demand=0, agents=None,
                 share_scale=0, is_allow_overtake=0, hold_once_arr=1, control_type=1, seed=1, all=0, weight=0,demand_abnormal = 0,
                                state_abnormal = 0):

        self.all = all
        self.busstop_list = busstop_list
        self.simulation_step = simulation_step
        self.pax_list = {}  # passenger on road
        self.arr_pax_list = {}  # passenger finihsed trip
        self.dispatch_buslist = {}
        self.agents = {}
        self.route_list = route_list
        self.dispatch_buslist = {}
        self.is_allow_overtake = is_allow_overtake
        self.hold_once_arr = hold_once_arr
        self.control_type = control_type
        self.agents = agents
        self.bus_list = bus_list
        self.bunching_times = 0
        self.arrstops = 0
        self.reward_signal = {}
        self.reward_signalp1 = {}
        self.reward_signalp2 = {}
        self.qloss = {}
        self.weight = weight / 10.
        self.demand = demand
        self.records = []
        self.share_scale = share_scale
        self.step = 0
        self.dispatch_times = dispatch_times
        self.cvlog = []
        self.demand_abnormal = demand_abnormal
        self.state_abnormal = state_abnormal
        abnormal_stops = [3]
        while len(abnormal_stops)<self.demand_abnormal or len(abnormal_stops)<self.state_abnormal:
            abnormal_stops.append(abnormal_stops[-1]+np.random.randint(1,5))
        self.demand_abnormal_stops = abnormal_stops#random.sample(numbers, self.demand_abnormal)#np.random.randint(2,20,self.demand_abnormal).tolist()
        self.state_abnormal_stops = abnormal_stops#random.sample(numbers, self.state_abnormal)# np.random.randint(2,20,self.state_abnormal).tolist()
        if len(abnormal_stops)>0:
            print(abnormal_stops)
        members = list(self.bus_list.keys())
        self.GM = Memory(members)

        for b_id, b in self.bus_list.items():
            self.reward_signal[b_id] = []
            self.reward_signalp1[b_id] = []
            self.reward_signalp2[b_id] = []

        self.arrivals = {}

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

        self.action_record = []
        self.reward_record = []
        self.state_record = []
        self.accident_rate = 0.# np.random.randint(1, 10) / 100.
        self.sensor_error_rate = 0.
        self.demand_impulse_rate= 0.



    def cal_statistic(self,name,train=1):
        print('total pax:%d'%(len(self.pax_list)))

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
        stop_miss = {}
        for pax_id, pax in self.pax_list.items():
            w = min(pax.onboard_time - pax.arr_time,self.simulation_step-pax.arr_time)
            miss.append(pax.miss)
            # print(pax.onboard_time,pax.arr_time,w)
            # if pax.origin not in stop_miss:
            #     stop_miss[pax.origin]=0
            # else:
            #     stop_miss[pax.origin]+=1
            wait_cost.append(w)
            if pax.origin in stop_wise_wait:
                stop_wise_wait[pax.origin].append(w)
            else:
                stop_wise_wait[pax.origin]=[w]
            if pax.onboard_time<99999999:
                # boards.append(pax.onboard_time )
                if pax.alight_time<999999:
                    travel_cost.append(pax.alight_time-pax.onboard_time )
                    delay.append(pax.alight_time-pax.arr_time-pax.onroad_cost)

            else:
               still_wait+=1
        # plt.scatter(list(stop_miss.keys()),list(stop_miss.values()))
        # plt.show()
        print('MISS:%g'%(np.max(miss)-1))
        hold_cost = []
        for bus_id, bus in self.bus_list.items():
            tt = [ ]
            for k,v in bus.stay.items():
                if v>0:
                    tt.append(bus.hold_cost[k])
                    hold_cost.append(bus.hold_cost[k])
                    if k in stop_wise_hold:
                        stop_wise_hold[k].append(bus.hold_cost[k])
                    else:
                        stop_wise_hold[k] = [bus.hold_cost[k]]
                else:
                    print(bus.id)

        stop_wise_wait_order = []
        stop_wise_hold_order = []

        arr_times = []
        buslog = pd.DataFrame()
        for bus_stop_id in bus.pass_stop:
            # lists = sorted(self.busstop_list[bus_stop_id].actual_arrivals.items())  # sorted by key, return a list of tuples
            # x, y = zip(*lists)  # unpack a list of pairs into two tuples
            # plt.plot(x,y,'-*',color='red',label='cumulative actural arrivals')
            #
            # lists = sorted(
            #     self.busstop_list[bus_stop_id].actual_departures.items())  # sorted by key, return a list of tuples
            # x, y = zip(*lists)  # unpack a list of pairs into two tuples
            # plt.plot(x, y, '-o',color='blue',label='cumulative actural departures')
            #
            # lists = sorted(
            #     self.busstop_list[bus_stop_id].arrivals.items())  # sorted by key, return a list of tuples
            # x, y = zip(*lists)  # unpack a list of pairs into two tuples
            # plt.plot(x, y, color='green', label='cumulative pre-arrivals')
            # plt.legend()
            # plt.show()

            buslog[bus_stop_id]=self.busstop_list[bus_stop_id].arr_log[bus.route_id]
            arr_times.append([bus_stop_id]+self.busstop_list[bus_stop_id].arr_log[bus.route_id])

            try:
                stop_wise_wait_order.append(np.mean(stop_wise_wait[ bus_stop_id ]))
            except:
                stop_wise_wait_order.append(0)
            try:
                stop_wise_hold_order.append(np.mean(stop_wise_hold[bus_stop_id]))
            except:
                stop_wise_hold_order.append(0)

            for k,v in self.busstop_list[bus_stop_id].arr_log.items():

                h =  np.array(v )[1:]  -  np.array(v)[:-1]
                if np.isnan(np.var(h)):
                    continue
                try:
                    headways_var[bus_stop_id].append(np.var(h))
                    headways_mean[bus_stop_id].append(np.mean(h))
                except:
                    headways_var[bus_stop_id]=[np.var(h)]
                    headways_mean[bus_stop_id]=[np.mean(h)]
        # plt.hist(wait_cost)
        # plt.show()
        log = {}
        log['wait_cost'] = wait_cost
        log['travel_cost'] = travel_cost
        log['hold_cost'] = hold_cost
        log['headways_var'] = headways_var
        log['headways_mean'] = headways_mean
        log['stw'] = stop_wise_wait_order
        log['sth'] = stop_wise_hold_order
        log['bunching'] = self.bunching_times
        log['delay'] = delay
        print('bunching times:%g headway mean:%g hedaway var:%g EV:%g'%(self.bunching_times, np.mean(list(headways_mean.values())),np.mean(list(headways_var.values())), (np.mean(list(headways_var.values()))/(np.mean(list(headways_mean.values()))**2))   ))

        AWT = []
        AHD = []
        AOD = []
        for k in bus.pass_stop:
            AHD.append(np.mean(stop_wise_hold[k]))
            # AWT.append(np.mean(stop_wise_wait[k]))

            if math.isnan(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load)):
                AOD.append(0)
            else:
                AOD.append(np.var(self.busstop_list[k].arr_bus_load) / np.mean(self.busstop_list[k].arr_bus_load))

            try:
                AWT.append(np.mean(stop_wise_wait[k]))
            except:
                AWT.append(0.)
        log['sto'] = AOD
        log['AOD'] = np.mean(AOD)
        if train==0  :
            print(np.max(wait_cost))
            print('AWT:%g'%(np.mean(wait_cost)))
            print('AHD:%g' % (np.mean(AHD)))
            print('AOD:%g' % (np.mean(AOD)))
            print('headways_var:%g' % (np.sqrt(np.mean(list(headways_var.values())))))

        log['arr_times'] = arr_times
        action_record = pd.DataFrame()
        action_record['a']=self.action_record
        # action_record.to_csv('a.csv')
        return log

    def close(self):
        return

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

            # Simulate burst
            if self.demand_abnormal>=1:
                if (len(bus.pass_stop)) in self.demand_abnormal_stops and  self.simulation_step -24149>3600*3 and self.simulation_step-24149<3600*4:
                # if ((len(bus.pass_stop) == 10 or len(bus.pass_stop) == 19 or len(bus.pass_stop) == 8 )and  self.simulation_step -24149>3600*6 and self.simulation_step-24149<3600*7) :
                   bus.abnormal = 1.
                   for i in range(50):
                       p = Passenger(id=-i,arr_time=self.simulation_step,origin=stop.id )
                       p.dest = bus.left_stop[np.random.randint(0,len(bus.left_stop))]
                       new_pax.append(p)
                else:
                    bus.abnormal = 0

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
                if bus != None and bus.capacity > len(bus.onboard_list)  and self.pax_list[
                    num].route == bus.route_id:
                    self.pax_list[num].onboard_time = self.simulation_step
                    bus.onboard_list.append(num)
                    board_cost += bus.board_period
                    pax_leave_stop.append(num)
                    self.busstop_list[stop.id].cum_dep += 1

            for num in pax_leave_stop:
                self.busstop_list[stop.id].waiting_list.remove(num)

        self.busstop_list[stop.id].actual_departures[self.simulation_step] = self.busstop_list[stop.id].cum_dep
        self.busstop_list[stop.id].actual_arrivals[self.simulation_step] = self.busstop_list[stop.id].cum_arr
        return alight_cost, board_cost, wait_num

    def serve(self,bus,stop):
        board_cost = 0
        alight_cost = 0
        board_pax = []
        alight_pax = []
        if bus!=None:

            alight_pax = bus.pax_alight_fix(stop, self.pax_list)
            for p in alight_pax:
                self.pax_list[p].alight_time = self.simulation_step
                bus.onboard_list.remove(p)
                self.arr_pax_list[p] = self.pax_list[p]

            alight_cost = len(alight_pax) * bus.alight_period

            # boarding procedure
            for d in stop.dest.keys():
                # new_arr = stop.pax_gen_od(bus, sim_step=self.simulation_step,dest_id=d)
                new_arr = stop.pax_gen_od(bus, demand_impulse_rate=self.demand_impulse_rate, sim_step=self.simulation_step,dest_id=d)
                if len(new_arr)==0:
                    continue
                num = len(self.pax_list) + 1
                for t in new_arr:
                    self.pax_list[num] = Passenger(id=num, origin=stop.id, arr_time=t)
                    self.pax_list[num].took_bus = bus.id
                    self.pax_list[num].route = bus.route_id
                    self.pax_list[num].dest= d
                    self.busstop_list[stop.id].waiting_list.append(num)
                    num += 1
            pax_leave_stop = []
            waitinglist = sorted(self.busstop_list[stop.id].waiting_list)[:]
            for num in waitinglist:
                # add logic to consider multiline impact (i.e. the passenger can not board bus this time can board the bus with same destination later?)
                if bus != None and self.pax_list[
                    num].route == bus.route_id:
                    self.pax_list[num].miss += 1
                if bus != None and bus.capacity - len(bus.onboard_list) > 0 and self.pax_list[
                    num].route == bus.route_id:
                    self.pax_list[num].onboard_time = self.simulation_step
                    bus.onboard_list.append(num)
                    board_cost += bus.board_period
                    pax_leave_stop.append(num)

            for num in pax_leave_stop:
                self.busstop_list[stop.id].waiting_list.remove(num)

        wait_num = len(stop.waiting_list)

        return alight_cost,board_cost,wait_num


    def sim(self):
        # update bus state
        ## dispatch bus
        Flag = False

        for bus_id, bus in self.bus_list.items():
            if bus.is_dispatch == 0 and bus.dispatch_time <= self.simulation_step:
                bus.is_dispatch = 1
                bus.current_speed = bus.speed * np.random.randint(60., 120.) / 100.
                self.dispatch_buslist[bus_id] = bus
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
            bus.serve_remain = max(bus.serve_remain - 1,0)
            bus.hold_remain = max(bus.hold_remain - 1, 0)

            ### on-arrival
            if bus.arr==0 and abs(bus.loc[-1]-bus.stop_dist[bus.left_stop[0]])<bus.speed :
                # Train & Test uncertainty
                r = np.clip(np.random.normal(1.,self.accident_rate),a_min=0.1,a_max=1.5)
                bus.current_speed = r*bus.speed
                #
                # print(bus.current_speed,self.accident_rate,bus.speed)
                ## Test for vis
                bus.abnormal = 0
                if self.state_abnormal>=1:
                    if (len(bus.pass_stop)) in self.state_abnormal_stops and self.simulation_step -24149>3600*3 and self.simulation_step-24149<3600*4:
                    # # if (len(bus.pass_stop) ==30 and int(bus.id) ==6263) or (len(bus.pass_stop) ==12 and int(bus.id) ==7594) \
                    # #         or (len(bus.pass_stop) ==24 and int(bus.id) ==16227) or (len(bus.pass_stop) ==6 and int(bus.id) ==2259):
                        bus.current_speed = bus.speed* 0.1
                        bus.abnormal = 1
                    # print('accident',self.accident,self.count)
                #### determine boarding and alight cost
                if bus.left_stop[0] not in self.busstop_list:
                    self.busstop_list[bus.left_stop[0]] = self.busstop_list[bus.left_stop[0].split('_')[0]]

                curr_stop = self.busstop_list[bus.left_stop[0]]
                self.busstop_list[bus.left_stop[0]].arr_bus_load.append(len(bus.onboard_list))
                if bus.route_id in self.busstop_list[curr_stop.id].arr_log:
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id].append(self.simulation_step)#([bus.id, self.simulation_step])
                else:
                    self.busstop_list[curr_stop.id].arr_log[bus.route_id] =[self.simulation_step]# [[bus.id, self.simulation_step]]
                board_cost,alight_cost,wait_num = self.serve_by_presetOD(bus,curr_stop)

                bus.arr=1

                bus.serve_remain = max(board_cost,alight_cost)

                bus.stay[curr_stop.id] = 1
                bus.cost[curr_stop.id] = bus.serve_remain
                bus.pass_stop.append(curr_stop.id)

                bus.left_stop = bus.left_stop[1:]

                ## if determine holding once arriving
                if self.hold_once_arr==1 and len(bus.pass_stop)>1 and self.dispatch_times[bus.route_id].index(bus.dispatch_time)>0 and len(bus.left_stop)>2:
                    if self.simulation_step in self.arrivals:
                        self.arrivals[self.simulation_step].append([curr_stop.id, bus_id, len(bus.onboard_list)])
                    else:
                        self.arrivals[self.simulation_step] = [[curr_stop.id, bus_id, len(bus.onboard_list)]]

                    bus.hold_remain = self.control(bus, curr_stop,wait_num,type=self.control_type)

                    if bus.hold_remain > 0:
                        bus.stay[curr_stop.id] = 1

                    if bus.hold_remain<10:
                        bus.hold_remain = 0

                    bus.hold_cost[curr_stop.id] = bus.hold_remain
                    bus.is_hold = 1
            if len(bus.left_stop)==0:
                bus.hold_remain = 0
                bus.serve_remain = 0
            if bus.hold_remain>0 or bus.serve_remain>0:
                bus.stop()

            else:
                if self.is_allow_overtake == 1:
                    bus.dep()
                else:
                    if bus.forward_bus in self.dispatch_buslist and bus.speed+bus.loc[-1]>=self.dispatch_buslist[bus.forward_bus].loc[-1]:
                        bus.stop()

                        if bus.b==0:
                            self.bunching_times+=1
                            bus.b=1
                    else:
                        bus.b = 0

                        bus.dep(bus.current_speed)
                        for p in bus.onboard_list:
                            self.pax_list[p].onroad_cost+=1
                        if len(bus.pass_stop)>0:
                            if bus.route_id in self.busstop_list:
                                self.busstop_list[bus.pass_stop[-1] ].dep_log[bus.route_id].append([bus.id, self.simulation_step])
                            else:
                                self.busstop_list[bus.pass_stop[-1] ].dep_log[bus.route_id] = [[bus.id, self.simulation_step]]

        self.simulation_step+=1
        Flag =False
        for bus_id, bus in self.bus_list.items():
            if bus.is_dispatch!=-1:
                Flag = True

        return Flag

    def control(self,bus,bus_stop,wait_num,type=0):
        if type==0:
            return 0
        if type==1:
            fh, bh = self.cal_headway(bus )
            if bus.forward_bus==None:
                return 0
            else:
                # return min(max(0, abs(bus.dispatch_time - self.bus_list[bus.forward_bus].dispatch_time) - fh), 90.)
                return max(0, 58 + 0.05 * ( (abs(bus.dispatch_time - self.bus_list[bus.forward_bus].dispatch_time) - fh)))#

        if type==2:
            return self.rl_control(bus,bus_stop,wait_num)

        return 0

    def rl_control(self, bus, bus_stop,wait_num):
        # retrive historical state
        current_interval = self.simulation_step


        state = []
        for record in self.arrivals[current_interval]:
            bus_stop_id_ = record[0]
            bus_id_ = record[1]
            onboard = record[2]
            if bus_id_ == bus.id:
                state = [onboard / bus.capacity]
                break

        fh, bh = self.cal_headway(bus)
        var, mean = self.route_info(bus)
        state += [wait_num/10.]

        state += [min(fh / 600., 2.), min(bh / 600., 2.)]

        self.state_record.append(state)
        if self.share_scale == 0:
            action = np.array(self.agents[bus.id].choose_action(np.array(state).reshape(-1, )))

        if self.share_scale == 1:
            action = np.array(self.agents[bus.route_id].choose_action(np.array(state).reshape(-1, )))

        mark = list(np.array(state + list(action)).reshape(-1, ))
        self.bus_list[bus.id].his[self.simulation_step] = mark

        if len(self.GM.temp_memory[bus.id]['a']) > 0:
            # organize fingerprint: consider impact of other agent between two consecutive control of the ego agent
            stop_dist = [0.]
            bus_dist = [0.]

            fp = [self.GM.temp_memory[bus.id]['s'][-1] +self.GM.temp_memory[bus.id]['a'][-1].tolist()+ stop_dist + bus_dist + [0.] + [bus.id]]
            temp = bus.last_vist_interval

            while temp <= current_interval:
                if temp in self.arrivals:
                    for record in self.arrivals[temp]:
                        bus_stop_id_ = record[0]
                        bus_id_ = record[1]
                        onboard = record[2]
                        if bus_id_ == bus.id:
                            continue

                        if (bus_id_ == bus.forward_bus or bus_id_ == bus.backward_bus) or (self.all == 1):
                            curr_bus = self.dispatch_times[bus.route_id].index(bus.dispatch_time)
                            neigh_bus = self.dispatch_times[bus.route_id].index(self.bus_list[bus_id_].dispatch_time)
                            bus_dist = [(curr_bus - neigh_bus) / len(self.bus_list)]
                            stop_dist = [
                                (bus.stop_list.index(bus.pass_stop[-2]) - bus.stop_list.index(bus_stop_id_)) / len(
                                    self.busstop_list)]
                            # equipment anomaly
                            if  self.sensor_error_rate>np.random.randn():
                                continue
                            fp.append(self.bus_list[bus_id_].his[temp] + stop_dist + bus_dist + [
                                abs(temp - current_interval)] + [bus_id_])

                temp += 1
            weight = self.weight
            reward1 = (-var / mean / mean) * (1 - weight) * 5
            reward2 = (-abs(self.GM.temp_memory[bus.id]['a'][-1])) *  weight
            reward = reward1 + reward2
            # # Debug the stability of cv
            # if len(bus.left_stop)==2:
            #     self.cvlog.append(reward1)
            self.reward_record.append(reward)
            self.reward_signal[bus.id].append(reward)
            self.reward_signalp1[bus.id].append(reward1)
            self.reward_signalp2[bus.id].append(reward2)

            self.GM.temp_memory[bus.id]['r'].append(reward)
            self.GM.temp_memory[bus.id]['fp'].append(fp)
            ##########################################
            # if self.share_scale == 0:
            #     self.agents[bus.id].record_meta(fp,bus.id)
            # if self.share_scale == 1:
            #     self.agents[bus.route_id].record_meta(fp, bus.id)


        ## update temporal memory with current state and action and mark
        self.GM.temp_memory[bus.id]['s'].append( state)
        self.GM.temp_memory[bus.id]['a'].append(action)
        if len(bus.left_stop) <1:
            self.GM.temp_memory[bus.id]['d'].append(1.)
        else:
            self.GM.temp_memory[bus.id]['d'].append(0.)
        if len(self.GM.temp_memory[bus.id]['s']) > 2:
            s = self.GM.temp_memory[bus.id]['s'][-3]
            ns = self.GM.temp_memory[bus.id]['s'][-2]
            fp = self.GM.temp_memory[bus.id]['fp'][-2]
            nfp = self.GM.temp_memory[bus.id]['fp'][-1]
            a = self.GM.temp_memory[bus.id]['a'][-3]
            r = self.GM.temp_memory[bus.id]['r'][-2]
            done = self.GM.temp_memory[bus.id]['d'][-1]

            self.GM.remember(s, fp, a, r, ns, nfp,0., bus.id)

        action = np.clip(abs(action), 0., 1 )
        self.action_record.append(action)
        self.bus_list[bus.id].last_vist_interval = self.simulation_step
        return 120. * action

    def cal_headway(self,bus):

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

    def route_info(self,bus):
        fh = [500 for _ in range(10)]
        bh = [500 for _ in range(10)]
        for bus_id, bus_ in self.dispatch_buslist.items():
            if bus_.route_id == bus.route_id:
                if bus_.forward_bus != None and bus_.forward_bus in self.dispatch_buslist:
                    fh.append(abs(bus_.loc[-1] - self.bus_list[bus_.forward_bus].loc[-1]) / bus_.speed)
                if bus_.backward_bus != None and bus_.backward_bus in self.dispatch_buslist:
                    bh.append(abs(bus_.loc[-1] - self.bus_list[bus_.backward_bus].loc[-1]) / bus_.speed)
        # TO MODIFY
        #  print(len(bh),np.var(bh)/ np.mean(bh)/ np.mean(bh))
        #  print(bh)
        if len(bh) < 2:
            return 999999, 999999

        return np.var(bh), np.mean(bh)

    def record_dt(self):
        for rid, r in self.route_list.items():
            for b in range(len(r.bus_list)):
                bus_id = r.bus_list[b]
                self.agents[rid].record_dt(self.GM.memory[bus_id], bus_id=bus_id)

    def learn(self,episode=0):
        ploss_set = []
        qloss_set = []
        metaloss_set = []
        if self.share_scale == 0:
            for bus_id, bus in self.bus_list.items():
                if (len(self.GM.memory[bus_id]) + 1) > 16:
                    ploss, qloss,meta_loss = self.agents[bus.id].learn(self.GM.memory[bus_id])
                    ploss_set.append(ploss)
                    qloss_set.append(qloss)
                    if meta_loss!=None:
                        metaloss_set.append(meta_loss)

        if self.share_scale == 1:
            for rid,r in self.route_list.items():
                b = np.random.randint(0,len(r.bus_list))
                bus_id = r.bus_list[b]
                while len(self.GM.memory[bus_id])<=0 :
                    b = np.random.randint(0, len(r.bus_list))
                    bus_id = r.bus_list[b]

                ploss, qloss,meta_loss = self.agents[rid].learn(self.GM.memory[bus_id],bus_id=bus_id)

                ploss_set.append(ploss)
                qloss_set.append(qloss)
                if meta_loss != None:
                    metaloss_set.append(meta_loss)


        if len(ploss_set) > 0 and len(self.reward_signal) > 0:
            if len(metaloss_set)>0:
                return np.mean(ploss_set), np.mean(qloss_set),np.mean(metaloss_set),True
            else:
                return np.mean(ploss_set), np.mean(qloss_set), None, True
        else:
            return _, _, _,False



