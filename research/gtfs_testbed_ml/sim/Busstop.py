import xml.etree.ElementTree as ET
import numpy as np
from random import seed
from random import gauss,randint
from collections import deque, OrderedDict
from sim.Passenger import Passenger
class Bus_stop():
    def __init__(self,id,lat,lon  ):
        '''

        :param id: bus stop unique id
        :param lat:  bus stop latitude in real-world
        :param lon:  bus stop longitude in real-world
        :param routes: bus stop serving routes set
        :param waiting_list:  waitting passenger list in this stop
        :param dyna_arr_rate:  dynamic passenger arrival rate for this stop
        :param arr_bus_load:  record arrving bus load
        :param arr_log:  (dictionay) record bus arrival time with respect to each route (route id is key)
        :param uni_arr_log: (list) record bus arrival time
        :param dep_log: (dictionay) record bus departure time with respect to each route (route id is key)
        :param uni_dep_log: (list) record bus departure time

        '''
        self.pre_gen_pax_list = {}
        self.id = id
        self.lat = lat
        self.lon = lon
        self.loc = 0.
        self.loc_route = {}
        self.next_stop = None
        self.prev_stop = None
        self.next_stop_route = {}
        self.prev_stop_route = {}
        self.routes = []
        self.waiting_list=[]
        self.dyna_arr_rate_sp ={}
        self.dyna_arr_rate = []
        # self.type = 0
        # self.demand = 0
        self.arr_bus_load =[]
        self.arr_log = {}
        self.uni_arr_log = []
        self.dep_log = {}
        self.uni_dep_log = []
        self.pax = {}
        self.is_in_shared_corridor = 0
        self.bus_serving = deque()
        self.dest = OrderedDict()
        self.dest_route = {}
        self.cum_arr = 0
        self.cum_dep = 0
        self.arrivals ={}
        self.actual_departures = {}
        self.actual_arrivals ={}
        self.rate = 1.0  # pax/sec
        self.bus_queue = []
        self.queue_detail = {}
        self.waiting = []

    def set_rate(self,r ):
        self.rate =r # pax/sec

    def pax_read(self,bus,sim_step=0):
        pax = []
        base = 0
        interval = 0

        self.arr_bus_load.append(len(bus.onboard_list))
        if len(self.arr_log[bus.route_id]) > 1:
            interval = (self.arr_log[bus.route_id][-1]  - self.arr_log[bus.route_id][-2] )
            base = self.arr_log[bus.route_id][-2]
        else:
            base = 0

        leave=[]
        for p_id,p in self.pax.items():
            if p.plan_board_time>base and p.plan_board_time<=sim_step and p.dest in bus.left_stop:
                p.took_bus = bus.id
                leave.append(p_id)
                if base>0:
                    p.arr_time = base+np.random.randint(0,sim_step-base)
                else:
                    p.arr_time = p.plan_board_time
                pax.append(p)

        return pax


    def  pre_pax_gen(self, route,begin=6 * 3600, demand_impulse_rate=0.,share_stops=[],shares=0.):
        flag = 0
        arr_cum = 0
        # train/test under uncertainty
        scales = np.clip(np.random.normal(1.5, demand_impulse_rate), a_min=1.5, a_max=5.)
        # scales = np.clip(np.random.normal(1.5, demand_impulse_rate), a_min=1.5, a_max=5.)
        interval = 60
        scales = 2.
        while int(begin / 3600) < 23:
            temp_pre_gen_pax_list = []
            times = []
            a = 0
            for dest_id in self.dest_route[route].keys():
                mu = self.dest_route[route][dest_id][int(begin / 3600) % 24]
                arr_rate = mu*scales
                a+=arr_rate
                if dest_id in share_stops and self.id in share_stops:
                    arr_rate = 10.*arr_rate

                pax = []
                if flag == 0:
                    sample = np.random.poisson((arr_rate+0.00001)*interval, int(900/interval))
                    for i in range(sample.shape[0]):
                        if sample[i] > 0:
                            pax += [begin + 3600 - i*interval for t in range(sample[i])]
                else:
                    sample =  np.random.poisson((arr_rate+0.00001)*interval, int(3600/interval))

                    for i in range(sample.shape[0]):
                        if sample[i] > 0:
                            pax += [begin + i *interval for t in range(sample[i])]

                current_num = len(self.pre_gen_pax_list)
                for i in range(len(pax)):
                    p = Passenger(origin=self.id, arr_time=pax[i], id=current_num + i)
                    p.dest = dest_id
                    p.route = route
                    # add route choice
                    if dest_id in share_stops and self.id in share_stops:
                        if np.random.randn()<=shares:
                            p.share=1

                    self.pre_gen_pax_list[current_num + i]=p

            print(a)
            begin += 3600
            flag = 1

    def get_pax(self, bus, sim_step=0):

        pax = []
        if bus != None:
            for k in range(len(self.pre_gen_pax_list)):
                if self.pre_gen_pax_list[k].arr_time <= sim_step and self.pre_gen_pax_list[k].born == 0 and self.pre_gen_pax_list[k].dest in bus.left_stop \
                        and (self.pre_gen_pax_list[k].route==bus.route_id or self.pre_gen_pax_list[k].share==1):
                    self.pre_gen_pax_list[k].born = 1
                    pax.append(self.pre_gen_pax_list[k])
        return pax

    def debug_waiting(self, sim_step):
        waiting = 0
        for k in range(len(self.pre_gen_pax_list)):
            if self.pre_gen_pax_list[k].arr_time <= sim_step and self.pre_gen_pax_list[k].onboard_time == 999999999:
                waiting += 1
        self.waiting.append(waiting)

if __name__ == '__main__':
    seed(1)
    # generate some integers
    r = 1. / randint(160, 240)
    for _ in range(10):
        for _ in range(10):
            value =  max(gauss(r,0.0004),0.00002)
            print(value*3600)
        print('--')