import xml.etree.ElementTree as ET
import numpy as np
from random import seed
from random import gauss, randint

from sim.Passenger import Passenger
class Bus_stop():
    def __init__(self, id, lat, lon):
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

        self.id = id
        self.lat = lat
        self.lon = lon
        self.loc = 0.
        self.next_stop = None
        self.routes = []
        self.waiting_list = []
        self.dyna_arr_rate = []
        self.dyna_arr_rate_sp = {}
        # self.type = 0
        # self.demand = 0
        self.arr_bus_load = []
        self.arr_log = {}
        self.uni_arr_log = []
        self.dep_log = {}
        self.uni_dep_log = []
        self.pax = {}
        from collections import OrderedDict
        self.dest = OrderedDict()

        self.pre_gen_pax_list = {}
        self.cum_arr = 0
        self.cum_dep = 0
        self.arrivals ={}
        self.actual_departures = {}
        self.actual_arrivals ={}
        self.rate = 1.0  # pax/sec

    def pre_pax_gen(self, begin=6 * 3600, demand_impulse_rate=0.):
        flag = 0
        # bus.abnormal = 0
        ## Test for vis
        # if ((len(bus.pass_stop) == 30 or len(bus.pass_stop) == 29 or len(bus.pass_stop) == 28 )and  sim_step-24149>3600*6 and sim_step-24149<3600*7) :
        #    r = 12.
        #   bus.abnormal = 1.
        arr_cum = 0
        # train/test under uncertainty
        scales = np.clip(np.random.normal(1., demand_impulse_rate), a_min=1., a_max=6.)

        # test anonamly
        # scales = np.clip(np.random.normal(demand_impulse_rate, 1.), a_min=1., a_max=6.)
        # scales = {}
        # for dest_id in self.dest.keys():
        #     scales[dest_id]= np.clip(np.random.normal(1.,demand_impulse_rate), a_min=1., a_max=6.)
        while int(begin / 3600) < 23:
            temp_pre_gen_pax_list = []
            times = []
            for dest_id in self.dest.keys():
                mu = self.dest[dest_id][int(begin / 3600) % 24]
                arr_rate = mu*scales
                # arr_rate = r * self.dest[dest_id][int(begin / 3600) % 24]
                pax = []
                if flag == 0:
                    sample = np.random.poisson(arr_rate+0.0001, 900)
                    for i in range(sample.shape[0]):
                        if sample[i] > 0:
                            pax += [begin + 3600 - i for t in range(sample[i])]
                else:
                    sample =  np.random.poisson(arr_rate+0.0001, 3600)

                    for i in range(sample.shape[0]):
                        if sample[i] > 0:
                            pax += [begin + i for t in range(sample[i])]
                # if len(pax)<=0:
                #     pax = [begin]
                times += pax
                for t in pax:
                    p = Passenger(origin=self.id, arr_time=t, id=-1)
                    p.dest = dest_id
                    temp_pre_gen_pax_list.append(p)

            times = sorted(times)
            current_num = len(self.pre_gen_pax_list)

            for i in range(len(temp_pre_gen_pax_list)):
                temp_pre_gen_pax_list[i].arr_time = times[i]
                temp_pre_gen_pax_list[i].id = current_num + i
                self.pre_gen_pax_list[current_num + i] = temp_pre_gen_pax_list[i]
                # arr_cum+=1
                # self.arrivals[times[i]] = arr_cum
            begin += 3600
            flag = 1

    def get_pax(self, bus, sim_step=0):

        pax = []
        if bus != None:
            for k in range(len(self.pre_gen_pax_list)):
                p = self.pre_gen_pax_list[k]
                if p.arr_time <= sim_step and p.born == 0:
                    self.pre_gen_pax_list[k].born = 1
                    pax.append(self.pre_gen_pax_list[k])
                if p.arr_time > sim_step:
                    break

        return pax


if __name__ == '__main__':
    np.random.seed(1)
    # generate some integers
    r = 1. / randint(160, 240)
    for _ in range(10):
        for _ in range(10):
            value = np.clip(np.random.normal(1., 1), a_min=1., a_max=6.)
            print(value * 3600)
        print('--')