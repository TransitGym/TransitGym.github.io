import xml.etree.ElementTree as ET
import numpy as np
from random import seed
from random import gauss,randint

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

        self.id = id
        self.lat = lat
        self.lon = lon
        self.loc = 0.
        self.next_stop = None
        self.routes = []
        self.waiting_list=[]
        self.dyna_arr_rate = []
        self.dyna_arr_rate_sp ={}
        self.arr_bus_load =[]
        self.arr_log = {}
        self.uni_arr_log = []
        self.dep_log = {}
        self.uni_dep_log = []
        self.pax = {}
        self.dest = {}
        self.served = 0

    # return a list of passengers arrival time
    def pax_gen(self,bus,sim_step=0):

        pax = []
        base=0
        interval = 0
        self.arr_bus_load.append(len(bus.onboard_list))
        if bus!=None:
            if len(self.arr_log[bus.route_id])>1:
                interval = (self.arr_log[bus.route_id][-1] -self.arr_log[bus.route_id][-2] )
                sample = (np.random.poisson(self.rate*self.dyna_arr_rate[int(sim_step/3600)%24]+0.0001,int(interval)  ))
                base=self.arr_log[bus.route_id][-2]
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base+i for t in range(sample[i])]
            else:
                # assume passenger will gather in less than 15min before the first bus began.
                sample = (np.random.poisson(self.rate*self.dyna_arr_rate[int(sim_step/3600)%24]+0.0001,900 ))
                base = self.arr_log[bus.route_id][-1]
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base-i for t in range(sample[i])]
        else:
            for k,v in self.arr_log.items():
                interval = sim_step - v[-1]
                sample = (np.random.poisson(self.rate*self.dyna_arr_rate[int(sim_step/3600)%24],int(interval) ))
                base = v[-1][1]
                for i in range(sample.shape[0]):
                    if sample[i] > 0:
                        pax += [base + i for t in range(sample[i])]

        return  pax

    def set_rate(self,r ):
        self.rate = r # pax/sec


    def pax_gen_od(self,bus,sim_step=0,dest_id=None):


        base=0
        interval = 0

        if bus!=None:
            if len(self.arr_log[bus.route_id])>1:
                interval = (self.arr_log[bus.route_id][-1] -self.arr_log[bus.route_id][-2] )
                sample = (np.random.poisson(self.rate*(self.dest[dest_id][int(sim_step/3600)%24])+0.0001,int(interval)  ))
                base=self.arr_log[bus.route_id][-2]
                pax = [ ]
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base+i for t in range(sample[i])]
            else:
                # assume passenger will gather in less than 15min before the first bus began.
                sample = (np.random.poisson(self.rate*(self.dest[dest_id][int(sim_step/3600)%24])+0.0001,900 ))

                base = self.arr_log[bus.route_id][-1]
                pax = [ ]
                for i in range(sample.shape[0]):
                    if sample[i]>0:
                        pax+=[base-i for t in range(sample[i])]
        else:
            for k,v in self.arr_log.items():
                interval = sim_step - v[-1]
                sample = (np.random.poisson(self.rate*self.dest[dest_id][int(sim_step/3600)%24],int(interval) ))

                base = v[-1][1]
                for i in range(sample.shape[0]):
                    if sample[i] > 0:
                        pax += [base + i for t in range(sample[i])]

        return  pax


