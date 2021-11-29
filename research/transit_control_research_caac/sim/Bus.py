import xml.etree.ElementTree as ET
import numpy as np

class Bus():
    def __init__(self,id,route_id,stop_list,dispatch_time,block_id,dir, round=1 ):
        '''
        :param id: bus unique id
        :param route_id: route id bus serving
        :param stop_dist: stop location list bus travelling
        :param stop_list: stop list bus travelling
        :param dispatch_time: the time when bus begin its trip
        :param schedule: planned departure time at each stop
        :param pass_stop: the stops bus has passed
        :param left_stop: the stops left to go
        :param time_step: record the simulation time step when this bus is alive
        :param trajectory: record the travel location
        :param onboard_list: passenger id list onboard
        :param forward_bus: bus id of forward bus
        :param backward_bus: bus id of backward bus
        :param current_stop_duration: how long the bus has been stopping at current stop
        :param speed: bus cruising speed
        :param arr: arr= 0: on road ; arr=1: on arr
        :param board_period:  how long for a passenger board
        :param alight_period: how long for a passenger alight
        :param capacity: the maximum number of passengers allowing in this bus
        :param his: (dictionary) record the bus activity for RL {simulation time step: [state,action]}
        :param is_dispatch: is_dispatch=0: bus has not dispatched
        '''

        self.id = id
        self.route_id = route_id
        self.schedule = {}
        self.dispatch_time = dispatch_time
        self.trajectory = [0]
        self.stop_dist = {}

        self.stop_list = stop_list
        self.pass_stop = []
        self.left_stop = stop_list
        self.time_step=[dispatch_time]
        self.onboard_list = []
        self.alight_period = 1.8
        self.board_period = 3

        self.arr = 0
        self.capacity = 120

        self.speed = 11 # km/s
        self.c_speed = 11
        self.current_stop_duration = 0
        self.current_speed = 0.

        self.forward_bus = None
        self.backward_bus = None
        self.b = 0
        self.his={}
        self.last_vist_interval = -1
        self.stops_record = [-1]


    def set(self):
        self.serve_remain = 0
        self.hold_remain = 0
        self.currentstop =None
        self.is_hold = 0
        self.is_finish = 0
        self.is_dispatch = 0
        self.cost = {}
        self.hold_cost = {}
        self.stay ={}
        self.pax_boarded = {}
        self.trip_info = []

        for stop in self.stop_list:
            self.cost[stop] =  0
            self.stay[stop] = 0
            self.hold_cost[stop] =  0
            self.pax_boarded[stop] = 0

        self.loc = [0]
        self.occp = [0]
        self.dwell=0

        self.hold_info = {}

        self.is_virtual = 0

    def move(self,d ):
        self.occp.append(len(self.onboard_list)/self.capacity)
        self.loc.append(d+self.loc[-1])
        self.time_step.append(self.time_step[-1]+1)
        self.stops_record.append(-1)
    def stop(self,):
        self.occp.append(len(self.onboard_list) / self.capacity)
        self.loc.append( self.loc[-1])
        self.time_step.append(self.time_step[-1]+1)
        self.stops_record.append(self.pass_stop[-1])
    def dep(self, d=0):
        try:
            self.move(d )
            self.arr = 0
            self.is_hold=0

        except:
            print(self.route_id)

    #  return a list of passengers alight
    def pax_alight(self,alight_rate=0.3):
        pax = []

        for i in range(len(self.onboard_list)):
            if np.random.binomial(1, alight_rate)>0:
                pax.append(self.onboard_list[i])

        return pax

    def pax_alight_fix(self,stop,pax_list):
        pax = []

        for i in range(len(self.onboard_list)):

            if pax_list[self.onboard_list[i]].dest==stop.id:
                pax.append(self.onboard_list[i])
        return pax



