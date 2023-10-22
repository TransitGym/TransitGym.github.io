import xml.etree.ElementTree as ET
import numpy as np

class Route():
    def __init__(self,id,stop_list, dist_list):
        '''

        :param id:  route id
        :param stop_list: stop id list along the route
        :param bus_list: bus id along the route
        :param headway: planned headway in this route
        :param dist_list: distance between two consecutive stops
        '''
        self.id = id
        self.stop_list = stop_list
        self.dist_list = dist_list
        self.schedule = []
        self.bus_list = []
        self.avg_wait = -1