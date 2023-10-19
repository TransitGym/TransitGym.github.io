import numpy as np
from collections import deque
import random


class Memory():
    def __init__(self, members, capacity=2000):
        self.memory = {}
        self.capcacity = capacity
        # temp memory to store last-step state and action because of no immediate feedback
        self.temp_memory = {}
        self.experience = {}
        for m in members:
            self.memory[m] = deque(maxlen=capacity)
            self.experience[m] = 0
            self.temp_memory[m] = {'s': [], 'a': [], 'fp': [], 'r1': [], 'r2': [], 'r3': [], 'stop_embed': []}

    def update(self):
        for m in list(self.memory.keys()):
            l = len(self.memory[m])
            for _ in range(int(l * 0.8)):
                self.memory[m].popleft()

                self.experience[m] -= 1
            self.temp_memory[m] = {'s': [], 'a': [], 'fp': [], 'r1': [], 'r2': [], 'r3': [], 'stop_embed': []}

    def remember(self, state, fp, action, reward, next_state, next_fp, stop_embed, next_stop_embed, member_id):
        self.experience[member_id] += 1  # min(self.experience[member_id]+1, self.capcacity)
        self.memory[member_id].append((state, fp, action, reward, next_state, next_fp, stop_embed, next_stop_embed))
