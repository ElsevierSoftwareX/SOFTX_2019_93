#!/usr/bin/env python
# encoding: utf-8 
"""
tslices.py

Created by Jörg Frauendiener on 2010-11-17, modifications by Ben Whale since
then.
Additional development by Ben Whale.
Copyright (c) 2010 University of Otago. All rights reserved.
"""

class timeslice:
   
    def __init__(self, data, domain, time):
        self.data = data
        self.domain = domain
        self.time = time

    @property
    def numFields(self):
        return len(self.data)
    
    def __repr__(self):
        s = "timeslice(fields = %s, grid = %s, time = %s)"%(
            repr(self.data), 
            repr(self.grid), 
            repr(self.time)
            )
        return s

    @property
    def dx(self):
        return self.step_sizes
      
    @property
    def step_sizes(self):
        return self.domain.step_sizes
        
    @property
    def fields(self):
        return self.data

    @property        
    def x(self):
        return self.domain

    def communicate(self):
        self.domain.sendrecv(self.data)
