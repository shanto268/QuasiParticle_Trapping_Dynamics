#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:04:41 2020

A QP trapping simulator that uses a simplified model:

@author: sas
"""
# Imports
#import numpy as np
from numpy import sin, cos, pi, exp, sqrt, zeros, ones, arange, mean, inf, cumsum, array, arctanh
from numpy.random import random, choice, uniform
from math import factorial
from time import perf_counter
import matplotlib.pyplot as plt


class QPtrapper:
    def __init__(self,
                 N=1000,
                 Lj=3.6e-11,
                 tauTrap=1e-5,
                 tauRelease=1e-5,
                 tauCommon=1e-4,
                 tauRare=1e-2,
                 tauRecomb=1e-4,
                 sampleRate=300e6,
                 phi=0.4,
                 Delta=2.72370016e-23,
                 T=0.025):

        self.N = int(N)  #number of samples (i.e. duration*sample_rate)
        """
        poisson paramaters
        """
        # self.Lj = Lj/self.cosd
        self.tauCommon = tauCommon  #poisson param for?
        self.tauRare = tauRare  #poisson param for cosmic ray events
        self.tauRecomb = tauRecomb  #poisson param for qp recombining
        self.tau = tauTrap  #poisson param for qp traps
        self.tauR = tauRelease  #poisson param for qp release
        """
        physical input parameters
        """
        self.sampleRate = sampleRate  #sample rate
        self.dt = 1 / sampleRate  #timesteps
        self.phi = phi  #flux bias
        self.Delta = Delta  #?
        self.T = T  #total time?
        self.de = pi * phi  #?
        #        if ((phi // 0.5) % 2):
        #            self.de += pi
        """
        trig calculations
        """
        self.cosd = cos(self.de)
        self.sind2 = sin(self.de)**2
        self.sin4 = sin(self.de / 2)**4
        self.sin2 = sin(self.de / 2)**2
        """
        fundamental constants
        """
        self.phi0 = 2.06783383 * 1e-15  #h/2e
        self.kb = 1.38064852e-23
        self.rphi0 = self.phi0 / (2 * pi)  #hbar/2e
        """
        physical parameters related to output?
        """
        self.Ne = 8 * (self.rphi0**2) / (self.Delta * Lj)  #?
        self.Lj0 = Lj  #inductance initial?
        self.Lj = Lj / (1 - sin(self.de / 2) * arctanh(sin(self.de / 2))
                        )  #inductance?
        # alpha = self.Delta/(4*(self.rphi0**2))
        alpha = self.Delta / (2 * (self.rphi0**2))  #?
        self.L1 = alpha * (self.cosd / sqrt(1 - self.sin2) + self.sind2 /
                           (4 * sqrt(1 - self.sin2)**3))  #?
        """
        simulation start command
        """
        self._getSwitchingEvents()

    def _dorokhov_boltz(self,
                        tau,
                        Ne=680,
                        delta=2.72370016e-23,
                        de=pi / 2,
                        T=0.025):
        return Ne / (tau * sqrt(1 - tau)) * exp(-self._Ea(tau, delta, de) /
                                                (self.kb * T))

    def _Ea(self, tau, delta=2.72370016e-23, de=pi / 2):
        return delta * sqrt(1 - tau * sin(de / 2)**2)

    def _MC_doro(self, Ne=680, delta=2.72370016e-23, de=pi / 2, T=0.025):
        scale = self._dorokhov_boltz(0.999999999, Ne, delta, de, T)
        while True:
            x = uniform(low=0.0, high=1)
            y = random() * scale
            try:
                if y < self._dorokhov_boltz(x, Ne, delta, de, T):
                    return x
                    break
            except (ZeroDivisionError, RuntimeWarning):
                pass

    def _Poisson(self, tau):
        return (self.dt / tau) * exp(-self.dt / tau)

    def _getSwitchingEvents(self, ):
        #        phaseFactor = sin(abs(self.de))
        #        pTrap = phaseFactor*self._Poisson(self.tau)
        #        pTrap = self._Poisson(self.tau)
        #        pRelease = self._Poisson(self.tauR)
        #        pCommon = self._Poisson(self.tauCommon)
        #        pRare = self._Poisson(self.tauRare)
        """
        Constants Defining
        """
        phi0 = self.phi0
        rphi0 = phi0 / (2 * pi)
        alpha = self.Delta / (2 * (rphi0**2))
        """
        Creating Poisson Distribution for the various process
        """
        cs = array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)  #channels?
        factorials = array([factorial(k) for k in cs],
                           dtype=int)  #factorials of cs values
        pR = ((self.dt / self.tauR)**cs) * exp(
            -self.dt /
            self.tauR) / factorials  #poisson distribution of release
        pT = ((self.dt / self.tau)**cs) * exp(
            -self.dt / self.tau) / factorials  #poisson distribution of trap
        pCommon = ((self.dt / self.tauCommon)**cs) * exp(
            -self.dt /
            self.tauCommon) / factorials  #poisson distribution of common
        pRare = ((self.dt / self.tauRare)**cs) * exp(
            -self.dt /
            self.tauRare) / factorials  #poisson distribution of cosmic
        pRecomb = ((self.dt / self.tauRecomb)**cs) * exp(
            -self.dt /
            self.tauRecomb) / factorials  #poisson distribution of recombine
        """
        Data Structures for storing info from simulation
        """
        trappedChannels = []  #?
        self.nTrapped = []  #number of trapped QP
        lFreqFactors = []  #?
        # alpha = self.Delta/(4*(rphi0**2))
        self.freqFactors = zeros(self.N)  #?
        self.nBulk = list(ones(3))  #3D vector for some reason?
        self.bulkPop = []  #?
        self.burstIndices = []  #?

        for n in range(self.N):
            """
            Produce common QP generation events
            """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nBulk.append(1)
                self.nBulk.append(1)
            """
            Produce rare QP generation events -- such as cosmic ray bursts
            """
            raremask = random() < pRare
            k = cs[raremask][-1] if raremask.any() else 0
            for _ in range(k):
                self.burstIndices.append(n)
                burst = int(random() * 50)
                for i in range(burst):
                    self.nBulk.append(1)
                    self.nBulk.append(1)
            """
            Produce pair recombination events
            """
            recombmask = random() < len(self.nBulk) * pRecomb
            k = cs[recombmask][-1] if recombmask.any() else 0
            k = min((k, len(self.nBulk) // 2))
            for _ in range(k):
                self.nBulk.remove(1)
                self.nBulk.remove(1)
            """        
            Produce trapping events
            """
            trapmask = random() < len(self.nBulk) * pT
            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nBulk)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
                if self.nBulk:
                    self.nBulk.remove(1)
            """                    
            Produce release events
            """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nBulk.append(1)


#           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.bulkPop.append(len(self.nBulk))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])

if __name__ == '__main__':
    """
    User Inputs for the simulation
    """
    duration = 1e-3  # seconds to record data
    sampleRate = 300e6
    N = int(duration * sampleRate)
    tauTrap = 140e-6
    tauRelease = 40e-6
    tauCommon = 4e-4
    tauRare = 1e-2
    tauRecomb = 2e-3
    phi = 0.45
    Lj = 21.2e-12
    args = {
        'N': N,
        'Lj': Lj,
        'tauTrap': tauTrap,
        'tauRelease': tauRelease,
        'tauCommon': tauCommon,
        'tauRare': tauRare,
        'tauRecomb': tauRecomb,
        'sampleRate': sampleRate,
        'phi': phi,
        'Delta': 2.72370016e-23,
        'T': 0.025
    }
    """
    Driver of the simulation
    """
    now = perf_counter()
    test = QPtrapper(**args)  #calling the simulator
    timer = perf_counter() - now
    print('qptrapper runtime: {} seconds'.format(timer))
    print('The average number of trapped QPs is {:.4}'.format(
        mean(test.nTrapped)))
    """
    Makes the plot
    """
    time = arange(N) / sampleRate

    h, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time * 1e3, test.nTrapped, '-g')
    plt.title('actual number of trapped QPs')
    ax[0].set_ylabel('n trapped')
    ax[1].plot(time * 1e3, test.bulkPop, '-b')
    ax[1].set_ylabel('number in bulk')
    ax[1].set_xlabel('Time [ms]')
