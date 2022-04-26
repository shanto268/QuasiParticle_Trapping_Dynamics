#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:04:41 2020

A QP trapping simulator that uses a simplified model:
    

@author: sas
"""
# Imports
from numpy import sin, cos, pi, exp, sqrt, zeros, ones, arange, mean, inf, cumsum, array, arctanh
from numpy.random import random, choice, uniform
from math import factorial
from time import perf_counter
from fitTools.Resonator import Resonator
from scipy.signal import butter, lfilter
from scipy.signal import windows, convolve
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dill
import sys
import random as rd


def create_high_low_state_switch_process(N, sampleUnit=5, fraction=0.1):
    samples = np.ones(N)
    freq = int(N / sampleUnit)
    fraction = 0.2

    samples = samples.reshape(sampleUnit, freq)

    for sample in samples:
        bound = int(fraction * len(sample))
        sample[:bound] = 2

    samples = samples.flatten()
    return samples


def create_high_low_state_switch_stochastic_process(N,
                                                    sampleUnit=5,
                                                    fraction=0.1):
    samples = np.ones(N)
    freq = int(N / sampleUnit)

    samples = samples.reshape(sampleUnit, freq)

    for sample in samples:
        bound = int(fraction * len(sample)) + int(
            rd.uniform(0,
                       rd.uniform(-fraction, fraction) * len(sample)))
        sample[:bound] = 2

    samples = samples.flatten()
    return samples


matplotlib.use('agg')


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
                 T=0.025,
                 model="default"):

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
        if model == "default":
            print("simulation with bulk material")
            self._getSwitchingEvents()
        elif model == "con":
            print("simulation with constant pT")
            self._getSwitchingEvents_SAS()
        elif model == "p":
            print("simulation with periodic pT")
            self._getSwitchingEvents_periodic_pT()
        elif model == "ps":
            print("simulation with periodic-stochastic pT")
            self._getSwitchingEvents_periodic_stochastic_pT()
        elif model == "corcon":
            print("simulation with correlated-constant pT")
            self._getSwitchingEvents_periodic_correlated_constant_pT()
        elif model == "corp":
            print("simulation with correlated-periodic pT")
            self._getSwitchingEvents_periodic_correlated_periodic_pT()
        elif model == "corps":
            print("simulation with correlated-periodic_stochastic pT")
            self._getSwitchingEvents_periodic_correlated_periodic_stochastic_pT(
            )
        else:
            print("model parameter missing")
            pass

    def _getSwitchingEvents_periodic_correlated_periodic_pT(self, ):
        """
            Constants Defining
            """
        factor_increase = 2.0
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
        self.nGen = []
        self.nRel = []
        self.releasePop = []
        self.nGenerated = []

        pT_coefficient = create_high_low_state_switch_process(N,
                                                              sampleUnit=10,
                                                              fraction=0.1)
        self.trapped = False

        for n in range(self.N):
            """
                Produce common QP generation events
                """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nGen.append(1)
                self.nGen.append(1)
            """        
                Produce trapping events
                """
            if not self.trapped:
                trapmask = random() < len(
                    self.nGen) * pT * pT_coefficient[n]  #periodic_pT
            elif self.trapped:
                trapmask = random() < len(
                    self.nGen) * pT * factor_increase  #factor * constant_pT

            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nGen)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
                self.trapped = True
            if k == 0:
                self.trapped = False
            """                    
                Produce release events
                """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nRel.append(1)

    #           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.releasePop.append(len(self.nRel))
            self.nGenerated.append(len(self.nGen))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])

    def _getSwitchingEvents_periodic_correlated_periodic_stochastic_pT(self, ):
        """
            Constants Defining
            """
        factor_increase = 2.0
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
        self.nGen = []
        self.nRel = []
        self.releasePop = []
        self.nGenerated = []

        pT_coefficient = create_high_low_state_switch_stochastic_process(
            N, sampleUnit=10, fraction=0.1)
        self.trapped = False

        for n in range(self.N):
            """
                Produce common QP generation events
                """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nGen.append(1)
                self.nGen.append(1)
            """        
                Produce trapping events
                """
            if not self.trapped:
                trapmask = random() < len(
                    self.nGen) * pT * pT_coefficient[n]  #periodic_pT
            elif self.trapped:
                trapmask = random() < len(
                    self.nGen) * pT * factor_increase  #factor * constant_pT

            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nGen)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
                self.trapped = True
            if k == 0:
                self.trapped = False
            """                    
                Produce release events
                """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nRel.append(1)

    #           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.releasePop.append(len(self.nRel))
            self.nGenerated.append(len(self.nGen))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])

    def _getSwitchingEvents_periodic_correlated_constant_pT(self, ):
        """
            Constants Defining
            """
        factor_increase = 2.0
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
        self.nGen = []
        self.nRel = []
        self.releasePop = []
        self.nGenerated = []

        # pT_coefficient = create_high_low_state_switch_stochastic_process(N,sampleUnit=10,fraction=0.1)
        self.trapped = False

        for n in range(self.N):
            """
                Produce common QP generation events
                """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nGen.append(1)
                self.nGen.append(1)
            """        
                Produce trapping events
                """
            if not self.trapped:
                trapmask = random() < len(self.nGen) * pT  #constant_pT
            elif self.trapped:
                # trapmask = random() < len(self.nGen)*pT*pT_coefficient[n], 1) #periodic_pT
                trapmask = random() < len(
                    self.nGen) * pT * factor_increase  #factor * constant_pT

            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nGen)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
                self.trapped = True
            if k == 0:
                self.trapped = False
            """                    
                Produce release events
                """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nRel.append(1)

    #           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.releasePop.append(len(self.nRel))
            self.nGenerated.append(len(self.nGen))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])

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

    def _getSwitchingEvents_periodic_stochastic_pT(self, ):
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
        self.nGen = []
        self.nRel = []
        self.releasePop = []
        self.nGenerated = []

        pT_coefficient = create_high_low_state_switch_stochastic_process(
            N, sampleUnit=10, fraction=0.1)

        for n in range(self.N):
            """
                Produce common QP generation events
                """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nGen.append(1)
                self.nGen.append(1)
            """        
                Produce trapping events
                """
            trapmask = random() < len(
                self.nGen) * pT * pT_coefficient[n]  #periodic_pT
            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nGen)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
            """                    
                Produce release events
                """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nRel.append(1)

    #           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.releasePop.append(len(self.nRel))
            self.nGenerated.append(len(self.nGen))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])

    def _getSwitchingEvents_periodic_pT(self, ):
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
        self.nGen = []
        self.nRel = []
        self.releasePop = []
        self.nGenerated = []

        pT_coefficient = create_high_low_state_switch_process(N,
                                                              sampleUnit=10,
                                                              fraction=0.1)

        for n in range(self.N):
            """
                Produce common QP generation events
                """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nGen.append(1)
                self.nGen.append(1)
            """        
                Produce trapping events
                """
            trapmask = random() < len(
                self.nGen) * pT * pT_coefficient[n]  #periodic_pT
            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nGen)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
            """                    
                Produce release events
                """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nRel.append(1)

    #           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.releasePop.append(len(self.nRel))
            self.nGenerated.append(len(self.nGen))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])

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

    def _getSwitchingEvents_SAS(self, ):
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
        self.nGen = []
        self.nRel = []
        self.releasePop = []
        self.nGenerated = []

        for n in range(self.N):
            """
            Produce common QP generation events
            """
            commask = random() < pCommon
            k = cs[commask][-1] if commask.any() else 0
            for _ in range(k):
                self.nGen.append(1)
                self.nGen.append(1)
            """        
            Produce trapping events
            """
            trapmask = random() < len(self.nGen) * pT  #this needs updating!!!
            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(self.nGen)))
            for _ in range(k):
                tau = self._MC_doro(self.Ne, self.Delta, self.de, self.T)
                E = self._Ea(tau, self.Delta, self.de)
                trappedChannels.append({'t': tau, 'E': E})
            """                    
            Produce release events
            """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                self.nRel.append(1)


#           Track changes
            self.nTrapped.append(len(trappedChannels))
            self.releasePop.append(len(self.nRel))
            self.nGenerated.append(len(self.nGen))

            #           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
            lFreqFactors.append([
                alpha * c['t'] *
                (self.cosd / sqrt(1 - c['t'] * self.sin2) +
                 c['t'] * self.sind2 / (4 * sqrt(1 - c['t'] * self.sin2)**3))
                for c in trappedChannels
            ])
            self.freqFactors[n] += sum(lFreqFactors[n])


class NBResonator():
    def __init__(self,
                 trapper,
                 L=1e-9,
                 C=0.7e-12,
                 photonRO=1,
                 photonNoise=0.5,
                 Qi=5e4,
                 Qe=5e4,
                 sampleRate=300e6,
                 delKappa=-0.5,
                 fd=None):

        self.port1 = Resonator('R')  #Resonator Object Instantiation
        """
        info from QPTrapper Sim
        """
        self.N = trapper.N
        self.Lj0 = trapper.Lj0
        self.Lj = trapper.Lj
        """
        values calculated from QPTrapper Sim
        """
        self.f0 = 1 / (2 * np.pi * np.sqrt((L + self.Lj) * C))
        self.q0 = self.Lj / (L + self.Lj)
        """
        input parameters
        """
        self.photonRO = photonRO
        self.Qi = Qi
        self.Qe = Qe
        self.Qt = Qi * Qe / (Qi + Qe)
        self.sampleRate = sampleRate
        """
        some physical parameters
        """
        self.kappa = 2 * np.pi * self.f0 / self.Qt
        self.kappa_e = 2 * np.pi * self.f0 / self.Qe
        self.fwhm = self.f0 / self.Qt
        self.diameter = 2 * self.Qt / self.Qe
        self.f = self.f0 + delKappa * self.kappa / (
            2 * np.pi)  # the resonator drive frequency for measurement

        if fd != None:
            self.f = fd
        self.f_form = self.f0 - (self.q0 * self.f0 * self.Lj *
                                 np.array(trapper.freqFactors) / 2)
        self.f_shift = self.q0 * self.f0 * self.Lj * trapper.L1 / 2
        #        self.SNR = photonRO*self.kappa/(4*sampleRate)*(1 - 4*self.Qt/self.Qe * (1-self.Qt/self.Qe))
        """
        SNR Calculation
        """
        self.pSNR = photonRO * self.kappa**2 / (4 * self.kappa_e *
                                                (0.5 + photonNoise) *
                                                sampleRate)
        self.pSNRdB = 10 * np.log10(self.pSNR)

        #        self.sigma = self.diameter/(2*np.sqrt(2*self.SNR))
        self.sigma = 1 / np.sqrt(self.pSNR)
        self.complex_noise = np.empty(self.N, dtype=complex)
        self.complex_noise.real = np.random.normal(scale=self.sigma,
                                                   size=self.N)
        self.complex_noise.imag = np.random.normal(scale=self.sigma,
                                                   size=self.N)
        #        self.complex_noise = np.array(
        #                [np.random.normal() * np.exp(1j*np.random.uniform(low=0,high=2*np.pi)) for i in range(self.N)
        #                ])
        #get the response at given frequency
        #        self._get_clean_response(self.w)
        """
        passing resonator paramaters to Resonator object to generate signal
        """
        kwargs = dict(fr=self.f_form,
                      Ql=self.Qt,
                      Qc=self.Qe,
                      a=1.,
                      alpha=0.,
                      delay=0.)
        signal = self.port1._S11_directrefl(self.f, **kwargs)
        self.signal = self.butter_lowpass_filter(
            signal - signal[0], self.kappa, self.sampleRate) + signal[0]
        """
        adding noise to signal
        """
        self.signal += self.complex_noise
        #        self.signal = self.port1._S11_directrefl(self.f,**kwargs)
        """
        dictionary of all parameters from NBResonator Analysis
        """
        self.dParams = {
            'fd': self.f,
            'f0': self.f0,
            'Qt': self.Qt,
            'Qi': self.Qi,
            'Qe': self.Qe,
            'N': self.N,
            'q': self.q0,
            'photonRO': self.photonRO,
            'sampleRate': self.sampleRate,
            'kappa': self.kappa,
            'fwhm': self.fwhm,
            'diameter': self.diameter,
            'freq_shift': self.f_shift,
            'SNR': self.pSNR,
            'SNRdB': self.pSNRdB,
            'sigma': self.sigma
        }

    def butter_lowpass(self, cutoff, fs, order=1):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=1):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


def makePlot(N, sampleRate, test, sim_type, sim_time):
    time = arange(N) / sampleRate
    h, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(time * 1e3, test.nTrapped, '-g')
    plt.title('actual number of trapped QPs')
    ax[0].set_ylabel('n trapped')
    ax[1].plot(time * 1e3, test.nGenerated, '-b')
    ax[1].set_ylabel('n generated')
    ax[1].set_xlabel('Time [ms]')
    ax[2].plot(time * 1e3, test.releasePop, '-b')
    ax[2].set_ylabel('n released')
    ax[2].set_xlabel('Time [ms]')
    plt.tight_layout()
    plt.savefig("qp_time_{}_{}.png".format(sim_type, sim_time))
    plt.close()


"""
MAIN CODE
"""
if __name__ == '__main__':
    """
    input paramaters
    """
    sim_type = str(sys.argv[1])
    model = str(sys.argv[2])  #options: default,p,ps,con,cor
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
    L = 1.89e-9
    C = 0.2776e-12
    Qi = 6000
    Qe = 500
    photonRO = 2
    delKappa = -0.1

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
        'T': 0.025,
        'model': model
    }

    now = perf_counter()
    test = QPtrapper(**args)

    from datetime import datetime

    sim_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    """
    pickle the QPtrapper object
    """
    try:
        dill.dump(test,
                  file=open(
                      "qp_trapper_{}_{}.pickle".format(sim_type, sim_time),
                      "wb"))
    except:
        pass

    timer = perf_counter() - now
    print('qptrapper runtime: {} seconds'.format(timer))
    print('The average number of trapped QPs is {:.4}'.format(
        np.mean(test.nTrapped)))
    """
    save the qp trapping plot
    """
    try:
        makePlot(N, sampleRate, test, sim_type, sim_time)
    except:
        pass

    now2 = perf_counter()
    """
    calling NBResonator Object
    """
    resArgs = {
        'L': L,
        'C': C,
        'photonRO': photonRO,
        'Qi': Qi,
        'Qe': Qe,
        'sampleRate': sampleRate,
        'delKappa': delKappa
    }
    res = NBResonator(test, **resArgs)
    """
    pickle the NBResonator object
    """
    try:
        dill.dump(res,
                  file=open(
                      "nb_resonator_{}_{}.pickle".format(sim_type, sim_time),
                      "wb"))
    except:
        pass
    duration2 = perf_counter() - now2
    print('Resonator runtime: {}'.format(duration2))

    avgTime = 4 * res.Qt * 50 / (photonRO * 2 * np.pi * res.f0)
    nAvg = int(max(avgTime * sampleRate, 1))

    window = windows.hann(nAvg)
    rhann = convolve(res.signal.real, window, mode='same') / sum(window)
    ihann = convolve(res.signal.imag, window, mode='same') / sum(window)
    """
    save the plot
    """
    plt.hist2d(rhann, ihann, bins=(50, 50))
    plt.savefig("resonator_plot_{}_{}.png".format(sim_type, sim_time))
    plt.close()
