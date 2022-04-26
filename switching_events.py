"""
## Steps:

1) define the various constants
2) create data structure and Poisson Distribution processes
3) loop over all samples
    i) generate "common" QP events
    ii) generate "rare" QP events
    iii) generate "pair recombination" events
    iv) generate "trapping events"
    v) generate "untrapping" events
4) compute number of QPs in trapped state and other states (annhilated,untrapped)
5) compute and store the effect on "frequency shift terms for each time point"

"""


def simulateTrappingDynamics():
    nGenerated = []
    nBulk = []
    nTrapped = []
    nReleased = []
    nAnnhilated = []

    for t in time_range:
        """
        first event
        """
        if t == 1:
            nBulk.append(0)
            nTrapped.append(0)
        
        """
        generate QPs
        """
        qp_generated = generate_qp(generation_process())
        
        """
        add generated QP stay to bulk
        """
        qp_bulk = nBulk[t - 1] + qp_generated
        
        """
        annhilate QPs from bulk from previous time step
        """
        qp_recombined = recombine_qp(annhilation_process(), nBulk[t - 1])
        
        """
        remove qp from bulk that recombined
        """
        qp_bulk = qp_bulk - qp_recombined
        
        """
        Trap QPs
        """
        qp_trapped = trap_qp(trapping_process(), qp_bulk) + nTrapped[t - 1]
        
        """
        Release QPs
        """
        qp_untrapped = release_qp(release_process(), nTrapped[t - 1])
        
        """
        Update the number of trapped QP
        """
        qp_trapped = qp_trapped - qp_untrapped
        
        """
        Update QP number in bulk
        """
        qp_bulk = qp_bulk - qp_trapped
        
        """
        store values to disc
        """
        nGenerated.append(qp_generated)
        nAnnhilated.append(qp_recombined)
        nTrapped.append(qp_trapped)
        nReleased.append(qp_untrapped)
        nBulk.append(qp_bulk)
        
        """
        Calculate Shift In Frequency
        """
        measure_frequency_shift()
def getSwitchingEvents():
    """
    Constants Defining
    """
    phi0 = phi0
    rphi0 = phi0 / (2 * pi)
    alpha = Delta / (2 * (rphi0**2))
    """
    Creating Poisson Distribution for the various process
    """
    cs = array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)  #channels?
    factorials = array([factorial(k) for k in cs],
                       dtype=int)  #factorials of cs values
    pR = ((dt / tauR)**cs) * exp(
        -dt / tauR) / factorials  #poisson distribution of release
    pT = ((dt / tau)**cs) * exp(
        -dt / tau) / factorials  #poisson distribution of trap
    pCommon = ((dt / tauCommon)**cs) * exp(
        -dt / tauCommon) / factorials  #poisson distribution of common
    pRare = ((dt / tauRare)**cs) * exp(
        -dt / tauRare) / factorials  #poisson distribution of cosmic
    pRecomb = ((dt / tauRecomb)**cs) * exp(
        -dt / tauRecomb) / factorials  #poisson distribution of recombine
    """
    Data Structures for storing info from simulation
    """
    trappedChannels = []  #?
    nTrapped = []  #number of trapped QP
    lFreqFactors = []  #?
    # alpha = Delta/(4*(rphi0**2))
    freqFactors = zeros(N)  #?
    nBulk = list(ones(3))  #3D vector for some reason?
    bulkPop = []  #?
    burstIndices = []  #?

    for n in range(N):
        """
        Produce common QP generation events
        """
        commask = random() < pCommon
        k = cs[commask][-1] if commask.any() else 0
        for _ in range(k):
            nBulk.append(1)
            nBulk.append(1)
            """
            Produce rare QP generation events -- such as cosmic ray bursts
            """
            raremask = random() < pRare
            k = cs[raremask][-1] if raremask.any() else 0
            for _ in range(k):
                burstIndices.append(n)
                burst = int(random() * 50)
                for i in range(burst):
                    nBulk.append(1)
                    nBulk.append(1)
            """
            Produce pair recombination events
            """
            recombmask = random() < len(nBulk) * pRecomb
            k = cs[recombmask][-1] if recombmask.any() else 0
            k = min((k, len(nBulk) // 2))
            for _ in range(k):
                nBulk.remove(1)
                nBulk.remove(1)
            """        
            Produce trapping events
            """
            trapmask = random() < len(nBulk) * pT
            k = cs[trapmask][-1] if trapmask.any() else 0
            k = min((k, len(nBulk)))
            for _ in range(k):
                tau = _MC_doro(Ne, Delta, de, T)
                E = _Ea(tau, Delta, de)
                trappedChannels.append({'t': tau, 'E': E})
                if nBulk:
                    nBulk.remove(1)
            """                    
            Produce release events
            """
            relmask = random() < len(trappedChannels) * pR
            k = cs[relmask][-1] if relmask.any() else 0
            k = min((k, len(trappedChannels)))
            for _ in range(k):
                ch = choice(trappedChannels)
                trappedChannels.remove(ch)
                nBulk.append(1)


#           Track changes
nTrapped.append(len(trappedChannels))
bulkPop.append(len(nBulk))

#           Calculate frequency shift terms for each time point -- Sum_i 1/L_i
lFreqFactors.append([
    alpha * c['t'] * (cosd / sqrt(1 - c['t'] * sin2) + c['t'] * sind2 /
                      (4 * sqrt(1 - c['t'] * sin2)**3))
    for c in trappedChannels
])
freqFactors[n] += sum(lFreqFactors[n])
