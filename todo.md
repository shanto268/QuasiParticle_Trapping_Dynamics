#Legacy todo

## To Do:

-[x] log-log plot of IQ
-[x] time-series plot of IQ
-[x] simulation time periods: few seconds @ **high SNR** (20 dB)
-[x] update correlation periods to **few micro to milliseconds**
-[x] update driver main code: args, pickle files
-[x] test run locally and hpc
-[x] run simulation of all models (including anti-correlated - pFactor=0.5 )

# Changes in loop code

- declare list:
self.lifeTimes = []

- add trapped time:
trappedChannels.append({'t': tau, 'E': E, 't_trapped': n})

- calculate and lifetime: 
self.lifeTimes.append(n - ch["t_trapped"])


## Updates Required:

-[x] histogram of #QPs trapped
-[x] histogram of #QPs released
-[x] histogram of release time of QPs
-[x] histogram of trapping times of QPs
-[x] histogram of lifetimes of #QPs trapped
-[x] histograms of how long QP spends in each mode

