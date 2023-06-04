import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

 #Loading data from MatLab file
dataout=scipy.io.loadmat('output_data.mat')
datahtraces = scipy.io.loadmat('hardware_traces.mat')

 # Output of the last round of encryption (1000)
out_values = dataout["output_data"]

# Power consumption matrix (1000, 200)

traces = datahtraces["traces"] 

drift_start = 5000

sudden_drift = traces.astype(float) 
sudden_drift2 = traces.astype(float) 
sudden_drift3 = traces.astype(float) 

# adding sudden drift based on Y[t] = a.X[t] + b + N (μ, σ)
sudden_drift[drift_start:,:] = (sudden_drift[drift_start:,:])*1+ 1 +np.round(np.random.normal(0, 1, size=(10000-drift_start, 2000))) 
sudden_drift2[drift_start:,:] = (sudden_drift2[drift_start:,:])*1.1+ 2 +np.round(np.random.normal(0, 2, size=(10000-drift_start, 2000)))
sudden_drift3[drift_start:,:] = (sudden_drift3[drift_start:,:])*1.2 + 2 +np.round(np.random.normal(1, 2, size=(10000-drift_start, 2000)))

# Calculating the mean per each column
row_indices = np.arange(10000)
sudden_drift_avg_values = np.mean(sudden_drift, axis=1)
sudden_drift2_avg_values = np.mean(sudden_drift2, axis=1)
sudden_drift3_avg_values = np.mean(sudden_drift3, axis=1)
traces_avg_values = np.mean(traces, axis=1)

# Plotting all the cases defined
plt.plot(row_indices, sudden_drift_avg_values, color='limegreen', label='Case 1: y=x+1+N(0,1)')
plt.plot(row_indices, sudden_drift2_avg_values, color='gold', label='Case 2: y=1.1*x+2+N(0,2)')
plt.plot(row_indices, sudden_drift3_avg_values, color='orange', label='Case 3: y=1.2*x+2+N(1,2)')
plt.plot(row_indices, traces_avg_values, color='deepskyblue', label='No Drift')


plt.xlabel('Traces')
plt.ylabel('Power Values')
plt.title('Power Consumption - Sudden Drift')
plt.legend()
plt.show()