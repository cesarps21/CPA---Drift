import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

#Loading data from MatLab file
dataout=scipy.io.loadmat('output_data.mat') 
datahtraces = scipy.io.loadmat('hardware_traces.mat')

 # Output of the last round of encryption (1000)
out_values = dataout["output_data"]

# Power consumption matrix (1000, 200)
traces = datahtraces["traces"] 


start_row = 5000
end_row = 8000
interval_width = 100 # The width of each interval
interval_width2 = 200
interval_width3 = 300
num_intervals = ( end_row - start_row) // interval_width # calculates the number of intervals
num_intervals2 = ( end_row - start_row) // interval_width2
num_intervals3 = ( end_row - start_row) // interval_width3
intervals = [] #Defining empty intervals
intervals2 = []
intervals3 = []
gradual_drift = traces.astype(float) #Copyng the initial traces
gradual_drift2 = traces.astype(float)
gradual_drift3 = traces.astype(float)

# Assign the initial and end point for each interval
for i in range(num_intervals):
    intervals.append([start_row + i * interval_width, start_row + (i + 1) * interval_width])

intervals.sort(key=lambda x: x[0])

for i in range(num_intervals2):
    intervals2.append([start_row + i * interval_width2, start_row + (i + 1) * interval_width2])

intervals2.sort(key=lambda x: x[0])

for i in range(num_intervals3):
    intervals3.append([start_row + i * interval_width3, start_row + (i + 1) * interval_width3])

intervals3.sort(key=lambda x: x[0])

# Apply the gradual drift according to P [Y [t] = a.X[t] + c.(i + N (μ, σ))]
for i in range(num_intervals):
    b = intervals[i][1]
    a = intervals[i][0]
    drift = 0.01 * ((interval_width) + np.round(np.random.normal(0, 1)))  # Adjust the drift magnitude based on the row
    
    if random.random() < 0.5: # bernoulli trial
       gradual_drift[a:b, :] += drift
    else:
        pass


for i in range(num_intervals2):
    d = intervals2[i][1]
    c = intervals2[i][0]
    drift2 = 0.01 * ((interval_width2) + np.round(np.random.normal(0, 1)))  # Adjust the drift magnitude based on the row
    
    if random.random() < 0.5: # bernoulli trial
       gradual_drift2[c:d, :] += drift2
    else:
        pass

for i in range(num_intervals3):
    f = intervals3[i][1]
    e = intervals3[i][0]
    drift3 = 0.01 * ((interval_width3) + np.round(np.random.normal(1, 1)))  # Adjust the drift magnitude based on the row
    
    if random.random() < 0.5: # bernoulli trial
       gradual_drift3[e:f, :] += drift3
    else:
        pass

#Apply the last value of drift to the rest of elements
gradual_drift[end_row:, :] += drift
gradual_drift2[end_row:, :] += drift2
gradual_drift3[end_row:, :] += drift3

# Calculating the mean per each column
row_indices = np.arange(10000)
traces_avg_values = np.mean(traces, axis=1)
gradual_drift_avg_values = np.mean(gradual_drift, axis=1)
gradual_drift2_avg_values = np.mean(gradual_drift2, axis=1)
gradual_drift3_avg_values = np.mean(gradual_drift3, axis=1)

# Plotting all the cases defined
plt.plot(row_indices, gradual_drift_avg_values, color='limegreen', label='Case 1: P[y =x+0.01*(100+N(0,1))]')
plt.plot(row_indices, gradual_drift2_avg_values, color='gold', label='Case 2: P[y =x+0.01*(200+N(0,1))]')
plt.plot(row_indices, gradual_drift3_avg_values, color='orange', label='Case 3: P[y =x+0.01*(300+N(1,1))]')
plt.plot(row_indices, traces_avg_values, color='deepskyblue', label='No Drift')

plt.xlabel('Traces')
plt.ylabel('Power Values')
plt.title('Power Consumption - Gradual Drift')
plt.legend()
plt.show()


