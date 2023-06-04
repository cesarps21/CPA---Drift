import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random

#Loading data from MatLab file
dataout=scipy.io.loadmat('output_data.mat') 
datahtraces = scipy.io.loadmat('hardware_traces.mat')

#Output of the last round of encryption (1000)
out_values = dataout["output_data"]

# Power consumption matrix (1000, 200)
traces = datahtraces["traces"] 

row_indices = np.arange(10000)

drift_start = 5000
start_row = 5000
end_row = 8000

sudden_drift2 = traces.astype(float)
increment_drift2 = traces.astype(float)
gradual_drift2 = traces.astype(float)

# # Assign the initial and end point for each interval
interval_width2 = 200
num_intervals2 = ( end_row - start_row) // interval_width2
intervals2 = []
for i in range(num_intervals2):
    intervals2.append([start_row + i * interval_width2, start_row + (i + 1) * interval_width2])

intervals2.sort(key=lambda x: x[0])


# Sudden drift based on Y[t] = a.X[t] + b + N (μ, σ)
sudden_drift2[drift_start:end_row,:] = (sudden_drift2[drift_start:end_row,:])*1.1+ 2 +np.round(np.random.normal(0, 2, size=(end_row-drift_start, 2000)))

# Incremental drift according to Y [t] = a.X[t] + c.(r + N (μ, σ))
for row in range(start_row, end_row):

    drift2 = 0.01 * ((row - start_row)/10 + np.round(np.random.normal(0, 1)) ) # Adjust the drift magnitude based on the row
    increment_drift2[row, :] += drift2

# Gradual drift according to P [Y [t] = a.X[t] + c.(i + N (μ, σ))]
for i in range(num_intervals2):
    d = intervals2[i][1]
    c = intervals2[i][0]
    drift2 = 0.01 * ((interval_width2) + np.round(np.random.normal(0, 1)))  # Adjust the drift magnitude based on the row
    
    if random.random() < 0.5: # bernoulli trial
       #print(intervals[i][1], intervals[i][0])
       gradual_drift2[c:d, :] += drift2   
    else:
        pass

# Calculating the mean per each column
traces_avg_values = np.mean(traces, axis=1)
sudden_drift2_avg_values = np.mean(sudden_drift2, axis=1)
increment_drift2_avg_values = np.mean(increment_drift2, axis=1)
gradual_drift2_avg_values = np.mean(gradual_drift2, axis=1)

# Plotting all the cases defined
plt.plot(row_indices, sudden_drift2_avg_values, color='limegreen', label='Case 1: f1->Sudden')
plt.plot(row_indices, increment_drift2_avg_values, color='gold', label='Case 2: f1->Incremental')
plt.plot(row_indices, gradual_drift2_avg_values, color='orange', label='Case 3: f2->Gradual')
plt.plot(row_indices, traces_avg_values, color='deepskyblue', label='No Drift')

plt.xlabel('Traces')
plt.ylabel('Power Values')
plt.title('Power Consumption - Recurring Drift')
plt.legend()
plt.show()













