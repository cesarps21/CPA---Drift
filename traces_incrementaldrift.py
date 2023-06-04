import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sys

# Create the baseline matrix
 #Loading data from MatLab file
dataout=scipy.io.loadmat('output_data.mat')
datahtraces = scipy.io.loadmat('hardware_traces.mat')

 # Output of the last round of encryption (1000)
out_values = dataout["output_data"]

 # Power consumption matrix (1000, 200)
traces = datahtraces["traces"]

# Define the starting and ending points of the drift
start_row = 5000
end_row = 8000

# Define the drift rate or magnitude
drift_rate = 0.01  # Example: Increase the values by 1% per row

increment_drift = traces.astype(float)
increment_drift2 = traces.astype(float)
increment_drift3 = traces.astype(float)
row_indices = np.arange(10000)

# Apply Incremental drift according to Y [t] = a.X[t] + c.(r + N (μ, σ))
for row in range(start_row, end_row):
    drift = 0.01 * ((row - start_row)/20 + np.round(np.random.normal(0, 1)) ) # Adjust the drift magnitude based on the row
    increment_drift[row, :] += drift

    drift2 = 0.01 * ((row - start_row)/10 + np.round(np.random.normal(0, 1)) ) # Adjust the drift magnitude based on the row
    increment_drift2[row, :] += drift2

    drift3 = 0.02* ((row - start_row)/10 + np.round(np.random.normal(1, 1)) ) # Adjust the drift magnitude based on the row
    increment_drift3[row, :] += drift3
   

# Apply the last value of drift to the rest of elements
increment_drift[end_row:, :] += drift
increment_drift2[end_row:, :] += drift2
increment_drift3[end_row:, :] += drift3


traces_avg_values = np.mean(traces, axis=1)
increment_drift_avg_values = np.mean(increment_drift, axis=1)
increment_drift2_avg_values = np.mean(increment_drift2, axis=1)
increment_drift3_avg_values = np.mean(increment_drift3, axis=1)


plt.plot(row_indices, increment_drift_avg_values, color='limegreen', label='Case 1: y=x+0.01[r1+N(0,1)]')
plt.plot(row_indices, increment_drift2_avg_values, color='gold', label='Case 2: y=x+0.01[r2+N(0,1)]')
plt.plot(row_indices, increment_drift3_avg_values, color='orange', label='Case 3: y=x+0.02[r2+N(1,1)]')
plt.plot(row_indices, traces_avg_values, color='deepskyblue', label='No Drift')

plt.xlabel('Traces')
plt.ylabel('Power Values')
plt.title('Power Consumption - Incremental Drift')
plt.legend()
plt.show()