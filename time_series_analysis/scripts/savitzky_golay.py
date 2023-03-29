import sys
import numpy as np
from scipy.signal import savgol_filter

# get input filename from command line argument
if len(sys.argv) < 2:
    print("Please provide an input filename as a command line argument")
    sys.exit()
    
input_filename = sys.argv[1]

# open the input file and read the data
with open(input_filename, 'r') as f:
    data = f.readlines()
    
# ignore first line (header)
data = data[1:]

# parse the data into separate lists for position and time
positions = []
times = []

for line in data:
    # split eacch line into it's position and time
    time, x, y, width, height, deformation = line.strip().split()
    positions.append(float(x))
    times.append(float(time))

# calculate the velocities based on the position and time data
velocities = []
for i in range(1, len(positions)):
    # calculate the velocity for each time step by dividing the change in position by the change in time
    vel = (positions[i] - positions[i-1]) / (times[i] - times[i-1])
    velocities.append(vel)
    
# calculate the moving average of the velocities using a window size of 5
smoothed_velocities = savgol_filter(velocities, window_length=5, polyorder=4)

# write finite difference velocity and moving averages to a file
with open('droplet_velocity_time.txt', 'w') as f:
    f.write("frame" + ' ' + "velocity" + ' ' + "smoothed_velocity" + '\n')
    for time, vel, smoothed_vel in zip(times[1:], velocities, smoothed_velocities):
        f.write(str(time) + ' ' + str(vel) + ' ' + str(smoothed_vel) + '\n')

# output a statement showing how to print it
# print("You can view the graph of the input position vs time file and the output velocity vs time file by running the following command in the terminal:")
print(f"""gnuplot -persist -e "set multiplot layout 1,2; set size square; plot '{input_filename}' every ::2; plot 'droplet_velocity_time.txt' every ::2; unset multiplot" """)
