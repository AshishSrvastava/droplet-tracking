import sys

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

# parse the data into separate lists for frame, x, y, width, height, and deformation
frames = []
xs = []
ys = []
widths = []
heights = []
deformations = []

for line in data:
    # split eacch line into it's frame, x, y, width, height, and deformation
    frame, x, y, width, height, deformation = line.strip().split()
    frames.append(float(frame))
    xs.append(float(x))
    ys.append(float(y))
    widths.append(float(width))
    heights.append(float(height))
    deformations.append(float(deformation))
    
# define the window size for the moving average
window_size = 10

# initialize a list to store the moving average values
moving_averages = []

# calculate the moving average of the deformation values
for i in range(len(deformations)):
    # calculate the start and end indices for the window
    start_idx = max(0, i - window_size)
    end_idx = min(len(deformations), i + window_size + 1)
    
    # calculate the average of the deformation values within the window
    window_average = sum(deformations[start_idx:end_idx]) / (end_idx - start_idx)
    
    # append the average value to the moving_averages list
    moving_averages.append(window_average)
    
# write the moving averages to a file
with open('moving_average_deformation.txt', 'w') as f:
    f.write("frame \t moving_average_deformation \n")
    for frame, avg in zip(frames, moving_averages):
        f.write(str(frame) + '\t' + str(avg) + '\n')

# output a statement showing how to print the graph
print("You can view the graph of the input deformation vs frame file and the output moving average deformation vs frame file by running the following command in the terminal:")
print(f"""gnuplot -persist -e "plot 'moving_average_deformation.txt' every ::2" """)
