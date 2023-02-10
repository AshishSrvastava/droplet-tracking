import numpy as np
from scipy.fft import fft

# read the input data
data = np.loadtxt('droplet_posn_time.txt', skiprows=1, usecols=[1])

# calculate the sample time and time array
sample_time = 1  # assuming that the sample time is 1 second
N = len(data)
time = sample_time * np.arange(N)

# perform the Fourier transform
fft_result = fft(data)

# calculate the frequency array
freq = np.fft.fftfreq(N, d=sample_time)

# calculate the derivative in Fourier space
d_fft = np.array(fft_result) * 1j * 2 * np.pi * freq

# perform low-pass filtering by setting high frequency components to zero
cutoff = 400  # specify the cutoff frequency in Hz
d_fft[(np.abs(freq) > cutoff)] = 0

# perform the inverse Fourier transform to get the velocity
velocity = np.real(np.fft.ifft(d_fft))

# save the velocity time data to a file
np.savetxt('droplet_velocity_time_fourier.txt', np.column_stack((time, velocity)))

