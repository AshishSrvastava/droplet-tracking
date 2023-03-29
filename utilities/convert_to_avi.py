from fileinput import filename
import os
# import subprocess

directory_path = 'videos/20220413'

for root, dirs, file_names in os.walk(directory_path, topdown=False):
    for file in file_names:
        print(f"File: {file}")
        # error catching
        try:
            _format = ""
            
            
            inputfile = os.path.join(root, filename)
            print('[INFO] 1',inputfile)
            outputfile = os.path.join(dst, filename.lower().replace(_format, ".mp4"))
            subprocess.call(['ffmpeg', '-i', inputfile, outputfile])  
      
        except:
            print("An error occured: not a valid input file format")
