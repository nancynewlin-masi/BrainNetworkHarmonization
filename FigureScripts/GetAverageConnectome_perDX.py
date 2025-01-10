import sys
import numpy as np
import os
import re
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# Directory containing the CSV files
directory_path = sys.argv[1]

# Pattern for matching filenames
patterns = [r"CONNECTOME_SiteID_(\w+)1_Sample_(\w+)\.csv",  r"CONNECTOME_SiteID_(\w+)2_Sample_(\w+)\.csv", r"CONNECTOME_SiteID_(\w+)3_Sample_(\w+)\.csv"]
#patterns = [r"CONNECTOME_ORIGINAL_SiteID_(\w+)1_Sample_(\w+)\.csv",  r"CONNECTOME_ORIGINAL_SiteID_(\w+)2_Sample_(\w+)\.csv", r"CONNECTOME_ORIGINAL_SiteID_(\w+)3_Sample_(\w+)\.csv"]


for dx in [1,2,3]:
    # Dictionary to store data per site
    site_data = {}
    pattern = patterns[dx-1]
    print(pattern)
    # Iterate over all CSV files in the directory that match the pattern
    for filename in glob(os.path.join(directory_path, "*.csv")):
        match = re.match(pattern, os.path.basename(filename))
        if match:
            site = match.group(1)
            # Read the CSV file as a NumPy array
            array_data = np.loadtxt(filename, delimiter=",")

            # Add the array to the site data dictionary
            if site not in site_data:
                site_data[site] = []
            site_data[site].append(array_data)

    # Calculate and print the average array for each site
    average_site_data = np.zeros((38,121,121))
    for site, arrays in site_data.items():
        print(site)

        # Stack arrays along a new axis and calculate the mean along that axis
        average_array = np.mean(np.stack(arrays), axis=0)
        #print("Size", average_array.shape)
        #average_array[average_array<np.max(average_array)*0.01] = 0
        #print(np.max(average_array)*0.001)
        average_site_data[int(site)-1,:,:] = average_array
        #print(f"Average array for site {site}:\n{average_array}\n")



    average_array = np.mean(np.stack(average_site_data), axis=0)
    print(average_array)
    np.save(f"AvgConnectome_MODEL_Dx-{dx}.npy",average_array)
    plt.imshow(average_array, interpolation='none', vmin=np.max(average_array)*0.005, vmax=np.max(average_array)*0.1)
    #plt.show()
    plt.savefig(f'AvgConnectome_MODEL_Dx-{dx}.png')
#exit(0)
