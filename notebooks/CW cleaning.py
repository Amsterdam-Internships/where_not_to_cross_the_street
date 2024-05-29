import geopandas as gpd
import laspy
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
import numpy as np
import matplotlib.pyplot as plt
import copy
import shapely as shp
import open3d as o3d
import pickle
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import re
from shapely.ops import transform
import pyproj
import os

os.system('sudo blobfuse /home/azureuser/cloudfiles/code/blobfuse/sidewalk --tmp-path=/mnt/resource/blobfusetmp --config-file=/home/azureuser/cloudfiles/code/blobfuse/fuse_connection_sidewalk.cfg -o attr_timeout=3600 -o entry_timeout=3600 -o negative_timeout=3600 -o allow_other -o nonempty')
os.system('sudo blobfuse /home/azureuser/cloudfiles/code/blobfuse/ovl --tmp-path=/mnt/resource/blobfusetmp --config-file=/home/azureuser/cloudfiles/code/blobfuse/fuse_connection_ovl.cfg -o attr_timeout=3600 -o entry_timeout=3600 -o negative_timeout=3600 -o allow_other -o nonempty')

# Load polygons
CW_polygons = gpd.read_file("/home/azureuser/cloudfiles/code/blobfuse/sidewalk/processed_data/crossings_project/T2N output/Venserpolder/CW polygons.shp")
CW_polygons = CW_polygons.drop(columns=['FID'])

project = pyproj.Transformer.from_proj(
    pyproj.Proj(init="EPSG:4326"), # source coordinate system
    pyproj.Proj(init="EPSG:28992")) # destination coordinate system

def apply_projection(geometry):
    # Your projection transformation code here
    transformed_geometry = transform(project.transform, geometry)
    return transformed_geometry

CW_polygons['geometry'] = CW_polygons['geometry'].apply(apply_projection)

# Function to get the PC file names
def get_PC_files(folder):
    # Initiate list to save file coordinates
    file_list = []
    file_names = []

    # Get file names
    files = os.listdir(folder)

    # Pattern to filter out integers
    pattern = r'\d+'

    for file_name in files:

        # Search for .laz files
        match = re.search("\.laz$", file_name)

        if match:
            integers = re.findall(pattern, file_name)
            file_list.append(integers)
            file_names.append(file_name)
    
    return file_list, file_names  

# Insert location point clouds
PC_location = "/home/azureuser/cloudfiles/code/blobfuse/ovl/pointcloud/Unlabeled/Amsterdam/nl-amsd-200923-7415-laz/las_processor_bundled_out"

# Get XY coordinates of PC files and file names
PC_XYs, PC_file_names = get_PC_files(PC_location)
def find_PC_files(CWs, PC_XYs, PC_file_names):
    CWs_dict = []
    PC_list = []

    # Loop over all crosswalks
    for CW in CWs.iterrows():

        # Get the polygon of the crosswalk
        polygon = CW[1]['geometry']
        
        # Get min and max coordinates of polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Divide by 50 to be in accordance with PC file names
        minx_50 = minx/50
        miny_50 = miny/50
        maxx_50 = maxx/50
        maxy_50 = maxy/50

        # Round the bounds to find the corresponding PC files
        rounded_bounds = [int(minx_50), int(miny_50), int(maxx_50), int(maxy_50)]
        
        # Find file that corresponds to each bound
        minx_miny = "filtered_" + str(rounded_bounds[0]) + "_" + str(rounded_bounds[1])
        minx_maxy = "filtered_" + str(rounded_bounds[0]) + "_" + str(rounded_bounds[3])
        maxx_miny = "filtered_" + str(rounded_bounds[2]) + "_" + str(rounded_bounds[1])
        maxx_maxy = "filtered_" + str(rounded_bounds[2]) + "_" + str(rounded_bounds[3])
        # We create a list of the files
        files = [minx_miny, minx_maxy, maxx_maxy, maxx_miny]

        # We also need to check if the boundaries are close to the PC border, if this is the case, we add the PC next to it to the PC list
        minx_dec = minx_50 - int(minx_50)
        miny_dec = miny_50 - int(miny_50)
        maxx_dec = maxx_50 - int(maxx_50)
        maxy_dec = maxy_50 - int(maxy_50)

        minx_low = rounded_bounds[0] - 1
        miny_low = rounded_bounds[1] - 1
        maxx_high = rounded_bounds[2] + 1
        maxy_high = rounded_bounds[3] + 1
        

        if minx_dec < 0.04:
            minx_miny_dec = "filtered_" + str(minx_low) + "_" + str(rounded_bounds[1])
            minx_maxy_dec = "filtered_" + str(minx_low) + "_" + str(rounded_bounds[3])

            files.append(minx_miny_dec)
            files.append(minx_maxy_dec)
        
        if miny_dec < 0.04:
            minx_miny_dec = "filtered_" + str(rounded_bounds[0]) + "_" + str(miny_low)
            maxx_miny_dec = "filtered_" + str(rounded_bounds[2]) + "_" + str(miny_low)

            files.append(minx_miny_dec)
            files.append(maxx_miny_dec)
        
        if maxx_dec > 0.96:         
            maxx_miny_dec = "filtered_" + str(maxx_high) + "_" + str(rounded_bounds[1])
            maxx_maxy_dec = "filtered_" + str(maxx_high) + "_" + str(rounded_bounds[3])

            files.append(maxx_miny_dec)
            files.append(maxx_maxy_dec)

        if maxy_dec > 0.96:
            minx_maxy_dec = "filtered_" + str(rounded_bounds[0]) + "_" + str(maxy_high)
            maxx_maxy_dec = "filtered_" + str(rounded_bounds[2]) + "_" + str(maxy_high)

            files.append(minx_maxy_dec)
            files.append(maxx_maxy_dec)

        if minx_dec < 0.04 and maxy_dec > 0.96: 
            minx_maxy_dec = "filtered_" + str(minx_low) + "_" + str(maxy_high)

            files.append(minx_maxy_dec)

        if minx_dec < 0.04 and miny_dec < 0.04:
            minx_miny_dec = "filtered_" + str(minx_low) + "_" + str(miny_low)

            files.append(minx_miny_dec)

        if maxx_dec > 0.96 and maxy_dec > 0.96:
            maxx_maxy_dec = "filtered_" + str(maxx_high) + "_" + str(maxy_high)

            files.append(maxx_maxy_dec)

        if maxx_dec > 0.96 and miny_dec < 0.04:
            maxx_miny_dec = "filtered_" + str(maxx_high) + "_" + str(miny_low)

            files.append(maxx_miny_dec)

        # Finally we create a set of the list of files to prevent duplicates
        files = list(set(files))
               
        CW_dict = {
            'CW_index': CW[0],
            'CW_polygon': polygon,
            'PC_list': files
        }

        CWs_dict.append(CW_dict)
        PC_list.extend(files)
    
    PC_list = list(set(PC_list))

    return CWs_dict, PC_list

CWs, PC_list = find_PC_files(CW_polygons, PC_XYs, PC_file_names)

def load_PCs(PC_list, folder):
    PCs = []
    
    for pc_name in PC_list:
        file = os.path.join(folder, pc_name + ".laz")
        if os.path.exists(file):
            laz_file = laspy.read(file)
            name = pc_name.split(".")[0]
            PC_coords = laz_file.xyz
            PC_intensity = laz_file.intensity

            PCs.append({"name": name, "laz_file": laz_file, "PC_coords": PC_coords, "PC_intensity": PC_intensity})
           
    return PCs

PCs = load_PCs(PC_list, PC_location)

def check_CW_PC(CWs, PCs):

    PC_names = []

    # Loop over PCs dictionary and save PC names
    for PC in PCs:
        PC_names.append(PC['name'])

    # Loop over CWs and check if the PCs exist

    for CW in CWs:

        for PC in CW['PC_list']:
            
            if PC not in PC_names:

                CW['PC_list'] = CW['PC_list'].remove(PC)
    
    # Remove CWs that have no PCs as they are outside the targeted area
    filtered_CWs = []

    for CW_check in CWs:

        if CW_check['PC_list'] is not None:
            filtered_CWs.append(CW_check)
    
    return filtered_CWs

def cut_PC(pc):
    # Find coordinates below threshold
    indices = np.where(pc['PC_coords'][:, 2] < 1)
    z = pc['PC_coords'][:, 2]
    average_z = np.mean(z)

    # Cut intensity accordingly
    pc['PC_intensity_low'] = pc['PC_intensity'][indices]
    pc['PC_coords_low'] = pc['PC_coords'][indices]

    # Compute average and std z
    z = pc['PC_coords_low'][:, 2]
    average_z = np.mean(z)
    sd_z = np.std(z)
    threshold = average_z + sd_z

    # Compute points above threshold
    indices_thres = np.where(pc['PC_coords_low'][:, 2] < threshold)
    pc['PC_intensity_low'] =  pc['PC_intensity_low'][indices_thres] 
    pc['PC_coords_low'] =  pc['PC_coords_low'][indices_thres] 

    return pc

def down_sample_PC(pc, coords, intensity_string):
    xyz = pc[coords]
    intensity = pc[intensity_string]
    
    # Convert to Open3D point cloud using only XYZ
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(xyz)

    # Perform voxel downsampling
    downsampled_pc_o3d = pc_o3d.voxel_down_sample(0.02)

    # Retrieve downsampled XYZ points
    pc[coords + "_ds"] = np.asarray(downsampled_pc_o3d.points) 

    # Create a KDTree for the original point cloud
    tree = cKDTree(xyz)

    # For each downsampled point, find its nearest neighbor in the original cloud
    _, indices = tree.query(pc[coords + '_ds'])

    # Get indices of intensity
    pc[intensity_string + "_ds"] = intensity[indices]

    return pc

def cut_ds_PC(PCs, coord_string, intensity_string):

    for pc in PCs:
        pc = cut_PC(pc)
        pc = down_sample_PC(pc, coord_string, intensity_string)
    
    return PCs

PCs_cut = cut_ds_PC(PCs, "PC_coords_low", "PC_intensity_low")

# Cut point clouds based on polygon coordinate
def PC_pol_match(PC, pol):

    # Get the bounding box (rectangle) of the polygon
    minx, miny, maxx, maxy = pol['CW_polygon'].bounds
    
    # Determine condition based on polygon bounds
    condition = ((PC['PC_coords_low_ds'][:, 0] > minx) & (PC['PC_coords_low_ds'][:, 0] < maxx) 
                &  (PC['PC_coords_low_ds'][:, 1] > miny) & (PC['PC_coords_low_ds'][:, 1] < maxy))

    # Apply condition to get indices
    indexes = np.where(condition)

    # Check if any matches were found
    if len(indexes[0]) > 0:
        
        # Apply indexing to coordinates and intensity
        intensity = PC['PC_intensity_low_ds'][indexes]
        coords = PC['PC_coords_low_ds'][indexes]

        return {'CW_index': pol['CW_index'], 'polygon': pol['CW_polygon'], 'PC_file': [PC['name']], 'PC_coords_low_ds': coords, 'PC_intensity_low_ds': intensity}

# Function to merge polygons that are spread over two point clouds
def merge_matches(match1, match2):

    # Concatenate coordinates and intensity of both PC files belonging to the same polygon 
    coords = np.vstack((match1['PC_coords_low_ds'], (match2['PC_coords_low_ds'])))
    intensity = np.hstack((match1['PC_intensity_low_ds'], (match2['PC_intensity_low_ds'])))
    
    # Create list of PC files to add to dictionary 
    PC_list = match1['PC_file'] + match2['PC_file']
    
    # Create dictionary for matched point clouds
    new_match = {'CW_index': match1['CW_index'], 'polygon': match1['polygon'], 'PC_file': PC_list, 'PC_coords_low_ds': coords, 'PC_intensity_low_ds': intensity}
    
    return new_match

def group_matches(all_matches):
    # Create list to group together polygons that are spread over multiple point clouds
    grouped_data = []

    # Create a deep copy of the previously identified matches
    match_copy = copy.deepcopy(all_matches)

    # Loop over all matches
    for item in match_copy:

        index = item['CW_index']

        found = False

        for sublist in grouped_data:

            # Check if the polygon is already in the list and append to the corresponding list item if this is the case
            if sublist and sublist[0]['CW_index'] == index:
                sublist.append(item)
                found = True
                break
            
        # If the polygon is not already in the list, append it 
        if not found:
            grouped_data.append([item])
    
    return grouped_data

def process_grouped_matches(grouped_data):
    # Loop over the grouped polygons
    for group in grouped_data:

        # Check if there is multiple PC files for one polygon 
        if len(group) > 1:

            # Loop over each item except the last one
            for i in range(len(group) - 1):

                # Merge the first item with the next one and replace the first item 
                match = merge_matches(group[0], group[1])
                
                group[0] = match

                group.pop(1)

    # Flatten the grouped data list as each list item only has one item now
    grouped_data_flat = [item for sublist in grouped_data for item in sublist]

    return grouped_data_flat

def match_PC_pol(CW_polygons, PCs):

    # Create list to save all matches found
    all_matches = []

    # Loop over all polygons
    for index in range(0, len(CW_polygons)):
        cw = CW_polygons[index]
        #print(cw['PC_list'])

        for pc in PCs:
            if pc['name'] in cw['PC_list']:
                match = PC_pol_match(pc, cw)
                if match:
                    all_matches.append(match)
    
    grouped_data = group_matches(all_matches)
    merged_data = process_grouped_matches(grouped_data)
    
    return merged_data, grouped_data, all_matches

merged_data, grouped_data, all_matches = match_PC_pol(CWs, PCs_cut)

def filter_buffer(buffer):
    buffer_copy = copy.deepcopy(buffer)

    # Determine condition based on polygon bounds
    condition = (buffer_copy['intensity'] > 30000)

    # Apply condition to get indices
    indexes = np.where(condition)
    
    # Check if any matches were found
    if len(indexes[0]) > 0:
        
        # Apply indexing to coordinates and intensity
        buffer_copy['intensity_filtered'] = np.array(buffer_copy['intensity'][indexes[0]])
        buffer_copy['coords_filtered'] = np.array(buffer_copy['coords'][indexes[0]])
        
    return buffer_copy

# Filter for intensity
def filter_intensity(cw, min_intensity):
    return_cw = copy.deepcopy(cw)

    # Determine condition based on polygon bounds
    condition = (return_cw['PC_intensity_low_ds'] > min_intensity)

    # Apply condition to get indices
    indexes = np.where(condition)

    # Check if any matches were found
    if len(indexes[0]) > 0:
        
        # Apply indexing to coordinates and intensity
        return_cw['PC_intensity_low_ds_filtered'] = return_cw['PC_intensity_low_ds'][indexes]
        return_cw['PC_coords_low_ds_filtered'] = return_cw['PC_coords_low_ds'][indexes]
       

        return return_cw

def cluster_pol(CW):
    # Create list to save the clusters that are found
    cluster_list = []

    # Filter the original polygon to only include points with a high intensity
    filtered = filter_intensity(CW, 30000)
    

    if filtered:
    
        # Use DBSCAN to cluster the points in the polygon
        dbscan = DBSCAN(eps=0.1, min_samples=5)
        dbscan.fit(filtered['PC_coords_low_ds_filtered'])

        # Get labels created by DBSCAN
        labels = dbscan.labels_

        # Create dictionary to save clusters
        cluster_data = {}

        # Loop over each point in the filtered polygon and check to which cluster it belongs
        # Group coordinates and intensity values based on their label in the cluster_data dictionary
        for label, point, value in zip(labels, filtered['PC_coords_low_ds_filtered'], filtered['PC_intensity_low_ds_filtered']):
            if label not in cluster_data:
                cluster_data[label] = {'coords': [], 'intensity': []}  
            cluster_data[label]['coords'].append(point)
            cluster_data[label]['intensity'].append(value)
        
        # Transform the coordinates and intensity values to np arrays to make them easier to work with
        for label in np.unique(labels):
            cluster_data[label]['coords'] = np.array(cluster_data[label]['coords'])
            cluster_data[label]['intensity'] = np.array(cluster_data[label]['intensity'])

        # Loop over the created clusters and save them in the cluster_list
        for cluster in cluster_data:
            cluster_dict = {}

            # Only keep clusters that are over 100 points to pre-emptively filter out noise
            if (len(cluster_data[cluster]['coords']) > 50):

                # Save cluster in a similar manner as the original polygon
                cluster_dict['CW_index'] = CW['CW_index']
                cluster_dict['PC_file'] = CW['PC_file']
                cluster_dict['coordinates'] = cluster_data[cluster]['coords']
                cluster_dict['intensity'] = cluster_data[cluster]['intensity']
                
                cluster_list.append(cluster_dict)
        
        return cluster_list
def grow_cluster(PC_coords, PC_intensity, cluster_coords):

    # Initialize an empty list to keep track of the points that are added to the clusters
    added = []

    # Build a KDTree for fast nearest neighbor search
    tree = cKDTree(PC_coords)

    # Define the radius within which points are considered neighbors
    radius = 0.12

    # Initialize the starting coordinates for the cluster growth as the original cluster
    coords = cluster_coords

    while True:

        # Find indices of neighbors within the specified radius
        neighbor_indices = tree.query_ball_point(coords, radius)

        # Initialize a list to store unique inidces of new points to add
        indices = []

        # Iterate through the neighbor indices to see if points have already been added
        for index in neighbor_indices:
            for i in index:
                # Add them if this is not the case
                if i not in added:
                    indices.append(i)
                    added.append(i)

        # Remove duplicates from the lists of indices
        indices = list(set(indices))
        added = list(set(added))

        # If no new points are found, exit the loop
        if len(indices) == 0:
            break

        # Retrieve coordinates and intensities of the neighboring points
        neighbor_coords = PC_coords[indices]
        neighbor_intensities = PC_intensity[indices]

        # Store the neighbors in a temporary dictionary
        temp = {'coords': neighbor_coords, 'intensity': neighbor_intensities}

        # Apply a filtering function to the temporary dictionary to only keep points with a high intensity
        temp_filtered = filter_buffer(temp)

        # If new filtered coordinates are available, update the coordinates for the next iteration
        if 'coords_filtered' in temp_filtered:
            coords = temp_filtered['coords_filtered']

    # Extract the final cluster coordinates and intensities
    cluster_coords = PC_coords[added]
    cluster_intensity = PC_intensity[added]

    # Store the final cluster information in a dictionary
    final = {'coords': cluster_coords, 'intensity': cluster_intensity}

    # Apply filtering to the final cluster
    final = filter_buffer(final)
    
    # Return the filtered final cluster
    return final

def get_clusters(polygon, PCs, PC_coords_string, PC_intensity_string): 

    # Initialize coordinate and intensity array
    PC_coords_temp = []
    PC_intensity_temp = []

    # Get PC file that corresponds to that of the original polygon
    for PC_name in polygon['PC_file']:
        PC = list(filter(lambda PC: PC['name'] == PC_name, PCs))
        
        sub_PC_coords = PC[0][PC_coords_string]
        sub_PC_intensity = PC[0][PC_intensity_string]
        
        PC_coords_temp.append(sub_PC_coords)
        PC_intensity_temp.append(sub_PC_intensity)
    
    
    PC_coords = np.concatenate(PC_coords_temp, axis=0)
    PC_intensity = np.concatenate(PC_intensity_temp, axis=0)

    # Get clusters from polygon
    cluster_dict = cluster_pol(polygon)

    if cluster_dict:

        # For each found cluster, grow it and update the cluster data
        for cluster in cluster_dict:
            clean_cluster = grow_cluster(PC_coords, PC_intensity, cluster['coordinates'])
            if 'coords_filtered' in clean_cluster:
                cluster['clean_coords'] = clean_cluster['coords_filtered']
                cluster['clean_intensity'] = clean_cluster['intensity_filtered']
        
        # Return cluster dictionary
        return cluster_dict
# Function to merge polygons that are spread over two point clouds
def merge_clusters(cluster_list):

    # Initialize arrays final cw
    CW_name = []
    PC_file = []
    PC_coords= []
    PC_intensity = []
    PC_coords_clean = []
    PC_intensity_clean = []

    # Get PC file that corresponds to that of the original polygon++++++++
    
    for cluster in cluster_list:
       
        CW_name.append(cluster['CW_index'])
        PC_file.append(cluster['PC_file'])
        PC_coords.append(cluster['coordinates'])
        PC_intensity.append(cluster['intensity'])

        if 'clean_coords' in cluster:
            PC_coords_clean.append(cluster['clean_coords'])
            PC_intensity_clean.append(cluster['clean_intensity'])
    
    PC_file = np.concatenate(PC_file, axis=0)
    PC_coords = np.concatenate(PC_coords, axis=0)
    PC_intensity = np.concatenate(PC_intensity, axis=0)
    PC_coords_clean = np.concatenate(PC_coords_clean, axis=0)
    PC_intensity_clean = np.concatenate(PC_intensity_clean, axis=0)

    cw = {}

    cw['CW_index'] = list(set(CW_name))
    cw['PC_file'] = list(set(PC_file))
    cw['coordinates'] = PC_coords
    cw['intensity'] = PC_intensity
    cw['coordinates_clean'] = PC_coords_clean
    cw['intensity_clean'] = PC_intensity_clean
       
    return cw

final = []
for merge in merged_data:
    cluster_dict = get_clusters(merge, PCs_cut, 'PC_coords_low_ds', 'PC_intensity_low_ds')
    if cluster_dict:
        final.append(cluster_dict)

path = "/home/azureuser/cloudfiles/code/blobfuse/sidewalk/processed_data/crossings_project/CW cleaning/Venserpolder/clusterdicts Venserpolder.pkl"

with open(path, 'wb') as file:
    pickle.dump(final, file)



