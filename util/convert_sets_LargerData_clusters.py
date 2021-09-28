#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
import os
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))

from tensorflow.keras.utils import to_categorical

#====================
# Functions =========
#====================

def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref '''
    ref = np.tile(ref, (len(coords[:,0]), 1))
    DeltaCoords = np.subtract(coords, ref)
    return np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2)

def find_max_dim(events, event_dict):
    ''' This function is designed to return the sizes of a numpy array such that we are efficient
    with zero padding. Notes: we add six to the maximum cluster number such that we have room for track info.
    Inputs:
        events: filtered list of events to choose from in an Nx3 format for event, track, cluster index 
        event_tree: the event tree dictionary
    Returns:
        3-tuple consisting of (number of events, maximum cluster_size, 6), 6 because of how we have structured
        the X data format in energyFlow to be Energy, Eta, Phi, rPerp, track_flag, sampling_layer
    '''
    nEvents = len(events)
    max_clust = 0
    for i in range(nEvents):
        evt = events[i,0]
        clust_idx = events[i,2]
        num_in_clust = len(event_dict['cluster_cell_ID'][evt][clust_idx])
        if num_in_clust > max_clust:
            max_clust = num_in_clust

    return (nEvents, max_clust+6, 6)

def find_max_dim_tuple_Maxcluster(events, event_dict):
    ###    find_max_dim_tuple_Maxcluster -> return (Number of selected events in file, max number of clusters per event, TOTAL NUMBER OF CLUSTERS IN FILE,  max number of cells per cluster) == (nEvents, max_clust_event, max_clust_total, max_cell, 6)
    ### Iput: events -> [sel evnt idx, nclust, clust arr]
    nEvents = len(events)
    max_clust_event = 0   #Max number of clusters per event
    max_cell = 0    #Max number of cells per cluster 
    max_clust_total = 0 # Total number of clusters in file   
    
    for i in range(nEvents):
        event =     events[i,0]
        tot_clust = events[i,1]
        clust_nums =events[i,2]

        if tot_clust>max_clust_event:
            max_clust_event=tot_clust

        max_clust_total += tot_clust
                    
        #if i<10 or tot_clust>4 :
        #    print("------ event ",i)            #Dilia
        #    print("nEvents",events[i,0])#Dilia
        #    print("tot_clust",events[i,1])#Dilia
        #    print("clust_nums",events[i,2])#Dilia
        #    print("max_clust_event",max_clust_event)#Dilia

 
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                if nInClust>max_cell:
                    max_cell=nInClust

         #       if i<10 or tot_clust>4:
         #           print(" -- cluster #", clst_idx)#Dilia
         #           print("  - cell nInClust ", nInClust)#Dilia
         #           print("  - max_cell ", max_cell)#Dilia
         #           if nInClust > 6:
         #               print("  - Big cluster with ncells:", nInClust)#Dilia
                        
                    
    
    print("---- Number of selected events in file : ", nEvents)         
    print("---- max number of clusters per event : ", max_clust_event)  
    print("---- TOTAL NUMBER OF CLUSTERS IN FILE : ", max_clust_total)  
    print("---- max number of cells per cluster : ", max_cell)          
    print("------------------------------------------------------")       
                    
     
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    return (nEvents, max_clust_event, max_clust_total , max_cell, 6)

def find_max_dim_tuple_cluster(events, event_dict):
    nEvents = len(events)
    max_clust = 0   #Max number of clusters per event
    max_cell = 0    #Max number of cells per cluster 
    
    for i in range(nEvents):
        event =     events[i,0]
        tot_clust = events[i,1]
        clust_nums =events[i,2]

        if tot_clust>max_clust:
            max_clust=tot_clust

                    
        #if i<10 or tot_clust>4 :
        #    print("------ event ",i)            #Dilia
        #    print("nEvents",events[i,0])#Dilia
        #    print("tot_clust",events[i,1])#Dilia
        #    print("clust_nums",events[i,2])#Dilia
        #    print("max_clust",max_clust)#Dilia

 
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                if nInClust>max_cell:
                    max_cell=nInClust

         #       if i<10 or tot_clust>4:
         #           print(" -- cluster #", clst_idx)#Dilia
         #           print("  - cell nInClust ", nInClust)#Dilia
         #           print("  - max_cell ", max_cell)#Dilia
         #           if nInClust > 6:
         #               print("  - Big cluster with ncells:", nInClust)#Dilia
                        
                    
        #if i<10 or tot_clust>4 :
        #    print("------------------")            #Dilia
                    
     
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    return (nEvents, max_clust,max_cell, 6)

def find_max_dim_tuple(events, event_dict):
    nEvents = len(events)
    max_clust = 0
    
    for i in range(nEvents):
        event = events[i,0]
        track_nums = events[i,1]
        clust_nums = events[i,2]
        
        clust_num_total = 0
        # set this to six for now to handle single track events, change later
        track_num_total = 6
        
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                # add the number in each cluster to the total
                clust_num_total += nInClust

        total_size = clust_num_total + track_num_total
        if total_size > max_clust:
            max_clust = total_size
    
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    # max_clust is the maximun number of cells per event
    return (nEvents, max_clust, 6)

def dict_from_tree(tree, branches=None, np_branches=None):
    ''' Loads branches as default awkward arrays and np_branches as numpy arrays. '''
    dictionary = dict()
    if branches is not None:
        for key in branches:
            branch = tree.arrays()[key]
            dictionary[key] = branch
            
    if np_branches is not None:
        for np_key in np_branches:
            np_branch = np.ndarray.flatten(tree.arrays()[np_key].to_numpy())
            dictionary[np_key] = np_branch
    
    if branches is None and np_branches is None:
        raise ValueError("No branches passed to function.")
        
    return dictionary

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec

#====================
# Metadata ==========
#====================
track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

ak_event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", "cluster_nCells",
                  "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", "trackPt", "trackP",
                  "trackMass", "trackEta", "trackPhi", "truthPartE", "cluster_ENG_CALIB_TOT", "cluster_E", "truthPartPt"]

np_event_branches = ["mcChannelNumber","nCluster", "eventNumber", "nTrack", "nTruthPart"]

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]

#====================
# File setup ========
#====================
fileNames = []
#pipm files
Nfile_pipm = 20 # 502 # Total Number of files for pipm
file_prefix_pipm = '/fast_scratch/atlas_images/v01-45/pipm/user.angerami.24559744.OutputStream._000'  
for i in range(1,Nfile_pipm+1):
    endstring = f'{i:03}'
    fileNames.append(file_prefix_pipm + endstring + '.root')

#pi0 files
Nfile_pi0 = Nfile_pipm  #~same number of files
file_prefix_pi0 = '/fast_scratch/atlas_images/v01-45/pi0/user.angerami.24559740.OutputStream._000'
for i in range(0,Nfile_pi0):
    endstring = f'{i+11:03}'
    fileNames.append(file_prefix_pi0 + endstring + '.root')
    

#Total number of files
Nfile = Nfile_pipm + Nfile_pi0


#====================
# Load Data Files ===
#====================

## GEOMETRY DICTIONARY ##
geo_file = ur.open('/fast_scratch/atlas_images/v01-45/cell_geo.root')
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

# Use this to compare with the dimensionality of new events
firstArray = True

## Maximun dimension of the memory map
MAX_cell_dim = 1500
MAX_clus_dim = 1700000

## MEMORY MAPPED ARRAY ALLOCATION ##
X_large = np.lib.format.open_memmap('/data/atlas/dportill/X_large.npy', mode='w+', dtype=np.float64,
                       shape=(MAX_clus_dim,MAX_cell_dim,6), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap('/data/atlas/dportill/Y_large.npy', mode='w+', dtype=np.float64,
                       shape=(MAX_clus_dim,2), fortran_order=False, version=None)

## For two files: Current size: (7497, 942, 6)
#X_large = np.zeros(shape=(10000,2000,6), dtype=np.float64 )
#Y_large = np.zeros(shape=(10000,3), dtype=np.float64 )

## The counters:
k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
tot_nCluster = 0 # Total number of clusters processed 
t_tot = 0 # total time
max_nPoints = 0 # used for keeping track of the largest 'point cloud' #Max number of cluster in a event
max_nCells = 0 # Max number of cells in a cluster
max_nclusters = 0 # Max number of clusters in a file


## Loop over files
for currFile in fileNames:
    # Check for file, a few are missing
    if not os.path.isfile(currFile):
        print()
        print('File '+currFile+' not found..')
        print()
        k += 1
        continue
    
    else:
        print()
        print('Working on File: '+str(currFile)+' - '+str(k)+'/'+str(Nfile))
        k += 1

    t0 = t.time()
    ## EVENT DICTIONARY ##
    event = ur.open(currFile)
    event_tree = event["EventTree"]
    event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)

    # Target Y label
    Label = -1 # classification label. 1:pipm, 0:pi0
    if event_dict['mcChannelNumber'][0]== 900247:
        print("pipm file")
        Label = 1
    elif event_dict['mcChannelNumber'][0]== 900246:
        print("pi0 file")
        Label = 0
    else:
        print("Error: can not associate the MC channel number to a known process")

    #===================
    # APPLY CUTS =======
    #===================
    # create ordered list of events to use for index slicing
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)

    nCluster = event_dict['nCluster']
    filtered_event_mask = nCluster != 0
    filtered_event = all_events[filtered_event_mask]
    print("* Selected events with clusters: ", len(filtered_event), "/", len(all_events))

    t1 = t.time()
    events_cuts_time = t1 - t0

    #============================================#
    ## CREATE INDEX ARRAY FOR  CLUSTERS ##
    #============================================#
    event_indices = []
    t0 = t.time()

    for evt in filtered_event:
        # pull cluster number, don't need zero index as it's loaded as a np array
        nClust = event_dict["nCluster"][evt]
        cluster_idx = np.arange(nClust)

        ## Cluster properties
        clusterEs = event_dict["cluster_E"][evt].to_numpy()
        clus_phi = event_dict["cluster_Phi"][evt].to_numpy()
        clus_eta = event_dict["cluster_Eta"][evt].to_numpy()

        ## ENERGY SELECTION AND ETA SELECTION
        eta_mask = abs(clus_eta) < 0.7
        e_mask = clusterEs > 0.5
        selection = eta_mask & e_mask
        selected_clusters = cluster_idx[selection]

        ## CREATE LIST ##
        if np.count_nonzero(selection) > 0:
            event_indices.append((evt, len(selected_clusters), selected_clusters)) #Added the number of selected cluster per event
            ##if (evt % 1000==0):
                ##print("** Event ", evt, " has ", np.count_nonzero(selection), "/" ,nClust," selected clusters") #Dilia

    event_indices = np.array(event_indices, dtype=np.object_)
    t1 = t.time()
    indices_time = t1 - t0

    #print("event_indices [sel evnt idx, nclust, clust arr]:", event_indices) #Dilia
    
    #=========================#
    ## DIMENSIONS OF X ARRAY ##
    #=========================#
    t0 = t.time()
    ## max_dims -> (events, clusters/event, clusters/file, cells/cluster,6)
    max_dims = find_max_dim_tuple_Maxcluster(event_indices, event_dict)
    evt_tot = max_dims[0]
    
    ## keeping track of total number of events, total number of clustes, max number of cells per cluster
    tot_nEvts += max_dims[0]
    tot_nCluster += max_dims[2]
    
    if max_dims[1] > max_nPoints:
        max_nPoints = max_dims[1]   
    if max_dims[3] > max_nCells:
        max_nCells = max_dims[3]

    print("max_dims for this file: ",max_dims)         
    #print('Total of processed events: '+str(tot_nEvts))   
    #print('Total number of processed clusters : '+str(tot_nCluster))

    t1 = t.time()
    find_create_max_dims_time = t1 - t0

    # Create arrays
    Y_new = np.zeros((max_dims[2]))                                #(total clusters/file)
    X_new = np.zeros((max_dims[2],max_dims[3],max_dims[4]))        #(total clusters/file,cells/cluster,features)


    print("X shape: ",X_new.shape) 
    print("Y shape: ",Y_new.shape)  

    print('* Processed events: '+str(tot_nEvts))    
    print('* Max number of clusters per event: '+str(max_nPoints))#Dim of largest point cloud
    print('* Max number of cells per cluster: '+str(max_nCells))
    

    
    print(" ************* FILLING ************* " )

    #===================#
    ## FILL IN ENTRIES ##==============================================================
    #===================#
    t0 = t.time()    
    clustindx = 0  # Index of the outtermost dimesion of X corresponding to clusters 
    # Loop over events
    for i in range(max_dims[0]):
        # pull all relevant indices
        evt = event_indices[i,0]
        evt_totclust = event_indices[i,1] #not anymore a track index
        # recall this now returns an array
        cluster_nums = event_indices[i,2]

        ##############
        ## CLUSTERS ##
        ##############
        if cluster_nums is not None:
            
            for c in cluster_nums:
                
                cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
                cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()            
                cell_indices = find_index_1D(cluster_cell_ID, cell_ID_dict)
                nInClust = len(cluster_cell_ID)
                
                cluster_cell_Eta = geo_dict['cell_geo_eta'][cell_indices]
                cluster_cell_Phi = geo_dict['cell_geo_phi'][cell_indices]
                cluster_cell_rPerp = geo_dict['cell_geo_rPerp'][cell_indices]
                cluster_cell_sampling = geo_dict['cell_geo_sampling'][cell_indices]

                ### input all the data
                X_new[clustindx,:nInClust,0] = np.log(cluster_cell_E)
                ### Normalize to  parent cluster center
                X_new[clustindx,:nInClust,1] = cluster_cell_Eta - event_dict['cluster_Eta'][evt][c] #cluster_cell_Eta - av_Eta
                X_new[clustindx,:nInClust,2] = cluster_cell_Phi - event_dict['cluster_Phi'][evt][c] #cluster_cell_Phi - av_Phi
                X_new[clustindx,:nInClust,3] = cluster_cell_sampling * 0.1
                X_new[clustindx,:nInClust,4] = cluster_cell_rPerp 
                
                
                ###########################
                ## Classification labels ##
                ###########################
                #if Label>=0:
                Y_new[clustindx] = Label

                # if i<=1:
                #     print("----------------------------------------------------  event ",i)
                #     print(" evt index: ", evt)                
                #     print(" totclust : ", evt_totclust)       
                #     print(" --- cluster : ", c) 
                #     print(" ---clustindx : ", clustindx) 
                #     #print("  Dimension check:")
                #     #print("  - nInClust (# Cells in cluster)", nInClust)
                #     #print("  - len(cluster_cell_E) : ", len(cluster_cell_E) )       #Dilia
                #     #print("  - shape X_new[clustindx,:nInClust,0]", X_new[clustindx,:nInClust,0].shape)      
                #     print()
                #     print("Last X cell entry, ",nInClust-1 )
                #     print(X_new[clustindx,nInClust-1])
                #     print()
                #     print("Current Y target")
                #     print(Y_new)
                #     print()    
                    
                clustindx = clustindx +1
                
        #if i<=1:
        #    print("-------------------- Loop over clusters finished for event ",i) 
        #    print(" - Current clustindx : ", clustindx)
                

    print("------------------------------------- Loop over ALL events finished") 
    print(" - Final clustindx : ", clustindx)
    print("Input shape: X_new.shape: ",X_new.shape)
    print("Taget shape: Y_new.shape: ",Y_new.shape)
    #print()
    #print("Y target")
    #print(Y_new)   
                
    
    #####################################################
    t1 = t.time()
    array_construction_time = t1 - t0

    #=======================#
    ## ARRAY CONCATENATION ##
    #=======================#
    t0 = t.time()


    if clustindx != max_dims[2]:
        print("WARNING!!  The max index of clusters is different than the max dimension for clusters")
    if len(X_new) != len(Y_new):
        print("WARNING!! the dimensions for X and Y are different in this file!!!")

    print("----  ARRAY CONCATENATION  ----")
    print("--- Total number of clusters in this file : ", max_dims[2])  
    print("--- max number of cells per cluster for this file: ", max_dims[3])
    print("----> MAX number of cells per cluster for ALL files: ", max_nCells)
    print('----> Total of processed events: '+str(tot_nEvts))   
    print('----> Total number of processed clusters : '+str(tot_nCluster))

    
    # #  Write to X
    old_tot = tot_nCluster - max_dims[2] #lower index for current file 
    X_large[old_tot:tot_nCluster, :max_dims[3], :6] = np.ndarray.copy(X_new)

    # # pad the remainder with zeros (just to be sure)
    fill_shape = (tot_nCluster - old_tot, MAX_cell_dim - max_dims[3], 6)
    X_large[old_tot:tot_nCluster, max_dims[3]:MAX_cell_dim, :] = np.zeros(fill_shape)
     
    # Write to Y
    Y_new_categorical = to_categorical( Y_new )
    Y_large[old_tot:tot_nCluster,:] = np.ndarray.copy(Y_new_categorical)

    # Total Time
    t1 = t.time()
    time_to_memmap = t1-t0
    thisfile_t_tot = events_cuts_time+find_create_max_dims_time+indices_time\
          +array_construction_time+time_to_memmap
    t_tot += thisfile_t_tot
    
    #print("old_tot: ", old_tot) #Dilia

    ##########################

    print('Max dimensions: '+str(max_dims))
    print('Time to create dicts and select events: '+str(events_cuts_time))
    print('Time to construct index array: '+str(indices_time))
    print('Time to find dimensions and make new array: '+str(find_create_max_dims_time))
    print('Time to populate elements: '+str(array_construction_time))

    print('Time to copy to memory map: '+str(time_to_memmap))
    print('Time for this file: '+str(thisfile_t_tot))
    print('Total time: '+str(t_tot))
    print()
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nCluster, max_nCells, 6)))
    print()

t0 = t.time()
X = np.lib.format.open_memmap('/data/atlas/dportill/X_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nCluster, max_nCells, 6))
np.copyto(dst=X, src=X_large[:tot_nCluster,:max_nCells,:], casting='same_kind', where=True)
del X_large
os.system('rm /data/atlas/dportill/X_large.npy')


Y = np.lib.format.open_memmap('/data/atlas/dportill/Y_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nCluster,2))
np.copyto(dst=Y, src=Y_large[:tot_nCluster,:], casting='same_kind', where=True)
del Y_large
os.system('rm /data/atlas/dportill/Y_large.npy')

t1 = t.time()
print()
print('Time to copy new and delete old: '+str(t1-t0)+' (s)')
print()

