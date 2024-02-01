
import numpy as np
import tqdm as tqdm

from numba import njit
from scipy.optimize import linear_sum_assignment
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def get_deltas_list_origami(particles,pixel_size,radius):
    deltas_list = []
    msd_per_part = []
    num_part = 0
    for particle in particles: 
            part_msd = []
            num_part+=1
            particle = np.array(particle)
            for delta_t in range(1,len(particle)):
                    for t in range(0,len(particle)-delta_t):
                        part_msd.append(particle[t+delta_t][0:3]-particle[t][0:3])
            p_msd = np.array(part_msd)

            r2 = np.sqrt(np.power(p_msd[:,1]-p_msd[1,1],2) + np.power(p_msd[:,2] - p_msd[1,2],2))
         
        

            if np.max( r2)>1/(0.108):

                deltas_list = deltas_list+ part_msd #.append(particle[t+delta_t][0:3]-particle[t][0:3])
    return np.array(deltas_list)

def get_deltas_list(particles):
    deltas_list = []

    for particle in particles: 
        part_msd = []


        for delta_t in range(1,len(particle)):
                for t in range(0,len(particle)-delta_t):
                    part_msd.append(particle[t+delta_t][0:3]-particle[t][0:3])
        

        
        deltas_list = deltas_list+ part_msd 
    return np.array(deltas_list)



def track_particles(xy_locs,start_frame, end_frame,mem_frame,max_distance):
        particles =[[p] for p in list(xy_locs[np.where(xy_locs[:,0]==1)[0],:])]

        for frame in tqdm.tqdm(range(start_frame+1,end_frame)):
            
                
            new_particles = []
            loc = xy_locs[np.where(xy_locs[:,0]==frame)[0],:]
            ids = []
            distance_matrix = []
           
            for id, particle in enumerate(particles):
                if np.abs(particle[-1][0] - (frame-1)) <=mem_frame :
                    distance_matrix.append(np.sqrt(np.sum(np.power(loc[:,1:3]-particle[-1][1:3],2),axis=1)))
                    ids.append(id)
                
            assigned_xy = []

            if len(distance_matrix)>0:
                distance_matrix = np.array(distance_matrix)
                ######## < max distance => >= max_distance should be set to 10000 ######
                distance_matrix[distance_matrix>=max_distance] = 10000

                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                assigned_xy = []
                for row_i in range(0,len(row_ind)):
                    if distance_matrix[row_ind[row_i],col_ind[row_i]]< 10000 :
                        particles[ids[row_ind[row_i]]].append(loc[col_ind[row_i],:])
                        assigned_xy.append(col_ind[row_i])

                
            for c in range(0,loc.shape[0]):
                if c not in assigned_xy:
                    new_particles.append([loc[c,:]]) 
                            


            particles+=new_particles

    
        return particles




@njit
def disambiguate_couples(particles, min_frames, max_distance):
    coupled_ids = []
    couples = []
    for p_1 , _ in enumerate(particles):
        part_1 = particles[p_1]    
        for p_2, _  in enumerate(particles):
                    

                    if p_1!=p_2 and p_2 not in coupled_ids and particles[p_2][0,-1]!=part_1[0,-1]:
                        part_2 = particles[p_2]
                        distances_x  = np.zeros(1)
                        distances_y  = np.zeros(1)

                        for t in part_1[:,0]:
                            ind_t1 = np.where(part_1[:,0]==t)[0]
                            if t in part_2[:,0]:
                                ind_t2 = np.where(part_2[:,0]==t)[0]


                                distances_x = np.hstack((distances_x,(part_1[ind_t1,1]-part_2[ind_t2,1])**2))
                                distances_y = np.hstack((distances_y,(part_1[ind_t1,2]-part_2[ind_t2,2])**2))
                               


                        distances_x = distances_x[1:]
                        distances_y = distances_y[1:]
                        distances = np.sqrt(distances_x + distances_y)
  
                            # if np.abs(np.mean(distances))<1.0: 
                        if len(np.where(distances<=max_distance)[0])>=min_frames: 
                                    
                                    coupled_ids.append(p_2 )
                                    couples.append([p_1,p_2])
                       


                        

    return  coupled_ids, couples

def find_multilobes(particles, min_frames, max_distance):
    coupled_ids, couples = disambiguate_couples(particles, min_frames, max_distance)

    for c_1 , _ in enumerate(couples):
        for c_2 , _ in enumerate(couples):
            if c_1!=c_2:
                if len(np.intersect1d(couples[c_1],couples[c_2]))!=0:
                    couples[c_1] = np.union1d(couples[c_1], couples[c_2])
                    couples[c_2] = []


    couples = [x  for x in couples if len(x)!=0]
    couple_colors = []
    for y in couples:
        col = np.unique( [particles[x][0,-1] for  x in y])       
        couple_colors.append(col)
    ####################################################################



    #Calculate number of red-green , red-orange and orange-green couples#
    aggregate_ids = []
    # 0 is red, 1 is green, 2 is orange #
    for id, c_c in enumerate(couple_colors):
        if 0 in c_c and 1 in c_c: 
            aggregate_ids.append(0)
        if 0 in c_c and 2 in c_c and 1 not in c_c: 
            aggregate_ids.append(1)    
        if 1 in c_c and 2 in c_c and 0 not in c_c: 
            aggregate_ids.append(2)
    #####################################################################


    ################# Get all the single lobes and remove multilobes ####
    single_particle = [particle for id, particle in enumerate(particles) if id not in coupled_ids] 
    #####################################################################
    ### Combine multiple single lobe tracks into 1 multilobe track ####
    colored_particles = [] #np.empty((0,5))
    for id, couple in enumerate(couples):
        p = np.vstack([particles[c] for c in couple])
        u_vals, _ = np.unique(p[:,0],return_index=True)

        combined_p = []
        for u in u_vals:
            combined_p.append(np.mean(p[ np.where(p[:,0]==u)[0],:],axis=0))

        p = np.hstack([np.array(combined_p), np.ones([len(combined_p),1])*aggregate_ids[id]])
        colored_particles.append(p)
       
    ####################################################################




    return  single_particle, colored_particles 