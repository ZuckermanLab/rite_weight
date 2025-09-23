import numpy as np
from scipy import sparse

def get_Tmatrix(weights, trans_from_to):
    # This is the body of the function
    n_cluster = (np.max(trans_from_to))+1
    flux_matrix = sparse.coo_matrix((weights,(trans_from_to[:,0],trans_from_to[:,1])), shape=(n_cluster,n_cluster)).toarray()
    fluxes_out = np.sum(flux_matrix, 1)
    transitionMatrix = flux_matrix.copy()
    for state_idx in range(n_cluster):
        transitionMatrix[state_idx, :] = (flux_matrix[state_idx, :] / fluxes_out[state_idx])
    return transitionMatrix 




def get_steady_state(Tmatrix_solve, max_iters_matrix_power=100):
    #### algebric pSS start
    # Getting algebraic_pss from eigenvalue solver
    eigenvalues, eigenvectors = np.linalg.eig(np.transpose(Tmatrix_solve))
    pSS = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    pSS = pSS.squeeze()
    assert not np.isclose(np.sum(pSS), 0), "EigenValue solver probability distribution sums to 0!"
    pSS = pSS / np.sum(pSS)
    
    #Solving with matrix power
    if sum(pSS < 0) > 0 and max_iters_matrix_power > 0:
        pSS_last = pSS
        tmatrix_power = Tmatrix_solve.copy()
        for N in range(max_iters_matrix_power):
            pSS_new = tmatrix_power.T @ pSS_last
            num_negative_elements = sum(pSS_new < 0)
            if num_negative_elements == 0:
                break
            pSS_last = pSS_new
            tmatrix_power = np.matmul(Tmatrix_solve, tmatrix_power)
        pSS = pSS_new
    
    algebraic_pSS = pSS
    return algebraic_pSS
    

def get_target_flux(Trans_matrix, algebraic_pSS):
    
    #Setting variable for target flux calculation
    Transition_to_target = np.full((1,len(Trans_matrix)),np.nan)
    Transition_to_target[0,:] = Trans_matrix[:,-1]
    algebraic_flux = (np.matmul(Transition_to_target,algebraic_pSS))
    
    return algebraic_flux
    
    
    

    
