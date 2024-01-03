from os import link
from re import T
import sys
import numpy as np
import scipy as sp

def read_vtk(fileName, scale_cell=1, verbose=False, select_seg=None):
    """ Read VTK file for dislocation data
    
        Only support cubic cell for now
        rn: position of the nodes               (nNodes, 3)
        links: connectivity of the segments     (nLinks, 2)
        b: Burgers vector of the segments       (nLinks, 3)
        n: slip plane normal of the segments    (nLinks, 3)

    Parameters
    ----------
    fileName : str
        Name of the VTK file
    scale_cell : float, optional
        Scaling factor for the coordinates
    verbose : bool, optional
        Print verbose information
    select_seg : ndarray, optional
        Select segments to read

    Returns
    -------
    rn : ndarray
        Coordinates of the nodes
    links : ndarray
        Connectivity of the nodes
    cell : ndarray
        Cell dimensions
    endpoints : ndarray
        Coordinates of the endpoints
    """
    with open(fileName, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('POINTS'):
                nNodes = int(line.split()[1])
                rn = np.zeros((nNodes, 3))
                for i in range(nNodes):
                    rn[i, :] = np.array(f.readline().split(), dtype=float)
                rn *= scale_cell
                nNodes = nNodes - 8
                origin = np.min(rn[:8, :], axis=0)
                rn -= origin
                endpoints = rn[:8, :]
                rn = rn[8:, :]
            elif line.startswith('CELLS'):
                nLinks = int(line.split()[1])
                links = np.zeros((nLinks, 2))
                for i in range(nLinks):
                    line = f.readline()
                    if i == 0:
                        edges = np.array(line.split()[1:], dtype=int)
                    else:
                        links[i, :] = np.array(line.split()[1:3], dtype=int)
                nLinks = nLinks - 1
                links = links[1:, :] - 8
            elif line.startswith('VECTORS'):
                line = f.readline()
                b = np.zeros((nLinks, 3))
                n = np.zeros((nLinks, 3))
                for i in range(nLinks):
                    b[i, :] = np.array(f.readline().split(), dtype=float)
                    t = rn[links[i, 1].astype(int), :] - rn[links[i, 0].astype(int), :]
                    n[i, :] = np.cross(b[i, :], t)
                    n[i, :] = n[i, :]/np.linalg.norm(n[i, :])
                    if verbose:
                        print(b[i, :], t/np.linalg.norm(t), n[i, :])
                b = b/np.linalg.norm(b, axis=1, keepdims=True)
                break

    L = np.max(endpoints, axis=0) - np.min(endpoints, axis=0)
    cell = np.diag(L)
    if verbose:
        print('cell')
        print(cell)
        print('links, b, n')
        print(links.shape, b.shape, n.shape)
    links = np.hstack([links, b, n])
    if select_seg is not None:
        links = links[select_seg, ...]
    rn = rn - L/2

    return rn, links, cell

default_ca_header = '''CA_FILE_VERSION 6
CA_LIB_VERSION 0.0.0
STRUCTURE_TYPES 5
STRUCTURE_TYPE 1
NAME fcc
FULL_NAME FCC
COLOR 0.400000006 1.0 0.400000006
TYPE LATTICE
BURGERS_VECTOR_FAMILIES 6
BURGERS_VECTOR_FAMILY ID 0
Other
0.0 0.0 0.0
0.9 0.2 0.2
BURGERS_VECTOR_FAMILY ID 1
1/2<110> (Perfect)
0.5 0.5 0.0
0.2 0.2 1.0
BURGERS_VECTOR_FAMILY ID 2
1/6<112> (Shockley)
0.1666666716 0.1666666716 0.3333333433
0.0 1.0 0.0
BURGERS_VECTOR_FAMILY ID 3
1/6<110> (Stair-rod)
0.1666666716 0.1666666716 0.0
1.0 0.0 1.0
BURGERS_VECTOR_FAMILY ID 4
1/3<100> (Hirth)
0.3333333433 0.0 0.0
1.0 1.0 0.0
BURGERS_VECTOR_FAMILY ID 5
1/3<111> (Frank)
0.3333333433 0.3333333433 0.3333333433
0.0 1.0 1.0
END_STRUCTURE_TYPE
STRUCTURE_TYPE 2
NAME hcp
FULL_NAME HCP
COLOR 1.0 0.400000006 0.400000006
TYPE LATTICE
BURGERS_VECTOR_FAMILIES 6
BURGERS_VECTOR_FAMILY ID 0
Other
0.0 0.0 0.0
0.9 0.2 0.2
BURGERS_VECTOR_FAMILY ID 1
1/3<1-210>
0.7071067691 0.0 0.0
0.0 1.0 0.0
BURGERS_VECTOR_FAMILY ID 2
<0001>
0.0 0.0 1.1547005177
0.200000003 0.200000003 1.0
BURGERS_VECTOR_FAMILY ID 3
<1-100>
0.0 1.224744916 0.0
1.0 0.0 1.0
BURGERS_VECTOR_FAMILY ID 4
1/3<1-100>
0.0 0.4082483053 0.0
1.0 0.5 0.0
BURGERS_VECTOR_FAMILY ID 5
1/3<1-213>
0.7071067691 0.0 1.1547005177
1.0 1.0 0.0
END_STRUCTURE_TYPE
STRUCTURE_TYPE 3
NAME bcc
FULL_NAME BCC
COLOR 0.400000006 0.400000006 1.0
TYPE LATTICE
BURGERS_VECTOR_FAMILIES 4
BURGERS_VECTOR_FAMILY ID 0
Other
0.0 0.0 0.0
0.9 0.2 0.2
BURGERS_VECTOR_FAMILY ID 1
1/2<111>
0.5 0.5 0.5
0.0 1.0 0.0
BURGERS_VECTOR_FAMILY ID 2
<100>
1.0 0.0 0.0
1.0 0.3000000119 0.8000000119
BURGERS_VECTOR_FAMILY ID 3
<110>
1.0 1.0 0.0
0.200000003 0.5 1.0
END_STRUCTURE_TYPE
STRUCTURE_TYPE 4
NAME diamond
FULL_NAME Cubic diamond
COLOR 0.0745098069 0.6274510026 0.9960784316
TYPE LATTICE
BURGERS_VECTOR_FAMILIES 5
BURGERS_VECTOR_FAMILY ID 0
Other
0.0 0.0 0.0
0.9 0.2 0.2
BURGERS_VECTOR_FAMILY ID 1
1/2<110>
0.5 0.5 0.0
0.200000003 0.200000003 1.0
BURGERS_VECTOR_FAMILY ID 2
1/6<112>
0.1666666716 0.1666666716 0.3333333433
0.0 1.0 0.0
BURGERS_VECTOR_FAMILY ID 3
1/6<110>
0.1666666716 0.1666666716 0.0
1.0 0.0 1.0
BURGERS_VECTOR_FAMILY ID 4
1/3<111>
0.3333333433 0.3333333433 0.3333333433
0.0 1.0 1.0
END_STRUCTURE_TYPE
STRUCTURE_TYPE 5
NAME hex_diamond
FULL_NAME Hexagonal diamond
COLOR 0.9960784316 0.5372549295 0.0
TYPE LATTICE
BURGERS_VECTOR_FAMILIES 5
BURGERS_VECTOR_FAMILY ID 0
Other
0.0 0.0 0.0
0.9 0.2 0.2
BURGERS_VECTOR_FAMILY ID 1
1/3<1-210>
0.7071067691 0.0 0.0
0.0 1.0 0.0
BURGERS_VECTOR_FAMILY ID 2
<0001>
0.0 0.0 1.1547005177
0.2 0.2 1.0
BURGERS_VECTOR_FAMILY ID 3
<1-100>
0.0 1.224744916 0.0
1.0 0.0 1.0
BURGERS_VECTOR_FAMILY ID 4
1/3<1-100>
0.0 0.4082483053 0.0
1.0 0.5 0.0
END_STRUCTURE_TYPE'''

simulation_cell_header = '''SIMULATION_CELL_ORIGIN %.10f %.10f %.10f
SIMULATION_CELL_MATRIX
%.10f %.10f %.10f
%.10f %.10f %.10f
%.10f %.10f %.10f
PBC_FLAGS 1 1 1'''

cluster_header = '''CLUSTERS 1
CLUSTER 1
CLUSTER_STRUCTURE 4
CLUSTER_ORIENTATION
%.10f %.10f %.10f
%.10f %.10f %.10f
%.10f %.10f %.10f
CLUSTER_COLOR 1.0 1.0 1.0
CLUSTER_SIZE 33452184
END_CLUSTER'''

def group_segments(filename, rn, links, cell, origin=(0, 0, 0), bmag=1, pbc=False):
    ''' Group connected segments into physical dislocation links '''
    cluster_orientation = bmag * np.identity(3) * 1e10
    rn = rn.copy() * bmag * 1e10
    cell = cell.copy() * bmag * 1e10
    if pbc:
        rn_scaled = np.linalg.inv(cell).dot(rn.T).T
        rn_scaled[np.isclose(rn_scaled, 0.5)] = -0.5
        # rn_scaled[np.isclose(rn_scaled, -0.5)] = -0.4
        rn = cell.dot(rn_scaled.T).T
    origin = tuple(np.multiply(origin, bmag*1e10))
    # find all the repeated nodes
    rn, rn_idx = np.unique(rn, axis=0, return_inverse=True)
    links = links.copy()
    links[:, 0:2] = rn_idx[links[:, 0:2].astype(int)]
    # construct a sparse connectivity matrix (sparse matrix)
    conn = sp.sparse.lil_array((rn.shape[0], rn.shape[0]), dtype=int)
    for i in range(links.shape[0]):
        conn[links[i, 0].astype(int), links[i, 1].astype(int)] = i + 1
        conn[links[i, 1].astype(int), links[i, 0].astype(int)] = i + 1
        # print(links[i])
    conn = conn.tocsr()
    # print(conn)
    disl_used = np.zeros(links.shape[0], dtype=bool)
    disl_list = []
    for i in range(links.shape[0]):
        if disl_used[i]:
            continue
        disl_used[i] = True
        disl_node = [links[i, 0].astype(int), links[i, 1].astype(int)]
        # print(disl_node[0], conn.getrow(disl_node[0]))
        # print(disl_node[-1], conn.getrow(disl_node[-1]))
        # extend the dislocation left
        while True:
            conn_row = conn.getrow(disl_node[0])
            if conn_row.getnnz() != 2:
                break
            link_exist = conn[disl_node[0], disl_node[1]] - 1
            link_left = conn_row.sum() - (link_exist + 1) - 1
            if disl_used[link_left]:
                break
            # print(link_exist, links[link_exist, :2])
            # print(link_left, links[link_left, :2])
            burgers_exist = links[link_exist, 2:5]
            burgers_left = links[link_left, 2:5]
            if np.linalg.norm(burgers_exist - burgers_left) > 1e-6:
                break
            conn_row_nonzero = conn_row.nonzero()[1]
            link_left_to = conn_row_nonzero.sum() - disl_node[1]
            disl_used[link_left] = True
            disl_node.insert(0, link_left_to)
        # extend the dislocation right
        while True:
            conn_row = conn.getrow(disl_node[-1])
            if conn_row.getnnz() != 2:
                break
            link_exist = conn[disl_node[-1], disl_node[-2]] - 1
            link_right = conn_row.sum() - (link_exist + 1) - 1
            if disl_used[link_right]:
                break
            burgers_exist = links[link_exist, 2:5]
            burgers_right = links[link_right, 2:5]
            if np.linalg.norm(burgers_exist - burgers_right) > 1e-6:
                break
            conn_row_nonzero = conn_row.nonzero()[1]
            link_right_to = conn_row_nonzero.sum() - disl_node[-2]
            disl_used[link_right] = True
            disl_node.append(link_right_to)
        disl_list.append(disl_node)

    with open(filename, 'w') as f:
        print(default_ca_header, file=f)
        # cell_str = np.savetxt(sys.stdout.buffer, cell, '%f')
        print(simulation_cell_header%(origin+tuple(cell.flatten())), file=f)
        # cluster_str = np.savetxt(sys.stdout.buffer, cluster_orientation, '%f')
        print(cluster_header%tuple(cluster_orientation.flatten()), file=f)
        ndisl = len(disl_list)
        print('DISLOCATIONS %d'%ndisl, file=f)
        for i, disl_i in enumerate(disl_list):
            print('%d'%i, file=f) # Dislocation ID
            bvec = links[conn[disl_i[0], disl_i[1]] - 1, 2:5]
            bvec = bvec/np.abs(bvec)[np.abs(bvec) > 0].min()
            burgers = bvec/np.sum(bvec**2)
            print('%.10f %.10f %.10f'%tuple(burgers), file=f)               # Burger's vector
            print('%d'%1, file=f) # Cluster ID
            print('%d'%len(disl_i), file=f) # number of nodes
            for j in disl_i:
                print('%.10f %.10f %.10f'%tuple(rn[j, :]), file=f)
    return rn, links, cell, disl_list

def write_ca(filename, rn, links, cell, origin=(0, 0, 0), bmag=1):
    """ Write Crystal Analysis file
    """
    ndisl = links.shape[0]
    cluster_orientation = bmag * np.identity(3) * 1e10
    rn = rn.copy() * bmag * 1e10
    cell = cell.copy() * bmag * 1e10
    origin = tuple(np.array(origin)*bmag*1e10)
    with open(filename, 'w') as f:
        print(default_ca_header, file=f)
        # cell_str = np.savetxt(sys.stdout.buffer, cell, '%f')
        print(simulation_cell_header%(origin+tuple(cell.flatten())), file=f)
        # cluster_str = np.savetxt(sys.stdout.buffer, cluster_orientation, '%f')
        print(cluster_header%tuple(cluster_orientation.flatten()), file=f)
        print('DISLOCATIONS %d'%ndisl, file=f)
        for i in range(ndisl):
            print('%d'%i, file=f) # Dislocation ID
            bvec = links[i, 2:5]
            bvec = bvec/np.abs(bvec)[np.abs(bvec) > 0].min()
            burgers = bvec/np.sum(bvec**2)
            print('%.10f %.10f %.10f'%tuple(burgers), file=f)               # Burger's vector
            print('%d'%1, file=f) # Cluster ID
            print('%d'%2, file=f) # number of nodes
            print('%.10f %.10f %.10f'%tuple(rn[links[i, 0].astype(int), :]), file=f)
            print('%.10f %.10f %.10f'%tuple(rn[links[i, 1].astype(int), :]), file=f)
    return rn, links, cell

def write_xyz(fileName, r, props=False, scale=1, ParticleTypes=None):
    """ Write XYZ file
    """
    nAtoms = r.shape[0]
    values = r.copy()*scale
    if props is not None:
        values = np.hstack([values, props])
    if ParticleTypes is None:
        ParticleTypes = 'O'
    np.savetxt(fileName, values, header='%d\nObservation points'%nAtoms, comments='')

    # with open(fileName, 'w') as f:
    #     print(nAtoms, file=f)
    #     print('Observation points', file=f)
    #     for i in range(nAtoms):
    #         if ParticleTypes is None:
    #             print('O %.10f %.10f %.10f'%tuple(r[i, :]), file=f)
    #         else:
    #             print('%d %.10f %.10f %.10f'%(ParticleTypes[i], r[i, 0], r[i, 1], r[i, 2]), file=f)
