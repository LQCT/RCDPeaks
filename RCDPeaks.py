#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:54:23 2021

  Author: Daniel Platero Rochart
  Contact: dplatero97@gmail.com
"""

# =============================================================================
# Imports
# =============================================================================

import sys
import heapq
import argparse
from collections import OrderedDict
import pickle

import numpy as np
import mdtraj as md
from bitarray import bitarray as ba

# =============================================================================
# Functions
# =============================================================================

def parse_arguments():
    '''
    DESCRIPTION
    Parse all user arguments from the command line.

    Return:
        user_inputs (parser.argparse): namespace with user input arguments.
    '''

    # Initializing argparse ---------------------------------------------------
    desc = '\nFDPeaks: DP clustering of long MD trajectories'
    parser = argparse.ArgumentParser(description=desc,
                                     add_help=True,
                                     epilog='As simple as that ;)')
    # Arguments: loading trajectory -------------------------------------------
    parser.add_argument('-top', dest='topology', action='store',
                        help='path to topology file (psf/pdb)',
                        type=str, required=False)
    parser.add_argument('-traj', dest='trajectory', action='store',
                        help='path to trajectory file',
                        type=str)
    parser.add_argument('-first', dest='first',  action='store',
                        help='first frame to analyze (starting from 0)',
                        type=int, required=False, default=0)
    parser.add_argument('-last', dest='last', action='store',
                        help='last frame to analyze (starting from 0)',
                        type=int, required=False, default=None)
    parser.add_argument('-stride', dest='stride', action='store',
                        help='stride of frames to analyze',
                        type=int, required=False, default=1)
    parser.add_argument('-sel', dest='selection', action='store',
                        help='atom selection (MDTraj syntax)',
                        type=str, required=False, default='all')
    parser.add_argument('-rmwat', dest='remove_waters', action='store',
                        help='remove waters from trajectory?',
                        type=bool, required=False, default=0,
                        choices=[True, False])
    # Arguments: clustering ---------------------------------------------------
    parser.add_argument('-cutoff', action='store', dest='cutoff',
                        help='RMSD cutoff for pairwise comparisons in A',
                        type=float, required=False, default=1.0)
    parser.add_argument('-distc', action='store', dest='distance_cut',
                        help='Distance cutoff in the desition graph in A',
                        type=float, required=False)
    parser.add_argument('-rhoc', action='store', dest='density_cut',
                        help='Density cutoff in the desition graph',
                        type=float, required=False)
    parser.add_argument('-ref', action='store', dest='reference',
                        help='reference frame to align trajectory',
                        type=int, required=False, default=0)
    parser.add_argument('-k', action='store', dest='knn', required=False,
                        type=int, default=150)
    # Arguments: analysis -----------------------------------------------------
    parser.add_argument('-odir', action='store', dest='outdir',
                        help='output directory to store analysis',
                        type=str, required=False, default='./')
    user_inputs = parser.parse_args()
    return user_inputs


def load_trajectory(args):
    '''
    DESCRIPTION
    Loads trajectory file using MDTraj. If trajectory format is h5, lh5 or
    pdb, topology file is not required. Otherwise, you should specify a
    topology file.

    Arguments:
        args (argparse.Namespace): user input parameters parsed by argparse.
    Return:
        trajectory (mdtraj.Trajectory): trajectory object for further analysis.
    '''

    traj_file = args.trajectory
    traj_ext = traj_file.split('.')[-1]
    # Does trajectory file format need topology ? -----------------------------
    if traj_ext in ['h5', 'lh5', 'pdb']:
        traj = md.load(traj_file)
    else:
        traj = md.load(traj_file, top=args.topology)

    # Reduce RAM consumption by loading selected atoms only -------------------
    if args.selection != 'all':
        try:
            sel_indx = traj.topology.select(args.selection)
        except ValueError:
            print('Specified selection is invalid')
            sys.exit()
        if sel_indx.size == 0:
            print('Specified selection in your system corresponds to no atoms')
            sys.exit()
        traj = traj.atom_slice(sel_indx)[args.first:args.last:args.stride]
    else:
        traj = traj[args.first:args.last:args.stride]

    # Center coordinates of loaded trajectory ---------------------------------
    traj.center_coordinates()
    return traj


def to_VMD(output_name, array, stride, first_frame, trajectory):
    '''
    '''
    with open(output_name, 'wt') as clq:
        for num in np.unique(array):
            clq.write('{}:\n'.format(num))
            numcluster = np.where(array == num)[0]
            frames = [str((x * stride) + first_frame) for x in numcluster]
            members = ' '.join(frames)
            clq.write('Members: ' + members + '\n\n')


def pickle_to_file(data, file_name):
    ''' Serialize data using **pickle**.

    Args:
        data (object)  : any serializable object.
        file_name (str): name of the **pickle** file to be created.
    Returns:
        (str): file_name
    '''
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    return file_name


def get_node_info(node, traj, k, cutoff):
    """
    Parameters
    ----------
    node : int
        Node to analyze.
    traj : mdtraj.Trajectory
        Trajectory to analyze.
    k : int
        Number of nearest neighbors to calculate.
    cutoff : numpy.array
        RMSD cutoff to calculate rho

    Returns
    -------
    node_info : tuple
        Tuple containing the necessary node information:
            node_info[0]: CoreDistance(node) (inverted for a "max heap")
            node_info[1]: node index
            node_info[2]: rmsd vector of node vs. all
            node_info[3]: iterator of the rmsd knn of node
    """
    # Get RMSD(node), Kd(node) and knn sorted partition -----------------------
    node_rmsd = md.rmsd(traj, traj, node, precentered=True)
    rms_minors = node_rmsd < cutoff
    node_rmsd_part = np.argpartition(node_rmsd, k)[:k + 1]
    node_rmsd_knn = np.nditer(
        node_rmsd_part[node_rmsd[node_rmsd_part[:k - 1]].argsort()], order='C')
    rho = np.count_nonzero(rms_minors)
    next(node_rmsd_knn)
    # Get CoreDistance(A) as Kd -----------------------------------------------
    node_info = (rho, node, node_rmsd, node_rmsd_knn)
    return node_info

# @profile
def fdpeaks(traj, cutoff, density_cut, distance_cut, k):

    N = traj.n_frames

    rho_arr = np.zeros(N, dtype=np.int32)
    delta_arr = np.zeros(N, dtype=np.float32)
    nnhd = np.zeros(N, dtype=np.int32)
    core = np.zeros(N, dtype=np.int32)


    pool = []
    exhausted = []
    not_visited = set(range(N))
    Ases = []
    matrix = OrderedDict()

    # 1. Find node 'A' whose neighborhood will be exhausted -------------------
    while True:
        # get ( Kd(A), A, RMSD(A), and the sorted knn(A) partition ) ----------
        try:
            A_rho, A, A_rms, A_knn = heapq.heappop(pool)
        # if pool is empty, check for a random not-yet-visited ----------------
        except IndexError:
            try:
                A = not_visited.pop()
                (A_rho, A, A_rms, A_knn) = get_node_info(A, traj, k, cutoff)
                rho_arr[A] = A_rho
            # if all nodes visited, break & check exhausted heap --------------
            except KeyError:
                break

        # 2. Exhaust knn of A searching for a node 'B' for which: Kd(A) > Kd(B)
        while True:
            try:
                # consume the knn(A) iterator (in rmsd ordering) --------------
                B = int(next(A_knn))
                Ases.append(A)
            except StopIteration:
                # if knn(A) exhausted, send A to exhausted heap then break ----
                heapq.heappush(exhausted, (A_rho, A, A_rms))
                break

            if B in not_visited:
                (B_rho, B, B_rms, B_knn) = get_node_info(B, traj, k, cutoff)
                # update_matrix(matrix, B_minors, B)
                heapq.heappush(pool, (B_rho, B, B_rms, B_knn))
                not_visited.remove(B)
                rho_arr[B] = B_rho
            else:
                B_rho = rho_arr[B]

            # cases where Kd(A) > Kd(B) before exhaustion ---------------------
            if (B_rho > A_rho):
                nnhd[A] = B
                delta_arr[A] = A_rms[B]
                break

    # 3. Analizing the Exhaust Heap -------------------------------------------
    while True:
        try:
            ex_rho, ex_node, ex_rms = heapq.heappop(exhausted)
        except IndexError:
            break
        lesser_rho = (rho_arr <= ex_rho)
        hdn_distances = np.copy(ex_rms)
        hdn_distances[lesser_rho] =np.inf
        if hdn_distances.min() == np.inf:
            delta_arr[ex_node] = ex_rms.max()
            nnhd[ex_node] = ex_node
        else:
            nnhd[ex_node] = hdn_distances.argmin()
            delta_arr[ex_node] = hdn_distances.min()

    # 4. Constructing Clusters and cluster's core -----------------------------
    pickle_to_file(nnhd, (args.outdir + 'nnhd.pickle'))
    pickle_to_file(rho_arr, (args.outdir + 'rho.pickle'))
    pickle_to_file(delta_arr, (args.outdir + 'delta.pickle'))
    # Choosing Clusters Centers -----------------------------------------------
    delta_arr_c = np.greater_equal(delta_arr,
                                   np.fromiter([distance_cut / 10] * N,
                                               dtype=np.float32))
    rho_arr_c = np.greater_equal(rho_arr,
                                 np.fromiter([density_cut] * N,
                                             dtype=np.float32))
    cluster_centers = np.where((delta_arr_c & rho_arr_c))[0]
    nnhd[cluster_centers] = -1

    # Recomputing cluster's center --------------------------------------------
    for center in cluster_centers:
        node_rmsd = md.rmsd(traj, traj, center, precentered=True)
        rms_minors = node_rmsd < cutoff
        bitarr = ba()
        bitarr.pack(rms_minors.tobytes())
        matrix.update({center: bitarr})

    # Asigning each node to its cluster and merging clusters with near centers
    clusters_array = np.zeros(N, dtype=np.int32)
    c_count = 0
    visited_centers = set()
    for center in cluster_centers:
        equal = False
        for visited in visited_centers:
            if (rho_arr[center] == rho_arr[visited]) and (matrix[center][visited]):
                node_neigh = set()
                node_neigh.add(center)
                clusters_array[center] = clusters_array[visited]
                while True:
                    cluster_members = []
                    for n in node_neigh:
                        cluster_members.extend((nnhd == n).nonzero()[0])
                    if len(cluster_members) == 0:
                        break
                    clusters_array[cluster_members] = clusters_array[visited]
                    node_neigh = set.difference(set(cluster_members),
                                                node_neigh)
                equal = True
                break
        if equal:
            continue
        visited_centers.add(center)
        c_count += 1
        node_neigh = set()
        node_neigh.add(center)
        clusters_array[center] = c_count
        while True:
            cluster_members = []
            for i in node_neigh:
                cluster_members.extend((nnhd == i).nonzero()[0])
            if len(cluster_members) == 0:
                break
            clusters_array[cluster_members] = c_count
            node_neigh = set.difference(set(cluster_members), node_neigh)



    # Finfing cluster's core using Daura's algorithm --------------------------

    centers_heap = []
    for center in visited_centers:
        heapq.heappush(centers_heap, (rho_arr[center], center))

    while True:
        try:
            center_rho, center = heapq.heappop(centers_heap)
        except IndexError:
            break
        core[np.frombuffer(matrix[center].unpack(),
                       dtype=np.bool)] = clusters_array[center]

    return clusters_array, core, cluster_centers


# =============================================================================
# Main algorithm
# =============================================================================

# loc = '/home/platero/Documentos/LQCT/Trajs/'
# args = argparse.Namespace(topology=loc + 'aligned_tau.pdb',
#                           trajectory=loc + 'aligned_original_tau_6K.dcd',
#                           first=0, last=None, stride=1,
#                           selection='all', cutoff=2.5, outdir='./',
#                           distance_cut=3,
#                           density_cut=110)

args = parse_arguments()

traj = load_trajectory(args)
cutoff = np.full(traj.n_frames, args.cutoff / 10, dtype=np.float32)

cluster_array, core, centers = fdpeaks(traj, cutoff, args.density_cut,
                              args.distance_cut, args.knn)

pickle_to_file(cluster_array, (args.outdir + 'clusters.pickle'))
pickle_to_file(core, (args.outdir + 'core.pickle'))
pickle_to_file(centers, (args.outdir + 'centers.pickle'))
# to_VMD(args.outdir + 'Core.out', core, args.stride, args.first,
#        args.trajectory)
# to_VMD(args.outdir + 'Cluster.out', cluster_array, args.stride,
#        args.first, args.trajectory)
