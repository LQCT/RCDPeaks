#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Platero Rochart
@contact: dplatero97@gmail.com
"""

import heapq
import argparse
from collections import OrderedDict
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import mdtraj as md
from bitarray import bitarray as ba


def parse_arguments():
    """
    Parse all user arguments from the command line.

    Returns
    -------
        user_inputs (parser.argparse): namespace with user input arguments.
    """
    # Initializing argparse ---------------------------------------------------
    desc = '\nRCDPeaks: An efficient implementation of Density Peaks \
        Clustering Algorithm for long Molecular Dynamics'
    parser = argparse.ArgumentParser(prog='rcdpeaks',
                                     description=desc,
                                     add_help=True,
                                     epilog='As simple as that ;)',
                                     allow_abbrev=False,
                                     usage='%(prog)s -traj trajectory [options]')
    # Arguments: loading trajectory -------------------------------------------
    traj = parser.add_argument_group(title='Trajectory options')
    traj.add_argument('-traj', dest='trajectory', action='store',
                      help='Path to trajectory file (pdb/dcd) \
                      [default: %(default)s]', type=str,
                      metavar='trajectory', required=True)
    traj.add_argument('-top', dest='topology', action='store',
                      help='Path to the topology file (psf/pdb)', type=str,
                      required=False, metavar='topology', default=None)
    traj.add_argument('-first', dest='first',  action='store',
                      help='First frame to analyze (start counting from 0)\
                      [default: %(default)s]', type=int, required=False,
                      default=0, metavar='first_frame')
    traj.add_argument('-last', dest='last', action='store',
                      help='Last frame to analyze (start counting from 0)\
                      [default: last frame]', type=int, required=False,
                      default=None, metavar='last_frame')
    traj.add_argument('-stride', dest='stride', action='store',
                      help='Stride of frames to analyze\
                      [default: %(default)s]', type=int, required=False,
                      default=1, metavar='stride')
    traj.add_argument('-sel', dest='selection', action='store',
                      help='Atom selection (MDTraj syntax)\
                      [default: %(default)s]', type=str, required=False,
                      default='all', metavar='selection')
    # Arguments: clustering parameters ----------------------------------------
    clust = parser.add_argument_group(title='Clustering options')
    clust.add_argument('-cutoff', action='store', dest='cutoff',
                       help='RMSD cutoff for pairwise comparison\
                       [default: %(default)s]',
                       type=float, required=False, default=1, metavar='cutoff')
    clust.add_argument('-dc', action='store', dest='distance_cut',
                       help='Distance threshold for the desition graph',
                       type=float, required=False, default=None,
                       metavar='distance_threshold')
    clust.add_argument('-rc', action='store', dest='density_cut',
                       help='Density threshold for the desition graph',
                       type=float, required=False, default=None,
                       metavar='density_threshold')
    clust.add_argument('-lnn', action='store', dest='load_nnhd',
                       help='Pickle file of nearest neighbors of high density',
                       type=str, required=False, default='None',
                       metavar='nearest_neaighbors')
    clust.add_argument('-lrho', action='store', dest='load_rho',
                       help='Pickle file ofdensity of each node',
                       type=str, required=False, default='None',
                       metavar='load_density')
    clust.add_argument('-ldelta', action='store', dest='load_delta',
                       help='Pickle file of distance of each node',
                       type=str, required=False, default='None',
                       metavar='load_distance')
    # Arguments: analysis -----------------------------------------------------
    out = parser.add_argument_group(title='Output options')
    out.add_argument('-odir', action='store', dest='outdir',
                     help='Output directory to store analysis\
                     [default: %(default)s]',
                     type=str, required=False, default='./', metavar='.')
    user_inputs = parser.parse_args()
    return user_inputs


def load_trajectory(args):
    """
    Load trajectory file using MDTraj. If trajectory format is h5, lh5 or
    pdb, a topology file is not required. Otherwise, you should specify a
    topology file.

    Parameters
    ----------
    args : argparse.Namespace
        user input parameters parsed by argparse (CLI).

    Returns
    -------
    traj : mdtraj.Trajectory
        MDTraj trajectory object.
    """
    # Catching bad extensions of topology and trajectories --------------------
    traj_file = args.trajectory
    traj_ext = traj_file.split('.')[-1]
    if args.topology:
        top_file = args.topology
        top_ext = top_file.split('.')[-1]
        if top_ext not in valid_tops:
            raise ValueError('The specified topology format "{}" is not \
available. Valid formats for topology objects are: {}'.format(top_ext,
                                                              valid_tops))
    if traj_ext not in valid_trajs:
        raise ValueError('The specified trajectory format "{}" is not \
available. Valid trajectory formats for topology objects are: {}'.format(
                                                                  traj_ext,
                                                                  valid_trajs))
    # Does the trajectory file format need topology ? -------------------------
    if traj_ext in ['h5', 'lh5', 'pdb']:
        traj = md.load(traj_file)
    else:
        traj = md.load(traj_file, top=args.topology)

    # Validating ranges from raw loaded traj ----------------------------------
    N = traj.n_frames
    if args.first not in range(0, N - 1):
        raise ValueError('"first" parameter should be in the interval [{},{}]'
                         .format(0, N))
    if (args.last is not None) and (args.last not in (range(args.first + 1, N))):
        raise ValueError('"last" parameter should be in the interval [{},{}]'
                         .format(args.first, N))
    try:
        delta = args.last - args.first
    except TypeError:
        delta = N - args.first
    if args.stride not in range(1, delta):
        raise ValueError('"stride" parameter should be in the interval [{},{}]'
                         .format(1, delta))
    # Reduce RAM consumption by loading selected atoms only -------------------
    if args.selection != 'all':
        try:
            sel_indx = traj.topology.select(args.selection)
        except Exception:
            raise ValueError('Specified selection is invalid')
        if sel_indx.size == 0:
            raise ValueError('Specified selection corresponds to no atoms')
        traj = traj.atom_slice(sel_indx)[args.first:args.last:args.stride]
    else:
        traj = traj[args.first:args.last:args.stride]
    traj.center_coordinates()
    return traj


valid_tops = set(['pdb', 'pdb.gz', 'h5', 'lh5', 'prmtop', 'parm7', 'prm7',
                  'psf', 'mol2', 'hoomdxml', 'gro', 'arc', 'hdf5', 'gsd'])
valid_trajs = set(['arc', 'dcd', 'binpos', 'xtc', 'trr', 'hdf5', 'h5', 'ncdf',
                   'netcdf', 'nc', 'pdb.gz', 'pdb', 'lh5', 'crd', 'mdcrd',
                   'inpcrd', 'restrt', 'rst7', 'ncrst', 'lammpstrj', 'dtr',
                   'stk', 'gro', 'xyz.gz', 'xyz', 'tng', 'xml', 'mol2',
                   'hoomdxml', 'gsd'])


def to_VMD(output_name, array, stride, first_frame):
    '''
    Write a .out file to be open by VMD as NMR cluster
    '''
    with open(output_name, 'wt') as clq:
        for num in np.unique(array):
            clq.write('{}:\n'.format(num))
            numcluster = np.where(array == num)[0]
            frames = [str((x * stride) + first_frame + 1) for x in numcluster]
            members = ' '.join(frames)
            clq.write('Members: ' + members + '\n\n')


def pickle_to_file(data, file_name):
    """
    Serialize data using **pickle**.

    Parameters
    ----------
        data : object
            Any serializable object.
        file_name : str
            Name of the **pickle** file to be created.
    """
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    return file_name


def unpickle_from_file(file_name):
    """
    Load data of a **pickle** file

    Parameters
    ----------
    file_name : str
        name of the **pickle** file

    Returns
    -------
    data : numpy.array
        Array containgn the information of the **pickle** file
    """
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


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
    rho = np.count_nonzero(rms_minors)

    node_rmsd_part = np.argpartition(node_rmsd, k)[:k + 1]
    argsort_indx = node_rmsd[node_rmsd_part].argsort()
    ordered_indx = node_rmsd_part[argsort_indx]
    node_knn = zip(ordered_indx, node_rmsd[ordered_indx])
    next(node_knn)
    return (rho, node, node_knn)


def rcdpeaks_data_info(traj, cutoff):
    """
    Compute the density and distance for the desition graph and find the
    nearest neighbor of high density of each node.

    Parameters
    ----------
    traj : MDTraj.Trajectory
        Trajectory object to analyse
    cutoff : numpy.array
        Array of the size of trajectory filled with the RMSD cutoff for
        pairwise comparison.

    Returns
    -------
    nnhd : numpy.array
        Array containing the nearest neighbor of high density (value) of each
        node (index).
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of high density.
    """
    N = traj.n_frames
    rho_arr = np.zeros(N, dtype=np.int32)
    delta_arr = np.zeros(N, dtype=np.float32)
    nnhd = np.zeros(N, dtype=np.int32)
    pool = []
    exhausted = []
    not_visited = set(range(N))
    k = round(N * 0.02)

    # 1. Find node 'A' whose neighborhood will be exhausted -------------------
    while True:
        # get ( Kd(A), A, RMSD(A), and the sorted knn(A) partition ) ----------
        try:
            A_rho, A, A_knn = heapq.heappop(pool)
        # if pool is empty, check for a random not-yet-visited ----------------
        except IndexError:
            try:
                A = not_visited.pop()
                (A_rho, A, A_knn) = get_node_info(A, traj, k, cutoff)
                rho_arr[A] = A_rho
            # if all nodes visited, break & check exhausted heap --------------
            except KeyError:
                break

        # 2. Exhaust knn of A searching for a node 'B' for which: Kd(A) > Kd(B)
        while True:
            try:
                # consume the knn(A) iterator (in rmsd ordering) --------------
                B, B_rmsd = next(A_knn)
            except StopIteration:
                # if knn(A) exhausted, send A to exhausted heap then break ----
                heapq.heappush(exhausted, (A_rho, A))
                break

            if B in not_visited:
                (B_rho, B, B_knn) = get_node_info(B, traj, k, cutoff)
                heapq.heappush(pool, (B_rho, B, B_knn))
                not_visited.remove(B)
                rho_arr[B] = B_rho
            else:
                B_rho = rho_arr[B]

            # cases where Kd(A) > Kd(B) before exhaustion ---------------------
            if (B_rho > A_rho):
                nnhd[A] = B
                delta_arr[A] = B_rmsd
                break

    # 3. Analizing the Exhaust Heap -------------------------------------------
    while True:
        try:
            ex_rho, ex_node = heapq.heappop(exhausted)
        except IndexError:
            break
        ex_rms = md.rmsd(traj, traj, ex_node, precentered=True)
        lesser_rho = (rho_arr <= ex_rho)
        hdn_distances = np.copy(ex_rms)
        hdn_distances[lesser_rho] =np.inf
        if hdn_distances.min() == np.inf:
            delta_arr[ex_node] = ex_rms.max()
            nnhd[ex_node] = ex_node
        else:
            nnhd[ex_node] = hdn_distances.argmin()
            delta_arr[ex_node] = hdn_distances.min()
    return nnhd, rho_arr, delta_arr


def rcdpeaks_clusters(nnhd, rho_arr, delta_arr, density_cut, distance_cut,
                      traj, cutoff):
    """
    Construct the clusters based on the Density Peaks clustering algorithm.
    Find a core region based on Daura's algorithm.

    Parameters
    ----------
    nnhd : numpy.array
        Array containing the nearest neighbor of high density (value) of each
        node (index).
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of high density.
    density_cut : float
        Density threshold for the desition graphs.
    distance_cut : float
        Distance threshold for the desition graphs.
    traj : MDTraj.Trajectory
        Trajectory object to analyse.
    cutoff : numpy.array
        Array of the size of trajectory filled with the RMSD cutoff for
        pairwise comparison.

    Returns
    -------
    clusters_array : numpy.array
        Array containing the cluster number (value) of each node (index).
    core : numpy.array
        Array containing the core number (value) of each node (index). The
        nodes with value 0 does not belong to any core.
    """
    N = traj.n_frames
    matrix = OrderedDict()
    core = np.zeros(N, dtype=np.int32)

    delta_arr_c = np.greater_equal(delta_arr,
                                   np.fromiter([distance_cut / 10] * N,
                                               dtype=np.float32))
    rho_arr_c = np.greater_equal(rho_arr,
                                 np.fromiter([density_cut] * N,
                                             dtype=np.float32))
    cluster_centers = np.where((delta_arr_c & rho_arr_c))[0]
    nnhd[cluster_centers] = -1
    centers_bits = ba()
    centers_bits.pack((delta_arr_c & rho_arr_c).tobytes())

    # Asigning each node to its cluster and merging clusters with near centers
    centers_heap = []
    for center in cluster_centers:
        heapq.heappush(centers_heap, (rho_arr[center], center))

    clusters_array = np.zeros(N, dtype=np.int32)
    c_count = 0
    visited_centers = set()
    while True:
        try:
            rho, center = heapq.heappop(centers_heap)
        except IndexError:
            break
        equal = False
        for visited in visited_centers:
            if (rho == rho_arr[visited]) and (center in matrix[visited]):
                print('>>Merging center {} with {}'.format(center, visited))
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
        print('>>Clustering center {}'.format(center))
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

        cluster_bits = ba()
        cluster_bits.pack((clusters_array == c_count).tobytes())

        node_rmsd = md.rmsd(traj, traj, center, precentered=True)
        rms_minors = node_rmsd < cutoff
        node_bits = ba()
        node_bits.pack(rms_minors.tobytes())

        matrix.update({center: (node_bits & centers_bits).search(ba('1'))})
        core[np.frombuffer(node_bits.unpack(), dtype=np.bool)] = c_count
    for i in range(1, c_count + 1):
        print('>>Cluster {}: {} frames and {} in core'.format(i,
               np.count_nonzero(clusters_array == i),
               np.count_nonzero(core == i)))

    return clusters_array, core


def desition_graph(outdir, density_array, distance_array):
    """
    Save the desition graph

    Parameters
    ----------
        density_array : numpy.array
           Array with the density of each frame.
        distance_array : numpy.array
            Array with the distance of each frame to its nearest neighbor of
           high density
    """
    mpl.rc('figure', autolayout=True, figsize=[3.33, 2.5], dpi=300)
    mpl.rc('font', family='STIXGeneral')
    mpl.rc('lines', markersize=2)
    mpl.rc('mathtext', fontset='stix')

    mpl.rc('axes', titlesize=10, linewidth=1)
    mpl.rcParams['axes.labelweight'] = 'bold'

    mpl.rc('xtick', labelsize=12, direction='out', top=False)
    mpl.rc('xtick.major', top=False)
    mpl.rc('xtick.minor', top=False)

    mpl.rc('ytick', labelsize=12, direction='out', right=False)
    mpl.rc('ytick.major', right=False)
    mpl.rc('ytick.minor', right=False)

    plt.xlabel('Density', fontsize=14)
    plt.ylabel('Distance (A)', fontsize=14)
    plt.scatter(density_array, distance_array*10, marker='+')
    plt.savefig((outdir + 'desition_graph'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Load user arguments ------------------------------------------------
    args = parse_arguments()
    sms = '\n\n ATTENTION !!! No trajectory passed.Run with -h for help.'
    assert args.trajectory, sms

    # Load trajectory
    print('\n\nLoading trajectory')
    trajectory = load_trajectory(args)
    cutoff = np.full(trajectory.n_frames, args.cutoff / 10, dtype=np.float32)

    # Load data from previous analysis
    if (args.load_nnhd, args.load_rho, args.load_delta) != ('None', 'None',
       'None'):
        print('\nNearest neighbors, density and distance provided. Loading' +
              ' information from pickle files')
        nnhd = unpickle_from_file(args.load_nnhd)
        rho_arr = unpickle_from_file(args.load_rho)
        delta_arr = unpickle_from_file(args.load_delta)

    # Calculating trajectory data
    else:
        print('\nComputing Nearest Neighbors, Densities and Distances.')
        nnhd, rho_arr, delta_arr = rcdpeaks_data_info(trajectory, cutoff)
        print('\nSaving desition graph')
        desition_graph(args.outdir, rho_arr, delta_arr)
        print('\nSaving trajectory information in pickle flies')
        pickle_to_file(nnhd, (args.outdir + 'nnhd.pickle'))
        pickle_to_file(rho_arr, (args.outdir + 'density.pickle'))
        pickle_to_file(delta_arr, (args.outdir + 'distance.pickle'))


    no_distance = '\n\nNo distance threshold provided. Exiting'
    assert args.distance_cut, no_distance
    no_density = '\n\nNo density threshold provided Exiting.'
    assert args.density_cut, no_density

    print('\nClustering...')
    clusters, core = rcdpeaks_clusters(nnhd, rho_arr, delta_arr,
                                       args.density_cut, args.distance_cut,
                                       trajectory, cutoff)

    # Write files to VMD
    print('\nWriting files for VMD')
    to_VMD(args.outdir + 'Core.out', core, args.stride, args.first)
    to_VMD(args.outdir + 'Cluster.out', clusters, args.stride, args.first)

    print('Normal termination')
