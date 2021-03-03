#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Daniel Platero Rochart & Roy González-Alemán
@contact: [daniel.platero, roy_gonzalez]@fq.uh.cu
"""

import os
import heapq
import argparse
import pickle
from os.path import join
from itertools import count
from collections import Counter

import numpy as np
import pandas as pd
import mdtraj as md
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


valid_tops = set(['pdb', 'pdb.gz', 'h5', 'lh5', 'prmtop', 'parm7', 'prm7',
                  'psf', 'mol2', 'hoomdxml', 'gro', 'arc', 'hdf5', 'gsd'])
valid_trajs = set(['arc', 'dcd', 'binpos', 'xtc', 'trr', 'hdf5', 'h5', 'ncdf',
                   'netcdf', 'nc', 'pdb.gz', 'pdb', 'lh5', 'crd', 'mdcrd',
                   'inpcrd', 'restrt', 'rst7', 'ncrst', 'lammpstrj', 'dtr',
                   'stk', 'gro', 'xyz.gz', 'xyz', 'tng', 'xml', 'mol2',
                   'hoomdxml', 'gsd'])


def parse_arguments():
    """
    Parse user arguments from the command line interface.

    Returns
    -------
        user_inputs (parser.argparse): namespace with user input arguments.
    """
    # Initializing argparse ---------------------------------------------------
    desc = ('\nRCDPeaks: Memory-Efficient Density Peaks Clustering of '
            'long Molecular Dynamics')
    usage = '%(prog)s -traj trajectory [options]'
    parser = argparse.ArgumentParser(prog='rcdpeaks',
                                     description=desc,
                                     add_help=True,
                                     epilog='As simple as that ;)',
                                     allow_abbrev=False,
                                     usage=usage)
    # Arguments: loading trajectory -------------------------------------------
    traj = parser.add_argument_group(title='Trajectory options')
    traj.add_argument('-traj', dest='trajectory', action='store',
                      help='Path to the trajectory file \
                      [default: %(default)s]', type=str,
                      metavar='trajectory', required=True, default=None)
    traj.add_argument('-top', dest='topology', action='store',
                      help='Path to the topology file', type=str,
                      required=False, metavar='topology', default=None)
    traj.add_argument('-first', dest='first', action='store',
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
                       help='RMSD cutoff for pairwise comparison in A\
                       [default: %(default)s]',
                       type=float, required=False, default=1, metavar='cutoff')
    clust.add_argument('-dcut', action='store', dest='distance_cut',
                       help='delta cutoff for the decision graph',
                       type=float, required=False, default=None,
                       metavar='delta_cut')
    clust.add_argument('-rcut', action='store', dest='density_cut',
                       help='rho cutoff for the decision graph',
                       type=float, required=False, default=None,
                       metavar='rho_cut')
    clust.add_argument('-restart_from', action='store', dest='restart',
                       help='restart clustering from previous job',
                       type=str, required=False, default=None,
                       metavar='file.pickle')
    clust.add_argument('-auto_centers', action='store', dest='automatic',
                       type=str, required=False, default='True',
                       metavar='bool', choices=['True', 'False'])

    # Arguments: analysis -----------------------------------------------------
    out = parser.add_argument_group(title='Output options')
    out.add_argument('-odir', action='store', dest='outdir',
                     help='Output directory to store analysis\
                     [default: %(default)s]',
                     type=str, required=False, default='./', metavar='.')
    args = parser.parse_args()
    return args


def is_valid_traj(traj, valid_trajs):
    """
    Check if the trajectory extension is supported by MDTraj.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.
    valid_trajs : set
        Set of supported trajectory extensions.

    Raises
    ------
    ValueError
        If trajectory extension is not supported by MDTraj.

    Returns
    -------
    bool
        True if trajectory extension is supported.

    """
    traj_ext = os.path.basename(traj).split('.')[-1]
    if traj_ext not in valid_trajs:
        raise ValueError('\n\n>>> Arguments Inconsistency\nThe trajectory'
                         ' extension "{}" '.format(traj_ext)
                         + 'is not available. Options'
                         ' are: {}'.format(valid_tops))

    return True


def traj_needs_top(traj):
    """
    Determine if the trajectory extension needs topological information.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.

    Returns
    -------
    bool
        True if trajectory needs topological information.

    """
    traj_ext = os.path.basename(traj).split('.')[-1]
    if traj_ext in ['h5', 'lh5', 'pdb']:
        return False
    return True


def is_valid_top(topology, valid_tops):
    """
    Check if the topology extension is supported by MDTraj.

    Parameters
    ----------
    topology : str
        Path to the topology file.
    valid_tops : set
        Set of supported topology extensions.

    Raises
    ------
    ValueError
        If topology extension is not supported by MDTraj.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    top_ext = os.path.basename(topology).split('.')[-1]

    if top_ext not in valid_tops:
        raise ValueError('\n\n>>> Arguments Inconsistency\nThe topology'
                         ' extension "{}" '.format(top_ext)
                         + 'is not available. Options'
                         ' are: {}'.format(valid_tops))
    return True


def load_raw_traj(traj, valid_trajs, topology=None):
    """
    Load the whole trajectory without any modifications.

    Parameters
    ----------
    traj : str
        Path to the trajectory file.
    valid_trajs : set
        Set of supported trajectory extensions.
    topology : str, optional
        Path to the trajectory file. The default is None.

    Returns
    -------
    mdtraj.Trajectory
        Raw trajectory.

    """
    if traj_needs_top(traj) and not topology:
        traj_ext = traj.split('.')[-1]
        raise ValueError('\n\n>>> Arguments Inconsistency\nYou should pass'
                         ' the -top argument for this trajectory extension'
                         ' ({}).'.format(traj_ext))

    if is_valid_traj(traj, valid_trajs) and traj_needs_top(traj):
        if is_valid_top(topology, valid_tops):
            return md.load(traj, top=topology)

    if is_valid_traj(traj, valid_trajs) and not traj_needs_top(traj):
        return md.load(traj)


def shrink_traj_selection(traj, selection):
    """
    Select a subset of atoms from the trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory object to which selection will be applied.
    selection : str
        Any MDTraj valid selection.

    Raises
    ------
    ValueError
        If specified selection is not valid.
        If specified selection corresponds to no atoms.

    Returns
    -------
    traj : mdtraj.Trajectory
        Trajectory containing the subset of specified atoms.

    """
    if selection != 'all':
        try:
            sel_indx = traj.topology.select(selection)
        except Exception:
            raise ValueError('\n\n>>> Arguments Inconsistency\nSpecified'
                             ' selection "{}"'.format(selection)
                             + ' is invalid in MDTraj.')

        if sel_indx.size == 0:
            raise ValueError('\n\n>>> Arguments Inconsistency\nSpecified'
                             ' selection "{}"'.format(selection)
                             + ' corresponds to no atoms.')
        traj = traj.atom_slice(sel_indx, inplace=True)
    return traj


def shrink_traj_range(first, last, stride, traj):
    """
    Select a subset of frames from the trajectory.

    Parameters
    ----------
    first : int
        First frame to consider (0-based indexing).
    last : int
        Last frame to consider (0-based indexing).
    stride : int
        Stride (step).
    traj : mdtraj.Trajectory
        Trajectory object to which slicing will be applied.

    Raises
    ------
    ValueError
        If first, last or stride are falling out of their valid ranges.

    Returns
    -------
    mdtraj.Trajectory
        Trajectory containing the subset of specified frames.

    """
    # Calculate range of available intervals ----------------------------------
    N = traj.n_frames
    first_range = range(0, N - 1)
    last_range = range(first + 1, N)
    try:
        delta = last - first
    except TypeError:
        delta = N - first
    stride_range = range(1, delta)
    # Raising if violations ---------------------------------------------------
    if first not in first_range:
        raise ValueError('\n\n>>> Arguments Inconsistency\n-first argument'
                         ' must be in the interval [{}, {}).'
                         .format(first_range.start, first_range.stop))
    if last and (last not in last_range):
        raise ValueError('\n\n>>> Arguments Inconsistency\n-last argument'
                         ' must be in the interval [{},{}).'
                         .format(last_range.start, last_range.stop))
    if stride not in stride_range:
        raise ValueError('\n\n>>> Arguments Inconsistency\n-stride argument'
                         ' must be in the interval [{},{}).'
                         .format(stride_range.start, stride_range.stop))
    # Slicing trajectory ------------------------------------------------------
    sliced = slice(first, last, stride)
    if sliced not in [slice(0, N, 1), slice(0, None, 1)]:
        return traj.slice(sliced)
    return traj


def unpickle_from_file(file_name):
    """
    Load data from a pickle file.

    Parameters
    ----------
    file_name : str
        Path to the **pickle** file.

    Returns
    -------
    data : numpy.array
        Array containing the information of the **pickle** file.
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
        Precentered trajectory to analyze.
    k : int
        Number of nearest neighbors to calculate.
    cutoff : numpy.array
        Array of lenght equals to the trajectory size filled with the cutoff
        value to calculate rho.

    Returns
    -------
    node_info : tuple
        Tuple containing the following node information:
            node_info[0]: CoreDistance(node) (negative value for a "max heap")
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


def compute_tree(traj, cutoff, N):
    """
    Compute the oriented Density Peaks Tree, where each node is pointing to its
    nearest neighbor of higher density.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Precentered trajectory to analyze.
    cutoff : numpy.array
        Array of lenght equals to the trajectory size filled with the cutoff
        value to calculate rho.

    Returns
    -------
    nnhd_arr : numpy.array
        Array containing the nearest neighbor of higher density (value) of each
        node (index).
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of higher density.
    """
    rho_arr = np.zeros(N, dtype=np.int32)
    delta_arr = np.zeros(N, dtype=np.float32)
    nnhd_arr = np.zeros(N, dtype=np.int32)
    pool = []
    exhausted = []
    not_visited = set(range(N))
    k = round(N * 0.02)
    # 1. Find node 'A' whose neighborhood will be exhausted -------------------
    while True:
        # get ( Kd(A), A, RMSD(A), and the sorted knn(A) partition ) ----------
        try:
            A_rho, A, A_knn = heapq.heappop(pool)
        # if pool is empty, check for a random not-yet-visited node -----------
        except IndexError:
            try:
                A = not_visited.pop()
                (A_rho, A, A_knn) = get_node_info(A, traj, k, cutoff)
                rho_arr[A] = A_rho
            # if all nodes visited, break & check the exhausted heap ----------
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
            if B_rho > A_rho:
                nnhd_arr[A] = B
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
        hdn_distances[lesser_rho] = np.inf
        if hdn_distances.min() == np.inf:
            delta_arr[ex_node] = ex_rms.max()
            nnhd_arr[ex_node] = ex_node
        else:
            nnhd_arr[ex_node] = hdn_distances.argmin()
            delta_arr[ex_node] = hdn_distances.min()
    return nnhd_arr, rho_arr, delta_arr


def autodetect_centers(delta_arr, rho_arr):
    """
    Autodetect clusters centers in the decision graph via iterative rounds of
    the "last gap method" described by Flores and Garza in
    10.1016/j.knosys.2020.106350.

    Parameters
    ----------
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of higher density.
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).

    Returns
    -------
    nodes_by_level : list
        List containing the array of selected centers for each iteration.

    """
    # Average magnitudes ------------------------------------------------------
    delta_ave = delta_arr.mean()
    rho_ave = rho_arr.mean()
    # flores-garza candidates ordered by gamma --------------------------------
    candidates = np.where((delta_arr > delta_ave) & (rho_arr > rho_ave))[0]
    gamma_arr = delta_arr * rho_arr
    gamma_candidates = gamma_arr[candidates]
    order = (-gamma_candidates).argsort()
    gamma_ordered = gamma_candidates[order]
    # flores-garza distances and retrieval by last gap procedure --------------
    d_i = abs(np.diff(gamma_ordered))
    d_ave = d_i.mean()
    nnodes = 10
    nodes_by_level = []
    while nnodes > 1:
        center_gaps = order[np.where(d_i > d_ave)[0]]
        nnodes = center_gaps.size
        last_center_gap = center_gaps[-1]
        last_center_idx = order.tolist().index(last_center_gap)
        nodes = order[:last_center_idx + 1]
        nodes_by_level.append(candidates[nodes])
        d_ave = d_i[np.where(d_i > d_ave)[0]].mean()
    return nodes_by_level


def merge_centers(nodes_by_level, traj, cutoff):
    """
    Merge centers via "leader" algorithm to retrieve an orthogonal set.

    Parameters
    ----------
    nodes_by_level : list
        List containing the array of centers selected from the Decision Graph.
    traj : mdtraj.Trajectory
        Precentered trajectory to analyze.
    cutoff : float
        distance cutoff value for merging centers.

    Returns
    -------
    merged_by_levels : list
        List containing the array of merged centers.
    neighborhoods_by_level : list
        List containing the array of merged centers neighborhoods.

    """
    merged_by_levels = []
    neighborhoods_by_level = []
    for level in nodes_by_level:
        if level.size == 1:
            last = level[0]
            rms = md.rmsd(traj, traj, last, precentered=True)
            neighbors = np.where(rms <= cutoff)[0]
            neighborhood = np.zeros(traj.n_frames, dtype=np.int32)
            neighborhood[neighbors] = last
            neighborhoods_by_level.append(neighborhood)
        neighborhood = np.zeros(traj.n_frames, dtype=np.int32)
        discard = set()
        true_centers = []
        for center in level:
            if center not in discard:
                rms = md.rmsd(traj, traj, center, precentered=True)
                neighbors = np.where(rms <= cutoff)[0]
                neighborhood[neighbors] = center
                discard.update(np.intersect1d(level, neighbors))
                true_centers.append(center)
        merged_by_levels.append(np.fromiter(true_centers, np.int32))
        neighborhoods_by_level.append(neighborhood)
    return merged_by_levels, neighborhoods_by_level


def dp_assign_clusters(nnhd_arr, merged_by_levels):
    """
    Get an exact Density Peaks assignment of cluster labels as originally
    described by Rodríguez and Laio in 10.1126/science.1242072.

    Parameters
    ----------
    nnhd_arr : numpy.array
        Array containing the nearest neighbor of higher density (value) of each
        node (index).
    merged_by_levels : list
        List containing the array(s) of merged centers.

    Returns
    -------
    dp_clusters_arrays_by_level : list
        List containing the array(s) of assigned labels.
    """
    dp_clusters_arrays_by_level = []
    for ccenters in merged_by_levels:
        if ccenters.size == 1:
            dp_clusters_arrays_by_level.append(np.ones(nnhd_arr.size,
                                                       dtype=np.int32))
        else:
            nnhd = nnhd_arr.copy()
            nnhd[ccenters] = -1
            clusters = []
            cluster_arr = np.zeros(nnhd_arr.size, dtype=np.int32)
            ids = count(1)
            for center in ccenters:
                to_explore = set()
                to_explore.update(np.where(nnhd == center)[0])
                cx = [center]
                sizes = []
                while True:
                    try:
                        current = to_explore.pop()
                    except KeyError:
                        break
                    cx.append(current)
                    sizes.append(len(to_explore))
                    to_explore.update(np.where(nnhd == current)[0])
                clusters.append(cx)
                cluster_arr[cx] = next(ids)
            dp_clusters_arrays_by_level.append(cluster_arr)
    return dp_clusters_arrays_by_level


def refine_dp_assignment(merged_by_level, neighborhoods_by_level,
                         dp_clusters_arrays_by_level):
    """
    Refine the exact Density Peaks assignment of cluster labels by restricting
    clusters radius.

    Parameters
    ----------
    merged_by_levels : list
        List containing the array of merged centers.
    neighborhoods_by_level : list
        List containing the array of merged centers neighborhoods.
    dp_clusters_arrays_by_level : list
        List containing the array(s) of assigned labels.

    Returns
    -------
    refined_by_level : list
        List of array(s) containing the refined clusters.
    """
    refined_by_level = []
    for i, ccenters in enumerate(merged_by_level):
        if ccenters.size == 1:
            last = neighborhoods_by_level[-1]
            last[last.nonzero()[0]] = 1
            refined_by_level.append(last)
        else:
            ids = count(1)
            refined_arr = np.zeros(neighborhoods_by_level[0].size, dtype=np.int32)
            for idx, center in enumerate(ccenters):
                ID = next(ids)
                neighbors = np.where(neighborhoods_by_level[i] == center)[0]
                dp_assign = np.where(dp_clusters_arrays_by_level[i] == ID)[0]
                refined = np.intersect1d(neighbors, dp_assign)
                refined_arr[refined] = ID
            refined_by_level.append(refined_arr)
    return refined_by_level


def pickle_to_file(data, file_name):
    """
    Serialize data using pickle.

    Parameters
    ----------
    data : object
         Any serializable object.
    file_name : str
        Name of the pickle file to be created.

    Returns
    -------
    file_name : str
        Name of the created pickle file.
    """
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    return file_name


def create_dir_hierarchy(outdir, traj, merged_by_level):
    """
    Create de directory hierarchy to store output results.

    Parameters
    ----------
    outdir : str
        Path where to store outputs.
    traj : str
        Path to the topology file.
    merged_by_levels : list
        List containing the array of merged centers.

    Raises
    ------
    Exception
        Raise an exception if the specified output dir already exists.

    Returns
    -------
    out_dir : str
        Path to the output dir.
    out_subdirs : list
        List of the output subdirs.
    """
    outlabel = 'RCDP-{}'.format(os.path.basename(traj).split('.')[0])
    out_dir = join(outdir, outlabel)
    out_subdirs = []
    for i, n in enumerate(merged_by_level):
        s = n.size
        if s > 1:
            label = 'L{}-{}_centers'.format(i + 1, n.size)
        else:
            label = 'L{}-{}_center'.format(i + 1, n.size)
        outdir_level = join(out_dir, label)
        out_subdirs.append(outdir_level)

    try:
        os.makedirs(out_dir)
        [os.makedirs(x) for x in out_subdirs]
    except FileExistsError:
        raise Exception('{} directory already exists.'.format(out_dir) +
                        ' Please specify another location or rename the'
                        ' existing directory.')
    return out_dir, out_subdirs


def generic_matplotlib():
    """
    Customize the graphs.

    Returns
    -------
    None.
    """
    mpl.rc('figure', figsize=[12, 8], dpi=300)
    mpl.rc('xtick', direction='in', top=True)
    mpl.rc('xtick.major', top=False, )
    mpl.rc('xtick.minor', top=True, visible=True)
    mpl.rc('ytick', direction='in', right=True)
    mpl.rc('ytick.major', right=True, )
    mpl.rc('ytick.minor', right=True, visible=True)

    mpl.rc('axes', labelsize=20)
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    mpl.rc('lines', linewidth=8, color='k')
    mpl.rc('font', family='monospace', size=20)
    mpl.rc('grid', alpha=0.5, color='gray', linewidth=1, linestyle='--')


def output_decision_graph(delta_arr, rho_arr, merged_by_level, out_subdirs):
    """
    Output the decision graph for each level of merged centers.

    Parameters
    ----------
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of higher density.
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).
    merged_by_levels : list
        List containing the array of merged centers.
    out_subdirs : list
        List of the output subdirs.

    Returns
    -------
    None.
    """
    generic_matplotlib()
    delta_ave = delta_arr.mean()
    rho_ave = rho_arr.mean()
    for i, selection in enumerate(merged_by_level):
        plt.axhline(delta_ave, lw=2, color='k', ls='--')
        plt.axvline(rho_ave, lw=2, color='k', ls='--')
        plt.suptitle('Decision Graph', fontsize=20)
        plt.xlabel("$\\rho$", fontsize=20)
        plt.ylabel("$\\delta$", fontsize=20)
        plt.scatter(rho_arr, delta_arr, color='gray', marker='.', alpha=0.5)
        plt.scatter(rho_arr[selection], delta_arr[selection], marker='o',
                    color='k')
        plt.savefig(join(out_subdirs[i], 'DecisionGraph'))
        plt.close()


def output_restart(outdir, first, stride, last, selection, topology,
                   traj, cutoff, nnhd_arr, delta_arr, rho_arr):
    """
    Output all arguments needed for a restart job. This avoids to recompute
    the information related to the exact oriented Density Peaks graph.

    Parameters
    ----------
    outdir : str
        Path where to store outputs.
    first : int
        First frame to consider (0-based indexing).
    stride : int
        Stride (step).
    last : int
        Last frame to consider (0-based indexing).
    selection : str
        Any MDTraj valid selection.
    topology : str
        Path to the topology file.
    traj : str
        Path to the trajectory file.
    cutoff : numpy.array
        Array of lenght equals to the trajectory size filled with the cutoff
        value to calculate rho.
    nnhd_arr : numpy.array
        Array containing the nearest neighbor of higher density (value) of each
        node (index).
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of higher density.
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).
    out_subdirs : list
        List of the output subdirs.

    Returns
    -------
    None.
    """
    info = (first, stride, last, selection, topology, traj, cutoff,
            nnhd_arr, delta_arr, rho_arr)
    pickle_to_file(info, (join(outdir, 'restart.pickle')))


def output_frames_info(N1, first, last, stride, refined_clusters_by_level,
                       subdirs, label):
    """
    Output "frames_stats.txt" containing frameID, and clusterID.

    Parameters
    ----------
    N1 : int
        default value when last == None.
    first : int
        First frame to consider (0-based indexing).
    last : int
        Last frame to consider (0-based indexing).
    stride : int
        Stride (step).
    refined_clusters_by_level : list
        List of numpy arrays of clusters ID.
    subdirs : str
        Path to the output dirs.
    label : str
        Label to name output files

    Returns
    -------
    frames_df : pandas.DataFrame
        dataframe with frames_statistics info.
    """
    start = first
    if not last:
        stop = N1
    else:
        stop = last
    slice_frames = np.arange(start, stop, stride, dtype=np.int32)
    dataframes = []
    out_label = 'frames_stats_{}.txt'.format(label)
    for i, level in enumerate(refined_clusters_by_level):
        frames_df = pd.DataFrame(columns=['frame', 'cluster_id'])
        frames_df['frame'] = range(N1)
        frames_df['cluster_id'].loc[slice_frames] = level
        with open(os.path.join(subdirs[i], out_label), 'wt') as on:
            frames_df.to_string(buf=on, index=False)
        dataframes.append(frames_df)
    return dataframes


def output_clusters_info(N1, first, last, stride, merged_nodes, dp_clusters,
                         ref_clusters, rho_arr, delta_arr, subdirs):
    """
    Output "clusters_stats.txt" containing 'cluster', 'center', 'rho', 'delta',
    'exact_population', and 'refined_population'.

    Parameters
    ----------
    N1 : int
        default value when last == None.
    first : int
        First frame to consider (0-based indexing).
    last : int
        Last frame to consider (0-based indexing).
    stride : int
        Stride (step).
    merged_nodes : list
        List of arrays containing the orthogonal nodes selected.
    dp_clusters : list
        List of arrays containing the exact Density Peaks assigned labels.
    ref_clusters : list
        List of arrays containing the refined Density Peaks assigned labels.
    rho_arr : numpy.array
        Array containing the density (value) of each node (index).
    delta_arr : numpy.array
        Array containing the distance (value) of each node (index) to its
        nearest neighbor of higher density.
    Returns
    -------
    None.

    """
    start = first
    if not last:
        stop = N1
    else:
        stop = last
    slice_frames = np.arange(start, stop, stride, dtype=np.int32)
    for i, centers in enumerate(merged_nodes):
        with open(join(subdirs[i], 'clusters_stats.txt'), 'w') as cl:
            cl.write('{:>12}{:>12}{:>12}{:>12}{:>15}{:>15}\n'
                     .format('cluster', 'center', 'rho', 'delta',
                             'exact_pop', 'refined_pop'))
            for idx, center in enumerate(centers):
                frame_id = center
                cluster_id = idx + 1
                rho_val = rho_arr[frame_id]
                delta_val = delta_arr[frame_id]
                ref_pop = np.where(ref_clusters[i] == idx + 1)[0].size
                xct_pop = np.where(dp_clusters[i] == idx + 1)[0].size
                cl.write('{:>12}{:>12}{:>12.0f}{:>12.2f}{:>15.0f}{:>15.0f}\n'
                         .format(cluster_id, slice_frames[frame_id], rho_val,
                                 delta_val, xct_pop, ref_pop))


def save_centers_pdb(trajectory, merged_by_level, subdirs):
    """
    Save selected clusters centers to a pdb file.

    Parameters
    ----------
    merged_by_levels : list
        List containing the array of merged centers.
    subdirs : str
        Path to the output dirs.

    Returns
    -------
    None.

    """
    for i, centers in enumerate(merged_by_level):
        trajectory[centers].save_pdb(join(subdirs[i], 'Centers.pdb'))


def top_has_coords(topology):
    """
    Check if topology has cartesian coordinates information.

    Parameters
    ----------
    topology : str
        Path to the topology file.

    Returns
    -------
    int
        Number of cartesian frames if topology contains cartesians.
        False otherwise.
    """
    try:
        tt = md.load(topology)
    except OSError:
        return False
    return tt.xyz.shape[0]


def to_VMD(outdir, topology, first, N1, last, stride, final_array):
    """
    Create a .log file for visualization of clusters in VMD through a
    third-party plugin.

    Parameters
    ----------
    outdir : str
        Path where to create the VMD visualization .log.
    topology : str
        Path to the topology file.
    first : int
        First frame to consider (0-based indexing).
    N1 : int
        default value when last == None.
    last : TYPE
        Last frame to consider (0-based indexing).
    stride : TYPE
        Stride (step).
    final_array : numpy.ndarray
        Final labeling of the selected clusters ordered by size (descending).

    Returns
    -------
    logname : str
        Log file to be used with VMD.
    """
    logname = outdir
    vmd_offset = top_has_coords(topology)
    start = first
    if not last:
        stop = N1
    else:
        stop = last
    slice_frames = np.arange(start, stop, stride, dtype=np.int32)
    nmr_offset = 1
    with open(logname, 'wt') as clq:
        for num in np.unique(final_array):
            if num != 0:
                clq.write('{}:\n'.format(num))
                cframes = np.where(final_array == num)[0]
                if vmd_offset:
                    real_frames = slice_frames[cframes] + nmr_offset + vmd_offset
                else:
                    real_frames = slice_frames[cframes] + nmr_offset
                str_frames = [str(x) for x in real_frames]
                members = ' '.join(str_frames)
                clq.write('Members: ' + members + '\n\n')
        if 0 in np.unique(final_array):
            clq.write('{}:\n'.format(0))
            cframes = np.where(final_array == 0)[0]
            if vmd_offset:
                real_frames = slice_frames[cframes] + nmr_offset + vmd_offset
            else:
                real_frames = slice_frames[cframes] + nmr_offset
            str_frames = [str(x) for x in real_frames]
            members = ' '.join(str_frames)
            clq.write('Members: ' + members + '\n\n')
    return logname


def output_transition_matrix(dp_clusters_arrays_by_level, subdirs):
    """
    Output several representations of the clusters transition matrix.

    Parameters
    ----------
    dp_clusters_arrays_by_level : list
        List containing the array(s) of assigned labels.
    subdirs : str
        Path to the output dirs.

    Returns
    -------
    None.

    """
    # Square Graph
    for i, dp_arr in enumerate(dp_clusters_arrays_by_level):
        mpl.rcdefaults()
        generic_matplotlib()
        plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
        tr = [(dp_arr[i], dp_arr[i + 1]) for i, x in enumerate(dp_arr[:-1])]
        counter = dict(Counter(tr))
        maxi = dp_arr.max()
        raw_matrix = np.zeros((maxi, maxi), dtype=np.int)
        for x in counter:
            A, B = x
            C = counter[x]
            raw_matrix[A - 1, B - 1] = C
        clusters_pop = raw_matrix.sum(axis=1)
        percent_matrix = raw_matrix / clusters_pop.reshape((maxi, 1)) * 100
        plt.matshow(percent_matrix, cmap='nipy_spectral', origin='upper', vmin=0,
                    vmax=100)
        interval = range(0, maxi)
        plt.xticks(interval, labels=[str(x + 1) for x in interval],
                   fontsize=14, rotation=45)
        plt.yticks(interval, labels=[str(x + 1) for x in interval], fontsize=14)
        plt.xlabel('To Cluster Y', fontweight='bold')
        plt.ylabel('From Cluster X', fontweight='bold')
        plt.grid(which='major', axis='both', color='white', alpha=0.5, lw=1)
        plt.colorbar(orientation='horizontal', label='% of Cluster Size')
        plt.savefig(join(subdirs[i], 'TransitionMatrix'))
        plt.close()
        # csv matrix
        np.savetxt(join(subdirs[i], 'TransitionMatrix.txt'), percent_matrix,
                   fmt='%6.1f', delimiter=',')
        # Cytoscape-readable matrix
        G = nx.DiGraph()
        for f, g in enumerate(percent_matrix):
            G.add_node(f + 1, size=clusters_pop[f])
            for idx, y in enumerate(g):
                if (y > 0) and (f != idx):
                    G.add_edge(f + 1, idx + 1, percent=y)
        nx.write_graphml(G, join(subdirs[i], 'TransitionMatrix.graphml'))
