#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Daniel Platero Rochart & Roy González-Alemán
@contact: [daniel.platero, roy_gonzalez]@fq.uh.cu
"""
import os
from os.path import join

import numpy as np
import dpfuncs as dpf


# >>>> Debugging section <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import argparse
args = argparse.Namespace()
# args.trajectory = "/home/rga/BSProject/runners/trajs/trajs/traj4clust/250K.dcd"
# args.topology = "/home/rga/BSProject/runners/trajs/trajs/traj4clust/250K.psf"
args.trajectory = None
args.topology = None
args.first = None
args.last = None
args.stride = None
args.selection = None
args.cutoff = None
args.density_cut = 54
args.distance_cut = 3.5
args.restart = './working/RCDP-aligned_tau/restart.pickle'
args.restart = os.path.abspath(args.restart)
args.automatic = 'False'
args.outdir = '/home/rga/Desktop/'
args.outdir = os.path.abspath(args.outdir)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def main():
    """
    Execute main function of RCDPeaks.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # =========================================================================
    # 1. Load and check user arguments
    # =========================================================================
    args = dpf.parse_arguments()

    # ++++ when automatic detecting cluster centers +++++++++++++++++++++++++++
    # cutoffs for the Decision Graph must be specified together
    cutoffs = [args.distance_cut, args.density_cut]
    if any(cutoffs) and not all(cutoffs):
        raise ValueError('\n\n>>> Arguments Inconsistency\n-dcut, and -rcut'
                         ' arguments must be specified together or not at all.')
    # automatic detection of centers does not use any cutoffs
    if all(cutoffs) and args.automatic == 'True':
        raise ValueError('\n\n>>> Arguments Inconsistency\nThe -auto_centers'
                         ' argument can not be set to True if -dcut and -rcut'
                         ' options are specified.')
    # detection of centers is mandatory
    if not all(cutoffs) and args.automatic == 'False':
        raise ValueError('\n\n>>> Arguments Inconsistency\nDetection of centers'
                         ' must be done either in an automatic way'
                         ' (-auto_centers) or by passing density and distance'
                         ' cutoffs (-dcut and -rcut).')
    # ++++ when restarting jobs +++++++++++++++++++++++++++++++++++++++++++++++
    if args.restart:
        args.restart = os.path.abspath(args.restart)
        if not os.path.exists(args.restart):
            raise ValueError('\n\n>>> Arguments Inconsistency\nThe path to the'
                             ' restart pickle file you provided does not exist.')
        # load pickled restart info
        (args.first, args.stride, args.last, args.selection, args.ptopology,
         args.ptrajectory, args.cutoff, nnhd_arr, delta_arr, rho_arr) = \
            dpf.unpickle_from_file(args.restart)
    # warning when paths for restarting topotraj differs from cli
    # the program will use cli parameters
    topotraj = [args.topology, args.trajectory]
    if args.restart and any(topotraj):
        print('\n\n>>> WARNING !!!\nYou are specifying the -restart_from'
              ' argument along with -top and/or -traj.\nNote that'
              ' restarting jobs MUST use the same trajectory and'
              ' topology files that were used in the original run'
              ' to avoid inconsistencies.\n\n'
              'The original trajectory and topology files that'
              ' generated your current restart.pickle ({}) were:\n\n'.format(
                  args.restart)
              + '  TOPOLOGY : {}\n'.format(args.ptopology)
              + 'TRAJECTORY : {}\n\n'.format(args.ptrajectory)
              + 'Please ensure that the -top and/or -traj arguments'
              ' you are passing now are consistent, with the'
              ' previous run.\n\n')
    if args.restart and not any(topotraj):
        args.trajectory = args.ptrajectory
        args.topology = args.ptopology
    if args.restart:
        print('\n\n>>> Restarting job with the following arguments:\n\n'
              '-top {}\n'.format(args.topology)
              + '-traj {}\n'.format(args.trajectory)
              + '-first {}\n'.format(args.first)
              + '-last {}\n'.format(args.last)
              + '-stride {}\n'.format(args.stride)
              + '-sel {}\n'.format(args.selection)
              + '-cutoff {}\n'.format(args.cutoff)
              + '-dcut {}\n'.format(args.distance_cut)
              + '-rcut {}\n'.format(args.density_cut)
              + '-restart_from {}\n'.format(args.restart)
              + '-autocenters {}\n'.format(args.automatic)
              + '-odir {}\n'.format(args.outdir)
              + '\nNote that -first, -last, -sel, and -cutoff options were set'
              ' to the values found in {}\n\n'.format(args.restart))

    outlabel = 'RCDP-{}'.format(os.path.basename(args.trajectory).split('.')[0])
    out_dir = join(args.outdir, outlabel)
    if os.path.exists(out_dir):
        raise ValueError('\n\n>>> Arguments Inconsistency\nOutput directory '
                         '"{}" exists. Aborting to avoid'.format(out_dir)
                         + ' overwriting.')
    # =========================================================================
    # 2. Loading trajectory
    # =========================================================================
    print('\n\nLoading trajectory')
    trajectory = dpf.load_raw_traj(args.trajectory, dpf.valid_trajs,
                                   args.topology)
    N1 = trajectory.n_frames
    trajectory = dpf.shrink_traj_selection(trajectory, args.selection)
    trajectory = dpf.shrink_traj_range(args.first, args.last, args.stride,
                                       trajectory)
    trajectory.center_coordinates()
    N = trajectory.n_frames
    cutoff = np.full(trajectory.n_frames, args.cutoff / 10, dtype=np.float32)

    # =========================================================================
    # 3. Getting nnhd_arr, rho_arr and delta_arr
    # =========================================================================
    # ... from previous run
    if args.restart:
        print('\n\nNearest neighbors, densities and distances provided.')
    # ... from scratch
    else:
        print('\n\nComputing Nearest Neighbors, densities and distances.')
        nnhd_arr, rho_arr, delta_arr = dpf.compute_tree(trajectory, cutoff, N)

    # =========================================================================
    # 4. RCDPeaks clustering
    # =========================================================================
    # ... auto-detect centers by iterative Garza-Flores method ................
    if args.automatic == 'True':
        nodes_by_level = dpf.autodetect_centers(delta_arr, rho_arr)
    # ... detect centers by user-specified arguments ..........................
    else:
        delta_arr_c = np.where(delta_arr >= args.distance_cut / 10)[0]
        rho_arr_c = np.where(rho_arr >= args.density_cut)[0]
        nodes_by_level = [np.intersect1d(delta_arr_c, rho_arr_c)]

    # ... merge non-orthogonal centers ........................................
    merged_by_level, neighborhoods_by_level = dpf.merge_centers(
        nodes_by_level, trajectory, cutoff[0])
    # ... assign DP-exact labels ..............................................
    dp_clusters_arrays_by_level = dpf.dp_assign_clusters(
        nnhd_arr, merged_by_level)
    # ... refine clusters by restricting radius ...............................
    refined_clusters_by_level = dpf.refine_dp_assignment(
        merged_by_level, neighborhoods_by_level, dp_clusters_arrays_by_level)

    # =========================================================================
    # 5. Generating outputs
    # =========================================================================
    # >>>> create hierarchy ___________________________________________________
    root_dir, subdirs = dpf.create_dir_hierarchy(args.outdir, args.trajectory,
                                                 merged_by_level)
    # >>>> restart pickle file ________________________________________________
    dpf.output_restart(root_dir, args.first, args.stride, args.last,
                       args.selection, args.topology, args.trajectory,
                       args.cutoff, nnhd_arr, delta_arr, rho_arr)
    # >>>> decision graph _____________________________________________________
    dpf.output_decision_graph(delta_arr, rho_arr, merged_by_level, subdirs)
    # >>>> frames info for refined & exact labeling ___________________________
    dpf.output_frames_info(N1, args.first, args.last, args.stride,
                           refined_clusters_by_level, subdirs, 'refined')
    dpf.output_frames_info(N1, args.first, args.last, args.stride,
                           dp_clusters_arrays_by_level, subdirs, 'exact')
    # >>>> clusters info for refined & exact labeling _________________________
    dpf.output_clusters_info(N1, args.first, args.last, args.stride,
                             merged_by_level, dp_clusters_arrays_by_level,
                             refined_clusters_by_level, rho_arr, delta_arr,
                             subdirs)
    # >>>> clusters centers in PDB ____________________________________________
    dpf.save_centers_pdb(trajectory, merged_by_level, subdirs)
    # >>>> VMD scripts ________________________________________________________
    for i, x in enumerate(refined_clusters_by_level):
        p1 = join(subdirs[i], 'refined_clusters.log')
        p2 = join(subdirs[i], 'exact_clusters.log')
        dpf.to_VMD(p1, args.topology, args.first, N1, args.last, args.stride,
                   refined_clusters_by_level[i])
        dpf.to_VMD(p2, args.topology, args.first, N1, args.last, args.stride,
                   dp_clusters_arrays_by_level[i])
    # >>>> Transition Matrix __________________________________________________
    dpf.output_transition_matrix(dp_clusters_arrays_by_level, subdirs)
    print('\n\nNormal termination of RCDPeaks')


if __name__ == '__main__':
    main()
