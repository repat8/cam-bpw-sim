***************
Generating cams
***************

.. py:currentmodule:: cam_bpw_sim

This repository contains the procedures for creating a 3D printable cam from an
arbitrary (with some restrictions) arterial blood pressure or artificial signal.

.. note::

    The rest of the mechanical parts of the simulator, along with assembly
    instructions and sample cams, can be found in the `hardware repository`_.


Using the CLI
=============

Here we demonstrate creating a cam representing a section of a BP recording from
the Autonomic Aging dataset [AACWeb]_.

Use the ``physionet`` subcommand to download a signal::

    $ cam-bpw-sim physionet 0002
    /home/user/bpsim_home/raw_signal/physionet_aac_0002_full.hdf5

.. image:: /_static/fig/cli-physionet.png

.. note::
    All commands support the ``--help`` option to display available parameters.

Alternatively, a section can be downloaded::

    $ cam-bpw-sim physionet 0002 --start 100 --stop 10000
    /home/user/bpsim_home/raw_signal/physionet_aac_0002_100_10000.hdf5

.. image:: /_static/fig/cli-physionet-section.png

The downloaded signals are now in the ``raw_signals`` folder::

    $ cam-bpw-sim list-raw
    raw_signal/physionet_aac_0002_full.hdf5
    raw_signal/physionet_aac_0002_100_10000.hdf5

You can use the ``view-signal`` subcommand to open a Matplotlib plot window and
see the waveform. The argument is the relative path from the app home::

    $ cam-bpw-sim view-signal raw_signal/physionet_aac_0002_100_10000.hdf5

.. image:: /_static/fig/cli-view-signal.png

Then we need to detect the characteristic points, most importantly the onsets::

    $ cam-bpw-sim chpoints physionet_aac_0002_100_10000.hdf5
    prep_signal/physionet_aac_0002_100_10000.chpoints.hdf5

.. image:: /_static/fig/cli-chpoints.png

This command created a copy of the signal to the folder ``prep_signal``.
We can examine the result with ``view-signal``, as shown above::

    $ cam-bpw-sim view-signal prep_signal/physionet_aac_0002_100_10000.chpoints.hdf5

Having the characteristic points, the starting point can be chosen.
For reliable validation results, it is recommended to choose clear and
unambiguous onsets.
The length of the cam surface signal, in terms of cardiac cycles,
should be determined based on the rotation speed, in order to produce a realistic
signal.
Based on these parameters, the preprocessed and named cam signal can be created::

    $ cam-bpw-sim cam-signal physionet_aac_0002_100_10000.chpoints.hdf5 \
    AAC2_0 \
    --from-onset 0 \
    --n-ccycles 6
    prep_signal/AAC1_0.camsig.hdf5

.. image:: /_static/fig/cli-cam-signal.png

This command performs the baseline correction as well, so the condition of the
same amplitude at the two end points is met.

Finally, the STL model needs to be generated.
The ``cam-stl`` subcommand asks for confirmation, so that the script can be canceled
if the parameters need to be changed (e. g. because of surface anomalies).

For example, generate a cam with 30mm radius, 1mm signal amplitude,
3mm cam follower diameter and a nominal 6mm shaft diameter
(increased based on printer tolerance test)::

    $ cam-bpw-sim cam-stl AAC2_0 -A 1 -r 30 -F 3 -d 6.05
    Continue? [y/N]: y
    Generating STL started...
    Geometries in cache: 43
    Geometry cache size in bytes: 9414048
    CGAL Polyhedrons in cache: 4
    CGAL cache size in bytes: 91253904
    Total rendering time: 0:02:10.058
       Top level object is a 3D object:
       Simple:        yes
       Vertices:    24130
       Halfedges:   95850
       Edges:       47925
       Halffacets:  47636
       Facets:      23818
       Volumes:         2
    STL is ready.
    cam/AAC2_0/d605r3000a100rf150/SC.1.1/AAC2_0_d605r3000a100rf150.cam.stl

.. image:: /_static/fig/cli-cam-stl-1.png

After printing, the new instance needs to be registered to the software,
so that manufacturing data can be documented in measurement files::

    $ cam-bpw-sim cam-inst AAC2_0/d605r3000a100rf150/SC.1.1
    technology (3D_print): 3D_print
    machine (Sonic Mini, CraftBot/1, CraftBot/2, CraftBot/3): Sonic Mini
    material (PLA, PLA+, Phrozen): Phrozen
    date [2024-03-21]:
    comment []:
    OK? [y/N]: y
    /home/user/bpsim_home/cam/AAC2_0/d605r3000a100rf150/SC.1.1/inst/1/inst.json


Using the Python API
====================

These steps can be performed using the Python API as well, see the example
Jupyter notebook ``notebooks/custom_cam_profile.ipynb``.
This way it is easier to process custom signals that are created already by
Python code.

.. _hardware repository: https://github.com/repat8/cam-bpw-sim-hardware

.. [AACWeb] https://www.physionet.org/content/autonomic-aging-cardiovascular/1.0.0/