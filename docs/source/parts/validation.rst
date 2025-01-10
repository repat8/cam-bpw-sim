**********
Validation
**********


Evaluate against the nominal signal of the cam
==============================================

The notebook can be executed with the ``meas-val`` subcommand::

    $ cam-bpw-sim meas-val 1.0/500R23D1_all/*{1,2,3}.m.hdf5
    Started: /home/user/path/bpsim_home/meas/1.0/500R23D1_all/1H__AAC27_rf50_Phr1__12V__240207_1h.1.m.hdf5
    ... [progress indication omitted]
    meas/1.0/500R23D1_all/meas_vs_cam/P5__AAC249_Phr2__12V__240207.2.m.ipynb
    ...

An executed and parameterized copy of the notebook, and an HDF5 file containing
statistics and other derived data is generated to the folder printed by the command.
By default it is a folder named after the evaluation notebook,
in the same folder as the first measurement file.

You can pass your own implementation of evaluation by passing a notebook path
to the ``--notebook`` option.

Use the ``meas-val-summary`` subcommand to aggregate these statistics::

    $ cam-bpw-sim meas-val-summary 1.0/500R23D1_all/meas_vs_cam
    Started
    meas/1.0/500R23D1_all/meas_vs_cam/meas_vs_cam_fcr_waveform_summary.ipynb


The first argument can match multiple folders.
This command accepts a ``--notebook`` option for custom aggregation and
you can use the ``--tag`` option to filter validation results based on the tag
part of the measurement file name.
