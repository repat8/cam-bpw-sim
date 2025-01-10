*****************
Measurement guide
*****************

.. py:currentmodule:: cam_bpw_sim

Introduction
============

This software requires to place measurement files to a well-defined location.

In the following section, we will use the following naming:

hardware version
    The version of the hardware components repository from which you built the simulator.

.. _meas-setup:

setup
    A specific configuration of the simulator that determines waveform scaling and pressure.
    You may want to change these settings only when

    * the sensor is replaced or
    * you insert a cam with a different amplitude or
    * you insert a cam with inverted signal or
    * the mechanical configuration of the simulator (fulcrum position, lever legth etc.) has changed.

.. _meas-session:

measurement session
    Consecutive measurements with the same cam.
    Users are advised to perform a baseline measurement in each measurement session,
    after replacing the cam.

.. _meas-par:

measurement parameters
    The settings describing an individual measurement of a measurement session.

    These include

    * the cam instance used,
    * the measurement session,
    * voltage of the power supply (cam speed) and
    * measurement protocol.

Some of this information is contained in configuration files while others are
determined by the folder and file naming convention.


Creating a measurement setup
============================

Use the CLI to prepare the file structure for a new setup::

    $ cam-bpw-sim create-setup 1.0 500R25C2D3
    Please edit meas/1.0/500R25C2D3/simulator_setup.json

Then fill the entries of ``simulator_setup.json``.


Measurement protocols
=====================

Our measurements were performed using the following protocols.

P3
    #. Set the cam before the start position (marked with ``<``)
    #. Start the measurement (the motor is still turned off)
    #. At 30 s, start the motor.
    #. At 6 min, stop the measurement.
    #. Stop the motor.

P4
    The same as P3, but measurement is stopped at 3 min.
    Used for cam distortion measurements.

P5
    #. Set the cam before the start position (marked with ``<``)
    #. Start the measurement (the motor is still turned off)
    #. At 20 s, start the motor.
    #. At 6 min 20 s, stop the measurement.
    #. Stop the motor.

1H
    Similar to P5, but contains a 1 hour long measurement.

Currently these protocols are supported and expected by the evaluation scripts.


Convert sensor logs with ``SignalReader``
=========================================

Sensor data may be logged in various file formats.

Evaluation code presented here makes use of the `bpwave`_ package,
and expects waveform input as a :class:`bpwave.Signal`.
You can convert logs with a subclass of :class:`bpwave.SignalReader`.

In our article OptoForce 3D sensor was used; the corresponding implementation of
:class:`bpwave.SignalReader` is :class:`~.sensors.optoforce.OptoForceCsvReader`.

:ref:`Measurement parameters <meas-par>` are by default parsed from the filename,
which should folllow the pattern ``<protocol name>__<cam reference>__<voltage>V__<tag>.<sequence>.<ext>``,
e. g. ``P3__AAC4__12_5V__20240101.1.csv``.

* ``<cam reference>`` should be already defined in ``simulator_setup.json``.
* In ``<voltage>``, decimal dot can be replaced with ``_``.
* Baseline measurements should have ``base`` instead of the ``<sequence>``,
  e. g. the baseline measurement for the previous example would be
  ``P3__AAC4__12_5V__20240101.base.csv``.
* ``<tag>`` can be anything that uniquely identifies the
  :ref:`measurement session <meas-session>`
  within the scope of the :ref:`measurement setup <meas-setup>`.

Example (importing OptoForce log with the CLI)::

    $ cam-bpw-sim import-optoforce --fs 333 1.0/500R25C2D3 /path/to/file/P3__AAC27_Phr1__12V__20231228.1.txt
    {
        "protocol": "P3",
        "cam_inst": "AAC27_Phr1",
        "u": 12.0
    }
    Are these parameters correct? [y/N]: y
    meas/1.0/500R25C2D3/P3__AAC27_Phr1__12V__20231228.1.m.hdf5

Using the API::

    out_path, data = app.convert_measurement_log(
        reader=optoforce.OptoForceCsvReader(fs_override=fs, channel=channel),
        in_path=in_path,
        meas_setup_folder=meas_setup_folder,
        confirm=confirm,
        force=force,
    )

You can load your own format by passing a different :class:`bpwave.SignalReader`
to the parameter ``reader``.

.. _bpwave: https://pypi.org/project/bpwave/
