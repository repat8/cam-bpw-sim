***************
Getting started
***************

This software follows the `Convention over Configuration`_ principle,
therefore the entities handled are stored on the user's computer in a conventional structure.

The CLI provides commands to initialize this structure and create these entities
at the right location.

The CLI can be invoked in two ways::

    $ python -m cam_bpw_sim  # note the underscores
    $ cam-bpw-sim  # note the hyphens

.. note::
    Use the ``--version`` option to find out which version is installed.
    The main command and all subcommands support the ``--help`` option to list
    subcommands and parameters, respectively::

        $ cam-bpw-sim --help
        Usage: cam-bpw-sim [OPTIONS] COMMAND [ARGS]...

        Options:
          -v, --version         Show the application's version and exit.
          --install-completion  Install completion for the current shell.
          --show-completion     Show completion for the current shell, to copy it or
                                customize the installation.
          --help                Show this message and exit.

        Commands:
           init-config        Create application config file at the user's config folder.                                                                                                                                                                                                                                       │
           init-home          Init application home folder structure at the location specified by the config.                                                                                                                                                                                                                   │
           physionet          Download BP signal from the Autonomic Aging dataset of Physionet.                                                                                                                                                                                                                                 │
           list-raw           List available raw signals.                                                                                                                                                                                                                                                                       │
           list-prep          List available preprocessed signals.                                                                                                                                                                                                                                                              │
           view-signal        Visualize a signal.                                                                                                                                                                                                                                                                               │
           chpoints           Create a preprocessed signal by detecting characteristic points.                                                                                                                                                                                                                                  │
           cam-signal         Create a preprocessed signal suitable for cam generation.                                                                                                                                                                                                                                         │
           cam-stl            Generate an STL file from a preprocessed cam signal.                                                                                                                                                                                                                                              │
           cam-inst           Register a physical cam instance.                                                                                                                                                                                                                                                                 │
           create-setup       Register a new measurement setup.                                                                                                                                                                                                                                                                 │
           draw-setup         Display a schematic drawing of the simulator given a measurement setup.                                                                                                                                                                                                                           │
           import-optoforce   Import a measurement log from an OptoForce 3D force sensor to a measurement setup.                                                                                                                                                                                                                │
           meas-val           Run measurement validation with imported measurement files.                                                                                                                                                                                                                                       │
           meas-val-summary   Calculate aggregated statistics on measurement validations.

.. _`Convention over Configuration`: https://en.wikipedia.org/wiki/Convention_over_configuration

Create config with ``init-config``
==================================

The following command creates the main configuration file in the default configuration folder of your system::

    $ cam-bpw-sim init-config

The command prints the location of the resulting ``config.toml`` file.

Create a folder at a location of your choice then edit the value of ``simulator_home``
to point to this folder:

.. code-block:: toml

    simulator_home = "/path/to/your/folder"

Note that TOML format allows commenting out lines with ``#``, so later it is easy
to switch between different folders on your computer.

This config file is the home of the choices that can be used when registering
cam instances. An example:

.. code-block:: toml

    [cam_instance]
    technology = ["3D_print"]
    machine = [
        "Sonic Mini",
        "CraftBot/1",
        "CraftBot/2",
        "CraftBot/3",
    ]
    material = ["PLA", "PLA+", "Phrozen"]

Initialize base structure with ``init-home``
============================================

The following command will read the aforementioned config file and create the
necessary subfolders::

    $ cam-bpw-sim init-home

