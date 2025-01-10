***********
cam_bpw_sim
***********

.. image:: https://www.mypy-lang.org/static/mypy_badge.svg
   :target: https://mypy-lang.org/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/

Blood pressure waveform simulation and evaluation toolbox.
See also our `hardware repo`_ for the mechanical parts of the simulator.

Installation
============

It is recommended to set up a new virtual enviroment for the installation::

    (base) $ python -m venv ~/some/path/cam-bpw-sim
    (base) $ . ~/some/path/cam-bpw-sim/bin/activate
    (cam-bpw-sim) $ pip install ipython ipykernel
    (cam-bpw-sim) $ ipython kernel install --user --name=cam-bpw-sim

Then cloe the repository and install it::

    (cam-bpw-sim) $ cd <project_root>
    (cam-bpw-sim) $ pip install -U .

Cam generation requires `OpenSCAD`_ as well.

.. _OpenSCAD: https://openscad.org/


Generating documentation
========================

::

    (cam-bpw-sim) $ cd <project_root>
    (cam-bpw-sim) $ pip install .[docs]
    (cam-bpw-sim) $ cd <project_root>/docs
    (cam-bpw-sim) $ make clean
    (cam-bpw-sim) $ make clean-autosum
    (cam-bpw-sim) $ make html

This creates ``<project_root>/docs/build/html/index.html``,
that can be opened in a browser.

Development
===========

This project uses some automated QA and source formatting tools, such as
isort_, Flake8_ and Black_::

    (cam-bpw-sim) $ cd <project_root>
    (cam-bpw-sim) $ pip install .[dev]
    (cam-bpw-sim) $ isort .
    (cam-bpw-sim) $ flake8 .
    (cam-bpw-sim) $ black .
    (cam-bpw-sim) $ mypy .

The project is typed.

Tests
=====

::

    (cam-bpw-sim) $ cd <project_root>
    (cam-bpw-sim) $ pip install .[tests]
    (cam-bpw-sim) $ python -m pytest

Tests requiring human evaluation (like visualisations) can be skipped or run
separately (these are marked with the ``@pytest.mark.human`` decorator)::

    $ python -m pytest -m human
    $ python -m pytest -m "not human"

.. _isort: https://pycqa.github.io/isort/
.. _Flake8: https://flake8.pycqa.org/en/latest/
.. _Black: https://black.readthedocs.io/en/stable/index.html

.. _hardware repo: https://github.com/repat8/cam-bpw-sim-hardware
