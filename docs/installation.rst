.. _installation:

Installation
------------

There are multiple methods for installing cv2tools. The recommended method is
to install cv2tools into a virtual environment. This will ensure that the
dependencies are isolated from other Python projects you may be
working on.

Methods
^^^^^^^
#. Pip:

   .. code-block:: console

      $ pip3 install cv2tools

#. From source:

   .. code-block:: console

      $ git clone https://github.com/justincdavis/cv2tools.git
      $ cd cv2tools
      $ pip3 install .

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

#. dev:

   .. code-block:: console

      $ pip3 install cv2tools[dev]
   
   This will install dependencies allowing a full development environment.
   All CI and tools used for development will be installed and can be run.
