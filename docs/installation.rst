.. _installation:

Installation
------------

There are multiple methods for installing cv2ext. The recommended method is
to install cv2ext into a virtual environment. This will ensure that the
dependencies are isolated from other Python projects you may be
working on.

Methods
^^^^^^^
#. Pip:

   .. code-block:: console

      $ pip3 install cv2ext

#. From source:

   .. code-block:: console

      $ git clone https://github.com/justincdavis/cv2ext.git
      $ cd cv2ext
      $ pip3 install .

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

#. dev:

   .. code-block:: console

      $ pip3 install cv2ext[dev]
   
   This will install dependencies allowing a full development environment.
   All CI and tools used for development will be installed and can be run.
