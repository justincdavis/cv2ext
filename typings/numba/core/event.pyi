"""
This type stub file was generated by pyright.
"""

import abc
import enum
from contextlib import contextmanager
from numba.core import config

"""
The ``numba.core.event`` module provides a simple event system for applications
to register callbacks to listen to specific compiler events.

The following events are built in:

- ``"numba:compile"`` is broadcast when a dispatcher is compiling. Events of
  this kind have ``data`` defined to be a ``dict`` with the following
  key-values:

  - ``"dispatcher"``: the dispatcher object that is compiling.
  - ``"args"``: the argument types.
  - ``"return_type"``: the return type.

- ``"numba:compiler_lock"`` is broadcast when the internal compiler-lock is
  acquired. This is mostly used internally to measure time spent with the lock
  acquired.

- ``"numba:llvm_lock"`` is broadcast when the internal LLVM-lock is acquired.
  This is used internally to measure time spent with the lock acquired.

- ``"numba:run_pass"`` is broadcast when a compiler pass is running.

    - ``"name"``: pass name.
    - ``"qualname"``: qualified name of the function being compiled.
    - ``"module"``: module name of the function being compiled.
    - ``"flags"``: compilation flags.
    - ``"args"``: argument types.
    - ``"return_type"`` return type.

Applications can register callbacks that are listening for specific events using
``register(kind: str, listener: Listener)``, where ``listener`` is an instance
of ``Listener`` that defines custom actions on occurrence of the specific event.
"""
class EventStatus(enum.Enum):
    """Status of an event.
    """
    START = ...
    END = ...


_builtin_kinds = ...
class Event:
    """An event.

    Parameters
    ----------
    kind : str
    status : EventStatus
    data : any; optional
        Additional data for the event.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``.
    """
    def __init__(self, kind, status, data=..., exc_details=...) -> None:
        ...
    
    @property
    def kind(self): # -> Any:
        """Event kind

        Returns
        -------
        res : str
        """
        ...
    
    @property
    def status(self): # -> Any:
        """Event status

        Returns
        -------
        res : EventStatus
        """
        ...
    
    @property
    def data(self): # -> None:
        """Event data

        Returns
        -------
        res : object
        """
        ...
    
    @property
    def is_start(self):
        """Is it a *START* event?

        Returns
        -------
        res : bool
        """
        ...
    
    @property
    def is_end(self):
        """Is it an *END* event?

        Returns
        -------
        res : bool
        """
        ...
    
    @property
    def is_failed(self): # -> bool:
        """Is the event carrying an exception?

        This is used for *END* event. This method will never return ``True``
        in a *START* event.

        Returns
        -------
        res : bool
        """
        ...
    
    def __str__(self) -> str:
        ...
    
    __repr__ = ...


_registered = ...
def register(kind, listener): # -> None:
    """Register a listener for a given event kind.

    Parameters
    ----------
    kind : str
    listener : Listener
    """
    ...

def unregister(kind, listener): # -> None:
    """Unregister a listener for a given event kind.

    Parameters
    ----------
    kind : str
    listener : Listener
    """
    ...

def broadcast(event): # -> None:
    """Broadcast an event to all registered listeners.

    Parameters
    ----------
    event : Event
    """
    ...

class Listener(abc.ABC):
    """Base class for all event listeners.
    """
    @abc.abstractmethod
    def on_start(self, event): # -> None:
        """Called when there is a *START* event.

        Parameters
        ----------
        event : Event
        """
        ...
    
    @abc.abstractmethod
    def on_end(self, event): # -> None:
        """Called when there is a *END* event.

        Parameters
        ----------
        event : Event
        """
        ...
    
    def notify(self, event): # -> None:
        """Notify this Listener with the given Event.

        Parameters
        ----------
        event : Event
        """
        ...
    


class TimingListener(Listener):
    """A listener that measures the total time spent between *START* and
    *END* events during the time this listener is active.
    """
    def __init__(self) -> None:
        ...
    
    def on_start(self, event): # -> None:
        ...
    
    def on_end(self, event): # -> None:
        ...
    
    @property
    def done(self): # -> bool:
        """Returns a ``bool`` indicating whether a measurement has been made.

        When this returns ``False``, the matching event has never fired.
        If and only if this returns ``True``, ``.duration`` can be read without
        error.
        """
        ...
    
    @property
    def duration(self): # -> Any | float:
        """Returns the measured duration.

        This may raise ``AttributeError``. Users can use ``.done`` to check
        that a measurement has been made.
        """
        ...
    


class RecordingListener(Listener):
    """A listener that records all events and stores them in the ``.buffer``
    attribute as a list of 2-tuple ``(float, Event)``, where the first element
    is the time the event occurred as returned by ``time.time()`` and the second
    element is the event.
    """
    def __init__(self) -> None:
        ...
    
    def on_start(self, event): # -> None:
        ...
    
    def on_end(self, event): # -> None:
        ...
    


@contextmanager
def install_listener(kind, listener): # -> Generator[Any, Any, None]:
    """Install a listener for event "kind" temporarily within the duration of
    the context.

    Returns
    -------
    res : Listener
        The *listener* provided.

    Examples
    --------

    >>> with install_listener("numba:compile", listener):
    >>>     some_code()  # listener will be active here.
    >>> other_code()     # listener will be unregistered by this point.

    """
    ...

@contextmanager
def install_timer(kind, callback): # -> Generator[TimingListener, Any, None]:
    """Install a TimingListener temporarily to measure the duration of
    an event.

    If the context completes successfully, the *callback* function is executed.
    The *callback* function is expected to take a float argument for the
    duration in seconds.

    Returns
    -------
    res : TimingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, TimingListener()) as res:
    >>>    ...
    """
    ...

@contextmanager
def install_recorder(kind): # -> Generator[RecordingListener, Any, None]:
    """Install a RecordingListener temporarily to record all events.

    Once the context is closed, users can use ``RecordingListener.buffer``
    to access the recorded events.

    Returns
    -------
    res : RecordingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, RecordingListener()) as res:
    >>>    ...
    """
    ...

def start_event(kind, data=...): # -> None:
    """Trigger the start of an event of *kind* with *data*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
    ...

def end_event(kind, data=..., exc_details=...): # -> None:
    """Trigger the end of an event of *kind*, *exc_details*.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    exc_details : 3-tuple; optional
        Same 3-tuple for ``__exit__``. Or, ``None`` if no error.
    """
    ...

@contextmanager
def trigger_event(kind, data=...): # -> Generator[None, Any, None]:
    """A context manager to trigger the start and end events of *kind* with
    *data*. The start event is triggered when entering the context.
    The end event is triggered when exiting the context.

    Parameters
    ----------
    kind : str
        Event kind.
    data : any; optional
        Extra event data.
    """
    ...

if config.CHROME_TRACE:
    ...
