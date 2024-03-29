"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod, abstractproperty

"""
Caching mechanism for compiled functions.
"""
class _Cache(metaclass=ABCMeta):
    @abstractproperty
    def cache_path(self): # -> None:
        """
        The base filesystem path of this cache (for example its root folder).
        """
        ...
    
    @abstractmethod
    def load_overload(self, sig, target_context): # -> None:
        """
        Load an overload for the given signature using the target context.
        The saved object must be returned if successful, None if not found
        in the cache.
        """
        ...
    
    @abstractmethod
    def save_overload(self, sig, data): # -> None:
        """
        Save the overload for the given signature.
        """
        ...
    
    @abstractmethod
    def enable(self): # -> None:
        """
        Enable the cache.
        """
        ...
    
    @abstractmethod
    def disable(self): # -> None:
        """
        Disable the cache.
        """
        ...
    
    @abstractmethod
    def flush(self): # -> None:
        """
        Flush the cache.
        """
        ...
    


class NullCache(_Cache):
    @property
    def cache_path(self): # -> None:
        ...
    
    def load_overload(self, sig, target_context): # -> None:
        ...
    
    def save_overload(self, sig, cres): # -> None:
        ...
    
    def enable(self): # -> None:
        ...
    
    def disable(self): # -> None:
        ...
    
    def flush(self): # -> None:
        ...
    


class _CacheLocator(metaclass=ABCMeta):
    """
    A filesystem locator for caching a given function.
    """
    def ensure_cache_path(self): # -> None:
        ...
    
    @abstractmethod
    def get_cache_path(self): # -> None:
        """
        Return the directory the function is cached in.
        """
        ...
    
    @abstractmethod
    def get_source_stamp(self): # -> None:
        """
        Get a timestamp representing the source code's freshness.
        Can return any picklable Python object.
        """
        ...
    
    @abstractmethod
    def get_disambiguator(self): # -> None:
        """
        Get a string disambiguator for this locator's function.
        It should allow disambiguating different but similarly-named functions.
        """
        ...
    
    @classmethod
    def from_function(cls, py_func, py_file):
        """
        Create a locator instance for the given function located in the
        given file.
        """
        ...
    
    @classmethod
    def get_suitable_cache_subpath(cls, py_file): # -> str:
        """Given the Python file path, compute a suitable path inside the
        cache directory.

        This will reduce a file path that is too long, which can be a problem
        on some operating system (i.e. Windows 7).
        """
        ...
    


class _SourceFileBackedLocatorMixin:
    """
    A cache locator mixin for functions which are backed by a well-known
    Python source file.
    """
    def get_source_stamp(self): # -> tuple[float, int]:
        ...
    
    def get_disambiguator(self): # -> str:
        ...
    
    @classmethod
    def from_function(cls, py_func, py_file): # -> Self | None:
        ...
    


class _UserProvidedCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator that always point to the user provided directory in
    `numba.config.CACHE_DIR`
    """
    def __init__(self, py_func, py_file) -> None:
        ...
    
    def get_cache_path(self): # -> str:
        ...
    
    @classmethod
    def from_function(cls, py_func, py_file): # -> _SourceFileBackedLocatorMixin | None:
        ...
    


class _InTreeCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by a regular Python module with a
    writable __pycache__ directory.
    """
    def __init__(self, py_func, py_file) -> None:
        ...
    
    def get_cache_path(self): # -> str:
        ...
    


class _UserWideCacheLocator(_SourceFileBackedLocatorMixin, _CacheLocator):
    """
    A locator for functions backed by a regular Python module or a
    frozen executable, cached into a user-wide cache directory.
    """
    def __init__(self, py_func, py_file) -> None:
        ...
    
    def get_cache_path(self): # -> str:
        ...
    
    @classmethod
    def from_function(cls, py_func, py_file): # -> Self | None:
        ...
    


class _IPythonCacheLocator(_CacheLocator):
    """
    A locator for functions entered at the IPython prompt (notebook or other).
    """
    def __init__(self, py_func, py_file) -> None:
        ...
    
    def get_cache_path(self): # -> str:
        ...
    
    def get_source_stamp(self): # -> str:
        ...
    
    def get_disambiguator(self): # -> str:
        ...
    
    @classmethod
    def from_function(cls, py_func, py_file): # -> Self | None:
        ...
    


class CacheImpl(metaclass=ABCMeta):
    """
    Provides the core machinery for caching.
    - implement how to serialize and deserialize the data in the cache.
    - control the filename of the cache.
    - provide the cache locator
    """
    _locator_classes = ...
    def __init__(self, py_func) -> None:
        ...
    
    def get_filename_base(self, fullname, abiflags): # -> str:
        ...
    
    @property
    def filename_base(self): # -> str:
        ...
    
    @property
    def locator(self):
        ...
    
    @abstractmethod
    def reduce(self, data): # -> None:
        "Returns the serialized form the data"
        ...
    
    @abstractmethod
    def rebuild(self, target_context, reduced_data): # -> None:
        "Returns the de-serialized form of the *reduced_data*"
        ...
    
    @abstractmethod
    def check_cachable(self, data): # -> None:
        "Returns True if the given data is cachable; otherwise, returns False."
        ...
    


class CompileResultCacheImpl(CacheImpl):
    """
    Implements the logic to cache CompileResult objects.
    """
    def reduce(self, cres):
        """
        Returns a serialized CompileResult
        """
        ...
    
    def rebuild(self, target_context, payload): # -> CompileResult:
        """
        Returns the unserialized CompileResult
        """
        ...
    
    def check_cachable(self, cres): # -> bool:
        """
        Check cachability of the given compile result.
        """
        ...
    


class CodeLibraryCacheImpl(CacheImpl):
    """
    Implements the logic to cache CodeLibrary objects.
    """
    _filename_prefix = ...
    def reduce(self, codelib):
        """
        Returns a serialized CodeLibrary
        """
        ...
    
    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CodeLibrary
        """
        ...
    
    def check_cachable(self, codelib): # -> bool:
        """
        Check cachability of the given CodeLibrary.
        """
        ...
    
    def get_filename_base(self, fullname, abiflags):
        ...
    


class IndexDataCacheFile:
    """
    Implements the logic for the index file and data file used by a cache.
    """
    def __init__(self, cache_path, filename_base, source_stamp) -> None:
        ...
    
    def flush(self): # -> None:
        ...
    
    def save(self, key, data): # -> None:
        """
        Save a new cache entry with *key* and *data*.
        """
        ...
    
    def load(self, key): # -> Any | None:
        """
        Load a cache entry with *key*.
        """
        ...
    


class Cache(_Cache):
    """
    A per-function compilation cache.  The cache saves data in separate
    data files and maintains information in an index file.

    There is one index file per function and Python version
    ("function_name-<lineno>.pyXY.nbi") which contains a mapping of
    signatures and architectures to data files.
    It is prefixed by a versioning key and a timestamp of the Python source
    file containing the function.

    There is one data file ("function_name-<lineno>.pyXY.<number>.nbc")
    per function, function signature, target architecture and Python version.

    Separate index and data files per Python version avoid pickle
    compatibility problems.

    Note:
    This contains the driver logic only.  The core logic is provided
    by a subclass of ``CacheImpl`` specified as *_impl_class* in the subclass.
    """
    _impl_class = ...
    def __init__(self, py_func) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    @property
    def cache_path(self):
        ...
    
    def enable(self): # -> None:
        ...
    
    def disable(self): # -> None:
        ...
    
    def flush(self): # -> None:
        ...
    
    def load_overload(self, sig, target_context): # -> None:
        """
        Load and recreate the cached object for the given signature,
        using the *target_context*.
        """
        ...
    
    def save_overload(self, sig, data): # -> None:
        """
        Save the data for the given signature in the cache.
        """
        ...
    


class FunctionCache(Cache):
    """
    Implements Cache that saves and loads CompileResult objects.
    """
    _impl_class = CompileResultCacheImpl


_lib_cache_prefixes = ...
def make_library_cache(prefix): # -> type[LibraryCache]:
    """
    Create a Cache class for additional compilation features to cache their
    result for reuse.  The cache is saved in filename pattern like
    in ``FunctionCache`` but with additional *prefix* as specified.
    """
    class CustomCodeLibraryCacheImpl(CodeLibraryCacheImpl):
        ...
    
    
    class LibraryCache(Cache):
        """
        Implements Cache that saves and loads CodeLibrary objects for additional
        feature for the specified python function.
        """
        ...
    
    

