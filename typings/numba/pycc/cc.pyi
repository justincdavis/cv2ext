"""
This type stub file was generated by pyright.
"""

from setuptools.extension import Extension
from numba.core.compiler_lock import global_compiler_lock

dir_util = ...
log = ...
extension_libs = ...
class CC:
    """
    An ahead-of-time compiler to create extension modules that don't
    depend on Numba.
    """
    _mixin_sources = ...
    _extra_cflags = ...
    _extra_ldflags = ...
    def __init__(self, extension_name, source_module=...) -> None:
        ...
    
    @property
    def name(self): # -> Any:
        """
        The name of the extension module to create.
        """
        ...
    
    @property
    def output_file(self):
        """
        The specific output file (a DLL) that will be generated.
        """
        ...
    
    @output_file.setter
    def output_file(self, value): # -> None:
        ...
    
    @property
    def output_dir(self): # -> Any:
        """
        The directory the output file will be put in.
        """
        ...
    
    @output_dir.setter
    def output_dir(self, value): # -> None:
        ...
    
    @property
    def use_nrt(self): # -> bool:
        ...
    
    @use_nrt.setter
    def use_nrt(self, value): # -> None:
        ...
    
    @property
    def target_cpu(self): # -> str:
        """
        The target CPU model for code generation.
        """
        ...
    
    @target_cpu.setter
    def target_cpu(self, value): # -> None:
        ...
    
    @property
    def verbose(self): # -> bool:
        """
        Whether to display detailed information when compiling.
        """
        ...
    
    @verbose.setter
    def verbose(self, value): # -> None:
        ...
    
    def export(self, exported_name, sig): # -> Callable[..., Any]:
        """
        Mark a function for exporting in the extension module.
        """
        ...
    
    @global_compiler_lock
    def compile(self): # -> None:
        """
        Compile the extension module.
        """
        ...
    
    def distutils_extension(self, **kwargs): # -> _CCExtension:
        """
        Create a distutils extension object that can be used in your
        setup.py.
        """
        ...
    


class _CCExtension(Extension):
    """
    A Numba-specific Extension subclass to LLVM-compile pure Python code
    to an extension module.
    """
    _cc = ...
    _distutils_monkey_patched = ...
    @classmethod
    def monkey_patch_distutils(cls): # -> None:
        """
        Monkey-patch distutils with our own build_ext class knowing
        about pycc-compiled extensions modules.
        """
        class _CC_build_ext(_orig_build_ext):
            ...
        
        
    

