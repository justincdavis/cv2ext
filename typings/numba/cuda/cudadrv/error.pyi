"""
This type stub file was generated by pyright.
"""

class CudaDriverError(Exception):
    ...


class CudaRuntimeError(Exception):
    ...


class CudaSupportError(ImportError):
    ...


class NvvmError(Exception):
    def __str__(self) -> str:
        ...
    


class NvvmSupportError(ImportError):
    ...


class NvvmWarning(Warning):
    ...


class NvrtcError(Exception):
    def __str__(self) -> str:
        ...
    


class NvrtcCompilationError(NvrtcError):
    ...


class NvrtcSupportError(ImportError):
    ...


