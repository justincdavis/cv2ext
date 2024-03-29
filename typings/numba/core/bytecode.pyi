"""
This type stub file was generated by pyright.
"""

from numba.core import serialize
from numba.core.utils import PYVERSION

if PYVERSION in ((3, 12), ):
    INSTR_LEN = ...
else:
    ...
opcode_info = ...
_ExceptionTableEntry = ...
_FIXED_OFFSET = ...
def get_function_object(obj): # -> Any:
    """
    Objects that wraps function should provide a "__numba__" magic attribute
    that contains a name of an attribute that contains the actual python
    function object.
    """
    ...

def get_code_object(obj): # -> Any | None:
    "Shamelessly borrowed from llpython"
    ...

JREL_OPS = ...
JABS_OPS = ...
JUMP_OPS = ...
TERM_OPS = ...
EXTENDED_ARG = ...
HAVE_ARGUMENT = ...
class ByteCodeInst:
    '''
    Attributes
    ----------
    - offset:
        byte offset of opcode
    - opcode:
        opcode integer value
    - arg:
        instruction arg
    - lineno:
        -1 means unknown
    '''
    __slots__ = ...
    def __init__(self, offset, opcode, arg, nextoffset) -> None:
        ...
    
    @property
    def is_jump(self): # -> bool:
        ...
    
    @property
    def is_terminator(self): # -> bool:
        ...
    
    def get_jump_target(self): # -> Any:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    @property
    def block_effect(self): # -> Literal[1, -1, 0]:
        """Effect of the block stack
        Returns +1 (push), 0 (none) or -1 (pop)
        """
        ...
    


CODE_LEN = ...
ARG_LEN = ...
NO_ARG_LEN = ...
OPCODE_NOP = ...
class ByteCodeIter:
    def __init__(self, code) -> None:
        ...
    
    def __iter__(self): # -> Self:
        ...
    
    def next(self): # -> tuple[Any | int, ByteCodeInst]:
        ...
    
    __next__ = ...
    def read_arg(self, size): # -> int:
        ...
    


class _ByteCode:
    """
    The decoded bytecode of a function, and related information.
    """
    __slots__ = ...
    def __init__(self, func_id) -> None:
        ...
    
    def __iter__(self): # -> Iterator[ByteCodeInst]:
        ...
    
    def __getitem__(self, offset): # -> ByteCodeInst:
        ...
    
    def __contains__(self, offset): # -> bool:
        ...
    
    def dump(self): # -> str:
        ...
    
    def get_used_globals(self): # -> dict[Any, Any]:
        """
        Get a {name: value} map of the globals used by this code
        object and any nested code objects.
        """
        ...
    


class ByteCodePy311(_ByteCode):
    def __init__(self, func_id) -> None:
        ...
    
    @staticmethod
    def fixup_eh(ent):
        ...
    
    def find_exception_entry(self, offset): # -> None:
        """
        Returns the exception entry for the given instruction offset
        """
        ...
    


class ByteCodePy312(ByteCodePy311):
    def __init__(self, func_id) -> None:
        ...
    
    @property
    def ordered_offsets(self): # -> list[Any | int]:
        ...
    
    def remove_build_list_swap_pattern(self, entries): # -> list[Any]:
        """ Find the following bytecode pattern:

            BUILD_{LIST, MAP, SET}
            SWAP(2)
            FOR_ITER
            ...
            END_FOR
            SWAP(2)

            This pattern indicates that a list/dict/set comprehension has
            been inlined. In this case we can skip the exception blocks
            entirely along with the dead exceptions that it points to.
            A pair of exception that sandwiches these exception will
            also be merged into a single exception.
        """
        ...
    


if PYVERSION == (3, 11):
    ByteCode = ...
else:
    ByteCode = ...
class FunctionIdentity(serialize.ReduceMixin):
    """
    A function's identity and metadata.

    Note this typically represents a function whose bytecode is
    being compiled, not necessarily the top-level user function
    (the two might be distinct).
    """
    _unique_ids = ...
    @classmethod
    def from_function(cls, pyfunc): # -> Self:
        """
        Create the FunctionIdentity of the given function.
        """
        ...
    
    def derive(self): # -> Self:
        """Copy the object and increment the unique counter.
        """
        ...
    


