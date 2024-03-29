"""
This type stub file was generated by pyright.
"""

import contextlib
from collections.abc import Sequence

class Rating:
    __slots__ = ...
    def __init__(self) -> None:
        ...
    
    def astuple(self): # -> tuple[int, int, int]:
        """Returns a tuple suitable for comparing with the worse situation
        start first.
        """
        ...
    
    def __add__(self, other): # -> _NotImplementedType | Rating:
        ...
    


class CallStack(Sequence):
    """
    A compile-time call stack
    """
    def __init__(self) -> None:
        ...
    
    def __getitem__(self, index):
        """
        Returns item in the stack where index=0 is the top and index=1 is
        the second item from the top.
        """
        ...
    
    def __len__(self): # -> int:
        ...
    
    @contextlib.contextmanager
    def register(self, target, typeinfer, func_id, args): # -> Generator[None, Any, None]:
        ...
    
    def finditer(self, py_func): # -> Generator[Any, Any, None]:
        """
        Yields frame that matches the function object starting from the top
        of stack.
        """
        ...
    
    def findfirst(self, py_func): # -> None:
        """
        Returns the first result from `.finditer(py_func)`; or None if no match.
        """
        ...
    
    def match(self, py_func, args): # -> None:
        """
        Returns first function that matches *py_func* and the arguments types in
        *args*; or, None if no match.
        """
        ...
    


class CallFrame:
    """
    A compile-time call frame
    """
    def __init__(self, target, typeinfer, func_id, args) -> None:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def add_return_type(self, return_type): # -> None:
        """Add *return_type* to the list of inferred return-types.
        If there are too many, raise `TypingError`.
        """
        ...
    


class BaseContext:
    """A typing context for storing function typing constrain template.
    """
    def __init__(self) -> None:
        ...
    
    def init(self): # -> None:
        """
        Initialize the typing context.  Can be overridden by subclasses.
        """
        ...
    
    def refresh(self): # -> None:
        """
        Refresh context with new declarations from known registries.
        Useful for third-party extensions.
        """
        ...
    
    def explain_function_type(self, func): # -> str:
        """
        Returns a string description of the type of a function
        """
        ...
    
    def resolve_function_type(self, func, args, kws): # -> None:
        """
        Resolve function type *func* for argument types *args* and *kws*.
        A signature is returned.
        """
        ...
    
    def resolve_getattr(self, typ, attr): # -> Buffer | Any | ExternalFunctionPointer | Opaque | ExternalFunction | Module | None:
        """
        Resolve getting the attribute *attr* (a string) on the Numba type.
        The attribute's type is returned, or None if resolution failed.
        """
        ...
    
    def find_matching_getattr_template(self, typ, attr): # -> dict[str, Any] | None:
        ...
    
    def resolve_setattr(self, target, attr, value): # -> Signature | None:
        """
        Resolve setting the attribute *attr* (a string) on the *target* type
        to the given *value* type.
        A function signature is returned, or None if resolution failed.
        """
        ...
    
    def resolve_static_getitem(self, value, index): # -> None:
        ...
    
    def resolve_static_setitem(self, target, index, value): # -> None:
        ...
    
    def resolve_setitem(self, target, index, value): # -> Any | None:
        ...
    
    def resolve_delitem(self, target, index): # -> Any | None:
        ...
    
    def resolve_module_constants(self, typ, attr): # -> Buffer | Any | ExternalFunctionPointer | Opaque | ExternalFunction | Module | None:
        """
        Resolve module-level global constants.
        Return None or the attribute type
        """
        ...
    
    def resolve_argument_type(self, val): # -> Buffer | Any | ExternalFunctionPointer | Opaque:
        """
        Return the numba type of a Python value that is being used
        as a function argument.  Integer types will all be considered
        int64, regardless of size.

        ValueError is raised for unsupported types.
        """
        ...
    
    def resolve_value_type(self, val): # -> Buffer | Any | ExternalFunctionPointer | Opaque | ExternalFunction | Module:
        """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        ValueError is raised for unsupported types.
        """
        ...
    
    def resolve_value_type_prefer_literal(self, value): # -> Buffer | Any | ExternalFunctionPointer | Opaque | ExternalFunction | Module | Literal:
        """Resolve value type and prefer Literal types whenever possible.
        """
        ...
    
    def load_additional_registries(self): # -> None:
        """
        Load target-specific registries.  Can be overridden by subclasses.
        """
        ...
    
    def install_registry(self, registry): # -> None:
        """
        Install a *registry* (a templates.Registry instance) of function,
        attribute and global declarations.
        """
        ...
    
    def insert_global(self, gv, gty): # -> None:
        ...
    
    def insert_attributes(self, at): # -> None:
        ...
    
    def insert_function(self, ft): # -> None:
        ...
    
    def insert_user_function(self, fn, ft): # -> None:
        """Insert a user function.

        Args
        ----
        - fn:
            object used as callee
        - ft:
            function template
        """
        ...
    
    def can_convert(self, fromty, toty): # -> Literal[Conversion.exact, Conversion.promote, Conversion.safe, Conversion.unsafe]:
        """
        Check whether conversion is possible from *fromty* to *toty*.
        If successful, return a numba.typeconv.Conversion instance;
        otherwise None is returned.
        """
        ...
    
    def install_possible_conversions(self, actualargs, formalargs): # -> bool:
        """
        Install possible conversions from the actual argument types to
        the formal argument types in the C++ type manager.
        Return True if all arguments can be converted.
        """
        ...
    
    def resolve_overload(self, key, cases, args, kws, allow_ambiguous=..., unsafe_casting=..., exact_match_required=...): # -> None:
        """
        Given actual *args* and *kws*, find the best matching
        signature in *cases*, or None if none matches.
        *key* is used for error reporting purposes.
        If *allow_ambiguous* is False, a tie in the best matches
        will raise an error.
        If *unsafe_casting* is False, unsafe casting is forbidden.
        """
        ...
    
    def unify_types(self, *typelist): # -> None:
        ...
    
    def unify_pairs(self, first, second): # -> None:
        """
        Try to unify the two given types.  A third type is returned,
        or None in case of failure.
        """
        ...
    


class Context(BaseContext):
    def load_additional_registries(self): # -> None:
        ...
    


