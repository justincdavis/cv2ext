"""
This type stub file was generated by pyright.
"""

from contextlib import contextmanager
from numba.core.compiler_machinery import AnalysisPass, FunctionPass, SSACompliantMixin, register_pass

@contextmanager
def fallback_context(state, msg): # -> Generator[None, Any, None]:
    """
    Wraps code that would signal a fallback to object mode
    """
    ...

@register_pass(mutates_CFG=True, analysis_only=False)
class ExtractByteCode(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """
        Extract bytecode from function
        """
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class TranslateByteCode(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """
        Analyze bytecode and translating to Numba IR
        """
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class RVSDGFrontend(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class FixupArgs(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class IRProcessing(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class RewriteSemanticConstants(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class DeadBranchPrune(SSACompliantMixin, FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
        ...
    
    def get_analysis_usage(self, AU): # -> None:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineClosureLikes(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class GenericRewrites(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """
        Perform any intermediate representation rewrites before type
        inference.
        """
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class WithLifting(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """
        Extract with-contexts
        """
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineInlinables(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.jit decorator directly
    into the site of its call depending on the value set in the 'inline' kwarg
    to the decorator.

    This is an untyped pass. CFG simplification is performed at the end of the
    pass but no block level clean up is performed on the mutated IR (typing
    information is not available to do so).
    """
    _name = ...
    _DEBUG = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        """Run inlining of inlinables
        """
        ...
    


@register_pass(mutates_CFG=False, analysis_only=False)
class PreserveIR(AnalysisPass):
    """
    Preserves the IR in the metadata
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[False]:
        ...
    


@register_pass(mutates_CFG=False, analysis_only=True)
class FindLiterallyCalls(FunctionPass):
    """Find calls to `numba.literally()` and signal if its requirement is not
    satisfied.
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[False]:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class CanonicalizeLoopExit(FunctionPass):
    """A pass to canonicalize loop exit by splitting it from function exit.
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class CanonicalizeLoopEntry(FunctionPass):
    """A pass to canonicalize loop header by splitting it from function entry.

    This is needed for loop-lifting; esp in py3.8
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=False, analysis_only=True)
class PrintIRCFG(FunctionPass):
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[False]:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class MakeFunctionToJitFunction(FunctionPass):
    """
    This swaps an ir.Expr.op == "make_function" i.e. a closure, for a compiled
    function containing the closure body and puts it in ir.Global. It's a 1:1
    statement value swap. `make_function` is already untyped
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class TransformLiteralUnrollConstListToTuple(FunctionPass):
    """ This pass spots a `literal_unroll([<constant values>])` and rewrites it
    as a `literal_unroll(tuple(<constant values>))`.
    """
    _name = ...
    _accepted_types = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class MixedContainerUnroller(FunctionPass):
    _name = ...
    _DEBUG = ...
    _accepted_types = ...
    def __init__(self) -> None:
        ...
    
    def analyse_tuple(self, tup): # -> defaultdict[Any, list[Any]]:
        """
        Returns a map of type->list(indexes) for a typed tuple
        """
        ...
    
    def add_offset_to_labels_w_ignore(self, blocks, offset, ignore=...): # -> dict[Any, Any]:
        """add an offset to all block labels and jump/branch targets
        don't add an offset to anything in the ignore list
        """
        ...
    
    def inject_loop_body(self, switch_ir, loop_ir, caller_max_label, dont_replace, switch_data):
        """
        Injects the "loop body" held in `loop_ir` into `switch_ir` where ever
        there is a statement of the form `SENTINEL.<int> = RHS`. It also:
        * Finds and then deliberately does not relabel non-local jumps so as to
          make the switch table suitable for injection into the IR from which
          the loop body was derived.
        * Looks for `typed_getitem` and wires them up to loop body version
          specific variables or, if possible, directly writes in their constant
          value at their use site.

        Args:
        - switch_ir, the switch table with SENTINELS as generated by
          self.gen_switch
        - loop_ir, the IR of the loop blocks (derived from the original func_ir)
        - caller_max_label, the maximum label in the func_ir caller
        - dont_replace, variables that should not be renamed (to handle
          references to variables that are incoming at the loop head/escaping at
          the loop exit.
        - switch_data, the switch table data used to generated the switch_ir,
          can be generated by self.analyse_tuple.

        Returns:
        - A type specific switch table with each case containing a versioned
          loop body suitable for injection as a replacement for the loop_ir.
        """
        ...
    
    def gen_switch(self, data, index):
        """
        Generates a function with a switch table like
        def foo():
            if PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            elif PLACEHOLDER_INDEX in (<integers>):
                SENTINEL = None
            ...
            else:
                raise RuntimeError

        The data is a map of (type : indexes) for example:
        (int64, int64, float64)
        might give:
        {int64: [0, 1], float64: [2]}

        The index is the index variable for the driving range loop over the
        mixed tuple.
        """
        ...
    
    def apply_transform(self, state): # -> bool:
        ...
    
    def unroll_loop(self, state, loop_info): # -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class IterLoopCanonicalization(FunctionPass):
    """ Transforms loops that are induced by `getiter` into range() driven loops
    If the typemap is available this will only impact Tuple and UniTuple, if it
    is not available it will impact all matching loops.
    """
    _name = ...
    _DEBUG = ...
    _accepted_types = ...
    _accepted_calls = ...
    def __init__(self) -> None:
        ...
    
    def assess_loop(self, loop, func_ir, partial_typemap=...): # -> bool | None:
        ...
    
    def transform(self, loop, func_ir, cfg): # -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=False, analysis_only=False)
class PropagateLiterals(FunctionPass):
    """Implement literal propagation based on partial type inference"""
    _name = ...
    def __init__(self) -> None:
        ...
    
    def get_analysis_usage(self, AU): # -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class LiteralPropagationSubPipelinePass(FunctionPass):
    """Implement literal propagation based on partial type inference"""
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    
    def get_analysis_usage(self, AU): # -> None:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class LiteralUnroll(FunctionPass):
    """Implement the literal_unroll semantics"""
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


@register_pass(mutates_CFG=True, analysis_only=False)
class SimplifyCFG(FunctionPass):
    """Perform CFG simplification"""
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state):
        ...
    


@register_pass(mutates_CFG=False, analysis_only=False)
class ReconstructSSA(FunctionPass):
    """Perform SSA-reconstruction

    Produces minimal SSA.
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> Literal[True]:
        ...
    


@register_pass(mutates_CFG=False, analysis_only=False)
class RewriteDynamicRaises(FunctionPass):
    """Replace existing raise statements by dynamic raises in Numba IR.
    """
    _name = ...
    def __init__(self) -> None:
        ...
    
    def run_pass(self, state): # -> bool:
        ...
    


