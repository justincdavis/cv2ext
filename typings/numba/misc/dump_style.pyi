"""
This type stub file was generated by pyright.
"""

from pygments.lexer import RegexLexer

class NumbaIRLexer(RegexLexer):
    """
    Pygments style lexer for Numba IR (for use with highlighting etc).
    """
    name = ...
    aliases = ...
    filenames = ...
    identifier = ...
    fun_or_var = ...
    tokens = ...


def by_colorscheme():
    """
    Get appropriate style for highlighting according to
    NUMBA_COLOR_SCHEME setting
    """
    ...
