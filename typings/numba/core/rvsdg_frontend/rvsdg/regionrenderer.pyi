"""
This type stub file was generated by pyright.
"""

import abc
from typing import Any
from dataclasses import dataclass
from contextlib import contextmanager
from numba_rvsdg.core.datastructures.basic_block import BasicBlock, RegionBlock
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor

"""
Graph rendering code is hard to test. Error messages from graphviz is not very
useful in debugging. This file defines ``GraphBacking`` class to isolate graph
building for visualization vs actual rendering the graph using graphviz.
In ``GraphBacking``, we can easily verify the graph structure and produce
better errors.

The ``RVSDGRenderer`` class contains logic to convert a RVSDG into
``GraphBacking``. To produce a graphviz output, use ``to_graphviz`` on the
``GraphBacking``.
"""
@dataclass(frozen=True)
class GraphNode:
    """A node in GraphBacking
    """
    kind: str
    parent_regions: tuple[str, ...] = ...
    ports: tuple[str, ...] = ...
    data: dict[str, Any] = ...


@dataclass(frozen=True)
class GraphEdge:
    """An edge in GraphBacking
    """
    src: str
    dst: str
    src_port: str | None = ...
    dst_port: str | None = ...
    headlabel: str | None = ...
    taillabel: str | None = ...
    kind: str | None = ...


@dataclass(frozen=True)
class GraphGroup:
    """A group in GraphBacking.

    Note: this is called a "group" to avoid name collison with "regions" in
    RVSDG and that the word "group" has less meaning as this is does not
    imply any property.
    """
    subgroups: dict[str, GraphGroup]
    nodes: set[str]
    @classmethod
    def make(cls): # -> Self:
        ...
    


class GraphBacking:
    """An ADT for a graph with hierarchical grouping so it is suited for
    representing regionalized flow graphs in SCFG.
    """
    _nodes: dict[str, GraphNode]
    _groups: GraphGroup
    _edges: set[GraphEdge]
    def __init__(self) -> None:
        ...
    
    def add_node(self, name: str, node: GraphNode): # -> None:
        """Add a graph node
        """
        ...
    
    def add_edge(self, src: str, dst: str, **kwargs): # -> None:
        """Add a graph edge
        """
        ...
    
    def verify(self): # -> None:
        """Check graph structure.

        * check for missing nodes
        * check for missing ports
        """
        ...
    
    def render(self, renderer: AbstractRendererBackend): # -> None:
        """Render this graph using the given backend.
        """
        ...
    


@dataclass(frozen=True)
class GraphNodeMaker:
    """Helper for making GraphNode and keep tracks of the hierarchical
    grouping.
    """
    parent_path: tuple[str, ...]
    def subgroup(self, name: str): # -> Self:
        """Start a subgroup with the given name.
        """
        ...
    
    def make_node(self, **kwargs) -> GraphNode:
        """Make a new node
        """
        ...
    


@dataclass(frozen=True)
class GraphBuilder:
    graph: GraphBacking
    node_maker: GraphNodeMaker
    @classmethod
    def make(cls) -> GraphBuilder:
        ...
    


class AbstractRendererBackend(abc.ABC):
    """Base class for all renderer backend.
    """
    @abc.abstractmethod
    def render_node(self, k: str, node: GraphNode): # -> None:
        ...
    
    @abc.abstractmethod
    def render_edge(self, edge: GraphEdge): # -> None:
        ...
    
    @contextmanager
    @abc.abstractmethod
    def render_cluster(self, name: str): # -> None:
        ...
    


class GraphvizRendererBackend(AbstractRendererBackend):
    """The backend for using graphviz to render a GraphBacking.
    """
    def __init__(self, g=...) -> None:
        ...
    
    def render_node(self, k: str, node: GraphNode): # -> None:
        ...
    
    def render_edge(self, edge: GraphEdge): # -> None:
        ...
    
    @contextmanager
    def render_cluster(self, name: str): # -> Generator[Self, Any, None]:
        ...
    


class RVSDGRenderer(RegionVisitor):
    """Convert a RVSDG into a GraphBacking
    """
    def visit_block(self, block: BasicBlock, builder: GraphBuilder): # -> GraphBuilder:
        ...
    
    def visit_linear(self, region: RegionBlock, builder: GraphBuilder): # -> GraphBuilder:
        ...
    
    def visit_graph(self, scfg: SCFG, builder):
        """Overriding"""
        ...
    
    def visit_loop(self, region: RegionBlock, builder: GraphBuilder): # -> GraphBuilder:
        ...
    
    def visit_switch(self, region: RegionBlock, builder: GraphBuilder): # -> GraphBuilder:
        ...
    
    def render(self, rvsdg: SCFG) -> GraphBacking:
        """Render a RVSDG into a GraphBacking
        """
        ...
    


def to_graphviz(graph: GraphBacking):
    """Render a GraphBacking using graphviz
    """
    ...

