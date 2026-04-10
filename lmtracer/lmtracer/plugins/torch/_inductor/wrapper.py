import sympy
from sympy import Expr
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SubgraphPythonWrapperCodegen
from torch._inductor.virtualized import V
from torch._inductor.scheduler import ExternKernelSchedulerNode
from torch._inductor import ir
from torch._dynamo.utils import counters
from torch._inductor.codegen.common import Kernel
from typing import Any, Callable, Optional, Sequence, Union
# from vllm.utils import init_logger
from torch._inductor.utils import cache_on_self, IndentedBuffer
from torch._inductor.ir import IRNode
from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode
import torch
from torch.utils._ordered_set import OrderedSet
import torch.utils._pytree as pytree
BufferLike = Union[ir.Buffer, WorkspaceArg]

# logger = init_logger(__name__)

TritonMetaParams = dict[str, int]

class LMTracerSubgraphPythonWrapperCodegen(SubgraphPythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LMTracerPythonWrapperCodegen(PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return LMTracerSubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return LMTracerPythonWrapperCodegen()

    
    def write_header(self):
        super().write_header()
        # self.header.writeline('from lmtracer.plugins.vllm.compilation.backends import clock_tensor_1, clock_tensor_2')
        self.header.writeline('from lmtracer.cuda import timer_kernel')
    
    