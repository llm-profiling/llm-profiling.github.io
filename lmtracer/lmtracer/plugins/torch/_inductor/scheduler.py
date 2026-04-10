from unittest.mock import patch
from torch._inductor.scheduler import Scheduler, ExternKernelSchedulerNode

class lmtracerScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode) -> None:
        return super().codegen_extern_call(scheduler_node)

