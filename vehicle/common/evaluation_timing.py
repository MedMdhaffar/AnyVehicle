"""Small monotonic timer; synchronization is explicit and testable."""
import time


def measured(call, *, synchronize=lambda:None, clock=time.perf_counter_ns):
    synchronize()
    start=clock()
    try:
        value=call()
    except BaseException as error:
        try:synchronize()
        except BaseException:
            error.evaluation_timing=dict(start_ns=str(start),completion_ns=None,duration_ms=None,
                                         synchronization_status='failed')
            raise error
        end=clock()
        error.evaluation_timing=dict(start_ns=str(start),completion_ns=str(end),duration_ms=(end-start)/1e6)
        raise
    try:synchronize()
    except BaseException as error:
        error.evaluation_timing=dict(start_ns=str(start),completion_ns=None,duration_ms=None,
                                     synchronization_status='failed')
        raise
    end=clock()
    return value,dict(start_ns=str(start),completion_ns=str(end),duration_ms=(end-start)/1e6)
