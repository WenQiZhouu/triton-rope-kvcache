"""Correctness regression tests for BH Tile + conservative dispatch."""
import pytest
import torch
from test_kernels import make_inputs
from kernels import torch_reference
from head_tiling_dispatch import launch,select_config

@pytest.mark.parametrize("T,Hq,Hkv,D,dtype,expected",[
    (1,32,8,128,torch.float16,(1,4)),
    (64,32,8,128,torch.float16,(4,4)),
    (256,64,8,128,torch.float16,(4,1)),
    (32,16,4,128,torch.float16,(1,4)),  # untested shape fallback
    (64,32,8,128,torch.float32,(1,4)), # untested dtype fallback
])
def test_dispatch_correctness(T,Hq,Hkv,D,dtype,expected):
    q,k,cos,sin,slots,kcache=make_inputs(T,Hq,Hkv,D,D,dtype,inactive=True)
    v=torch.randn_like(k)
    vc=torch.full_like(kcache,-7.)
    qout=torch.empty_like(q)
    kout=torch.empty_like(k)
    refq,refkc=torch_reference(q,k,cos,sin,slots,kcache,D)
    refko,_=torch_reference(k,k,cos,sin,slots,kcache,D)
    refvc=vc.clone()
    valid=slots>=0
    refvc.view(-1,Hkv,D)[slots[valid].long()]=v[valid]
    assert select_config(q,k)==expected
    chosen=launch(q,k,v,cos,sin,slots,qout,kout,kcache,vc)
    torch.cuda.synchronize()
    assert chosen==expected
    tol=0.006 if dtype==torch.float16 else 3e-5
    for name,x,y in (('Qout',qout,refq),('Kout',kout,refko),
                     ('Kcache',kcache,refkc),('Vcache',vc,refvc)):
        torch.testing.assert_close(x,y,rtol=tol,atol=tol,msg=name)
