import pytest
import torch
from test_kernels import make_inputs
from kernels import torch_reference
from half_split_dispatch import select_config,launch

@pytest.mark.parametrize('T,Hq,Hkv,D,dtype,expected',[
    (1,32,8,128,torch.float16,('half_split',1,4)),
    (64,32,8,128,torch.float16,('half_split',4,4)),
    (256,64,8,128,torch.float16,('v2_head_tile',2,1)),
    (32,16,4,128,torch.float16,('v1',1,4)),
    (64,32,8,128,torch.float32,('v1',1,4)),
])
def test_dispatch(T,Hq,Hkv,D,dtype,expected):
    q,k,c,s,slots,kc=make_inputs(T,Hq,Hkv,D,D,dtype,inactive=True)
    v=torch.randn_like(k)
    vc=torch.full_like(kc,-7.)
    qo=torch.empty_like(q)
    ko=torch.empty_like(k)
    refq,refkc=torch_reference(q,k,c,s,slots,kc,D)
    refk,_=torch_reference(k,k,c,s,slots,kc,D)
    refvc=vc.clone()
    active=slots>=0
    refvc.view(-1,Hkv,D)[slots[active].long()]=v[active]
    assert select_config(q,k)==expected
    assert launch(q,k,v,c,s,slots,qo,ko,kc,vc)==expected
    torch.cuda.synchronize()
    tol=0.006 if dtype==torch.float16 else 3e-5
    for x,y in ((qo,refq),(ko,refk),(kc,refkc),(vc,refvc)):
        torch.testing.assert_close(x,y,rtol=tol,atol=tol)
