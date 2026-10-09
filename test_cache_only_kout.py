"""Cache-only contract: Q and KV cache remain correct; Kout stays untouched."""
import pytest
import torch
from test_kernels import make_inputs
from kernels import torch_reference
from half_split_cache_only import launch

@pytest.mark.parametrize("T,Hq,Hkv,D,dtype",[
    (7,10,2,128,torch.float16),
    (4,32,8,96,torch.float32),
    (64,32,8,128,torch.float16),
    (256,64,8,128,torch.float16)
])
def test_cache_only_contract(T,Hq,Hkv,D,dtype):
    q,k,c,s,slots,kcache=make_inputs(T,Hq,Hkv,D,D,dtype,inactive=True)
    v=torch.randn_like(k)
    vcache=torch.full_like(kcache,-7)
    qout=torch.empty_like(q)
    kout=torch.full_like(k,135)
    ref_q,ref_kcache=torch_reference(q,k,c,s,slots,kcache,D)
    ref_vcache=vcache.clone()
    valid=slots>=0
    ref_vcache.view(-1,Hkv,D)[slots[valid].long()]=v[valid]
    bh=4 if Hq>=4 else 1
    launch(q,k,v,c,s,slots,qout,kout,kcache,vcache,
           bh=bh,num_warps=4,write_k_out=False)
    torch.cuda.synchronize()
    tol=0.006 if dtype==torch.float16 else 3e-5
    torch.testing.assert_close(qout,ref_q,atol=tol,rtol=tol)
    torch.testing.assert_close(kcache,ref_kcache,atol=tol,rtol=tol)
    torch.testing.assert_close(vcache,ref_vcache,atol=tol,rtol=tol)
    assert torch.all(kout==135).item()
