from pathlib import Path
import difflib
r=Path(__file__).resolve().parents[1];out=r/'artifacts/named-a0-design'
base=(r/'artifacts/arrival-order/source_control.cu').read_text();s=base
old='''        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dstA),"l"(src),"r"(valid?16:0));'''
assert s.count(old)==1
s=s.replace(old,'''        // Copied before the previous iteration's handoff for this one region.
        if(!(kt>=2 && threadIdx.x>=128 && threadIdx.x<256 && j==0))
    '''+old)
helper='''// DESIGN ONLY: not built or CUDA-validated. No production entrypoint.
__device__ __forceinline__ void early_a32_64(const signed char* a,signed char* stage,
                                          int cm,int copy_k,int k) {
    unsigned t=threadIdx.x;
    unsigned logical=t^((t>>3)&3u);
    unsigned row=logical/4,col=(logical%4)*16;
    unsigned dst=__cvta_generic_to_shared(stage+t*16);
    const signed char* src=a+(size_t(cm*BM)+row)*k+copy_k*64+col;
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,16;"
                 :: "r"(dst),"l"(src) : "memory");
    // No added commit: the unchanged stage_copy commits all future copies.
}
'''
point=s.index('template<int Lead,int TileK=64');s=s[:point]+helper+s[point:]
needle='    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'''        if(kt>0) {
            if(kt+1<k/64 && warp>=4 && warp<8) {
                unsigned free_id=1u+unsigned((kt-1)&1);
                asm volatile("barrier.cta.sync %0,192;" :: "r"(free_id) : "memory");
                early_a32_64(a,sa+((kt+1)&1)*BM*TileK,cm,kt+1,k);
            }
            // Same previous-round handoff, now after this one early copy.
            asm volatile("bar.sync %0,384;" :: "r"(own_bar) : "memory");
            own_bar^=2u;
        }
''')
old='''        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");'''
assert s.count(old)==1
s=s.replace(old,old+'''
        // Only read-only participants publish here. W4 is also a writer and
        // contributes exactly once, via next-round sync, never both forms.
        if(kt+2<k/64 && (warp==0 || warp==8)) {
            unsigned free_id=1u+unsigned(kt&1);
            asm volatile("barrier.cta.arrive %0,192;" :: "r"(free_id) : "memory");
        }''')
old='''        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");
        own_bar^=2u;
    }
    __syncthreads();'''
assert s.count(old)==1
s=s.replace(old,'''    }
    // Drain the last original handoff; no K64/K65 copies or free publications.
    asm volatile("bar.sync %0,384;" :: "r"(own_bar) : "memory");
    __syncthreads();''')
(out/'design-only.cu').write_text(s)
(out/'design-only.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='source_control.cu',tofile='design-only.cu')))
