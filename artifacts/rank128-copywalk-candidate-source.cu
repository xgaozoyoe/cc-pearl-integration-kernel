#include <cuda_runtime.h>
#include <cstdint>
namespace rank128_copywalk_unroll_baseline {
constexpr int BM=256, BN=192, THREADS=384;
constexpr int WORDS=(BM/16)*(BN/16)*16;
struct Fragment { unsigned v[4]; };

template<int Layout=0>
__device__ __forceinline__ unsigned swizzle(unsigned p) {
    static_assert(Layout>=0 && Layout<=6);
    // Layout5 keeps this original swizzle, but permutes producer lanes.
    if constexpr(Layout==1) return p; // Control: no swizzle.
    if constexpr(Layout==2) return p ^ ((p >> 3) & 0x70u);
    if constexpr(Layout==3) return p ^ ((p >> 2) & 0x30u);
    if constexpr(Layout==4) return p ^ ((p >> 2) & 0x70u);
    return p ^ ((p >> 3) & 0x30u);
}
template<int TileK,bool Planar,int Layout=0>
__device__ __forceinline__ unsigned offset(int row,int col,int height) {
    unsigned p=Planar ? (col/32)*height*32+row*32+col%32 : row*TileK+col;
    return swizzle<Layout>(p);
}
template<int TileK,bool Planar,int Layout=0>
__device__ __forceinline__ Fragment load16x32(const signed char* base,int row,int col,int lane,int height) {
    // Four 8x8 matrices of 16-bit pairs. Registers are low-row/low-K,
    // high-row/low-K, low-row/high-K, high-row/high-K respectively.
    unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(row+(lane&15),col+(lane>>4)*16,height));
    Fragment x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
    return x;
}
template<int TileK,bool Planar,bool Packed,int Layout=0>
__device__ __forceinline__ Fragment loadB16x32(const signed char* base,int row,int col,int lane,int height) {
    if constexpr(!Packed) return load16x32<TileK,Planar,Layout>(base,row,col,lane,height);
    else {
        // Matrix order: Nlow/Klow, Nlow/Khigh, Nhigh/Klow, Nhigh/Khigh.
        // Each adjacent output pair is now a complete MMA B operand.
        unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(
            row+(lane&7)+(lane>>4)*8,col+((lane>>3)&1)*16,height));
        Fragment x;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
        return x;
    }
}
template<bool Packed=false>
__device__ __forceinline__ void mma(unsigned* c,const Fragment& a,const Fragment& b,int half) {
    // An 8-row B half uses registers (0,2) or (1,3).
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.v[0]),"r"(a.v[1]),"r"(a.v[2]),"r"(a.v[3]),
          "r"(b.v[Packed?2*half:half]),"r"(b.v[Packed?2*half+1:half+2]));
}
template<int TileK,bool Planar,bool UnrollCopy=false,bool FixedCopy=false,int Layout=0>
__device__ __forceinline__ void stage_copy(const signed char* a,const signed char* b,
        signed char* sa,signed char* sb,int cm,int cn,int kt,int m,int n,int k) {
    if constexpr(FixedCopy) {
        static_assert(TileK==64 && !Planar);
        // Exact 384-thread assignment: A has 2 full rounds plus 256 lanes;
        // B has 2 full rounds. No dynamically bounded copy loops.
        #pragma unroll
        for(int j=0;j<3;++j) {
            if(j<2 || threadIdx.x<256) {
                int i=threadIdx.x+j*THREADS;
                int row=i/4,col=(i%4)*16,gr=cm*BM+row;
                const signed char* src=a+(size_t)(gr<m?gr:0)*k+kt*64+col;
                unsigned dst=__cvta_generic_to_shared(sa+swizzle<Layout>(row*64+col));
                int valid=gr<m?16:0;
                asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
            }
        }
        #pragma unroll
        for(int j=0;j<2;++j) {
            int i=threadIdx.x+j*THREADS;
            int row=i/4,col=(i%4)*16,gr=cn*BN+row;
            const signed char* src=b+(size_t)(gr<n?gr:0)*k+kt*64+col;
            unsigned dst=__cvta_generic_to_shared(sb+swizzle<Layout>(row*64+col));
            int valid=gr<n?16:0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
        }
        asm volatile("cp.async.commit_group;");
        return;
    }
    #pragma unroll (UnrollCopy ? 3 : 1)
    for(int i=threadIdx.x;i<BM*TileK/16;i+=(UnrollCopy ? THREADS : blockDim.x)) {
        // Invert the baseline chunk swizzle inside each producer warp.
        // This leaves the warp's global address set unchanged, but makes
        // adjacent producer lanes write adjacent shared 16-byte chunks.
        int logical=i;
        if constexpr(Layout>=5) logical=i^((i>>3)&3);
        int row=logical/(TileK/16),col=(logical%(TileK/16))*16,gr=cm*BM+row;
        int gc=kt*TileK+col;
        bool valid_row=gr<m && (TileK==64 || gc<k);
        const signed char* src;
        if constexpr(TileK==64) src=a+(size_t)(gr<m?gr:0)*k+gc;
        else src=a+(size_t)(valid_row?gr:0)*k+(valid_row?gc:0);
        unsigned dst;
        if constexpr(Layout==6) dst=__cvta_generic_to_shared(sa+i*16);
        else dst=__cvta_generic_to_shared(sa+offset<TileK,Planar,Layout>(row,col,BM));
        int valid=valid_row?16:0;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
    }
    #pragma unroll (UnrollCopy ? 2 : 1)
    for(int i=threadIdx.x;i<BN*TileK/16;i+=(UnrollCopy ? THREADS : blockDim.x)) {
        // Invert the baseline chunk swizzle inside each producer warp.
        // This leaves the warp's global address set unchanged, but makes
        // adjacent producer lanes write adjacent shared 16-byte chunks.
        int logical=i;
        if constexpr(Layout>=5) logical=i^((i>>3)&3);
        int row=logical/(TileK/16),col=(logical%(TileK/16))*16,gr=cn*BN+row;
        int gc=kt*TileK+col;
        bool valid_row=gr<n && (TileK==64 || gc<k);
        const signed char* src;
        if constexpr(TileK==64) src=b+(size_t)(gr<n?gr:0)*k+gc;
        else src=b+(size_t)(valid_row?gr:0)*k+(valid_row?gc:0);
        unsigned dst;
        if constexpr(Layout==6) dst=__cvta_generic_to_shared(sb+i*16);
        else dst=__cvta_generic_to_shared(sb+offset<TileK,Planar,Layout>(row,col,BN));
        int valid=valid_row?16:0;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
    }
    asm volatile("cp.async.commit_group;");
}

template<int Lead,int TileK=64,bool Planar=false,bool PackedB=false,bool UnrollCopy=false,bool FixedCopy=false,int Stages=2,bool Wide=false,int Layout=0>
__global__ void __launch_bounds__(Wide?256:THREADS,1) kernel(const signed char* a,const signed char* b,int m,int n,int k,int r,unsigned* out) {
    constexpr int WarpN=Wide?96:64, Groups=WarpN/16, Threads=Wide?256:THREADS;
    // Host launch rejects every K except4096. Retain dynamic-K code in
    // unpacked controls, specialize only the packed-B experiment.
    if constexpr(PackedB) k=4096;
    extern __shared__ signed char shared[];
    signed char* sa=shared;
    signed char* sb=shared+Stages*BM*TileK;
    unsigned* transcript=reinterpret_cast<unsigned*>(shared+Stages*(BM+BN)*TileK);
    int linear=blockIdx.x+gridDim.x*blockIdx.y;
    int first=linear/(8*gridDim.x)*8;
    int cols=min(8,int(gridDim.y)-first), offset=linear%(8*gridDim.x);
    int cm=offset/cols,cn=first+offset%cols;
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    int wm=warp%4,wn=warp/4;
    // Native MMA lane ownership: row=lane/4 (+8 for c2,c3),
    // column=2*(lane%4) (+1 for c1,c3). Two halves form a16x16 tile.
    unsigned c[4][Groups][2][4]={};
    Fragment ar[4], br[Lead+1];
    #pragma unroll
    for(int p=0;p<Stages-1;++p)
        stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+p*BM*TileK,sb+p*BN*TileK,cm,cn,p,m,n,k);
    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {
        if constexpr(Stages==2) asm volatile("cp.async.wait_group 0;");
        else {
            if(kt+Stages-1<(k+TileK-1)/TileK) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int current=kt%Stages,next=(kt+Stages-1)%Stages;
        if(kt+Stages-1<(k+TileK-1)/TileK)
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);
        signed char* aa=sa+current*BM*TileK;
        signed char* bb=sb+current*BN*TileK;
        if constexpr(Lead==0 || Lead==3) {
            #pragma unroll
            for(int slice=0;slice<2;++slice) {
                Fragment aa_rows[4],bb_groups[2];
                #pragma unroll
                for(int row=0;row<4;++row) aa_rows[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,slice*32,lane,BM);
                if constexpr(Lead==3) bb_groups[0]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN,slice*32,lane,BN);
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    int slot=group%2;
                    if constexpr(Lead==0) bb_groups[slot]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+group*16,slice*32,lane,BN);
                    else if(group+1<Groups) bb_groups[(group+1)%2]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+(group+1)*16,slice*32,lane,BN);
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int row=0;row<4;++row) mma<PackedB>(c[row][group][half],aa_rows[row],bb_groups[slot],half);
                    }
                }
            }
        } else {
        #pragma unroll
        for(int row=0;row<4;++row) ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,0,lane,BM);
        #pragma unroll
        for(int p=0;p<Lead;++p) br[p]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+p*16,0,lane,BN);
        #pragma unroll
        for(int g=0;g<Groups*(TileK/32);++g) {
            constexpr int slots=Lead+1;
            int slice=g/Groups,group=g%Groups,slot=g%slots;
            if constexpr(TileK!=64) { if(kt*TileK+slice*32>=k) continue; }
            #pragma unroll
            for(int half=0;half<2;++half) {
                #pragma unroll
                for(int row=0;row<4;++row) {
                    if(half==0 && row==2 && g+Lead<Groups*(TileK/32) && (TileK==64 || kt*TileK+((g+Lead)/Groups)*32<k))
                        br[(g+Lead)%slots]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN);
                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);
                    if(slice+1<TileK/32 && (TileK==64 || kt*TileK+(slice+1)*32<k) && group==Groups-1 && half==1)
                        ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM);
                }
            }
            if constexpr(TileK!=64) {
        if(group==Groups-1 && (kt*TileK+(slice+1)*32)%128==0) {
            int rb=(kt*TileK+(slice+1)*32)/128-1;
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb%16);
                        if(rb<16) transcript[idx]=x;
                        else { unsigned prev=transcript[idx]; transcript[idx]=((prev<<13)|(prev>>19))^x; }
                    }
                }
            }
        }
            }
        }
        }
        if constexpr(TileK==64) {
        if((kt+1)%2==0) {
            int rb=(kt+1)/2-1;
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb%16);
                        if(rb<16) transcript[idx]=x;
                        else { unsigned prev=transcript[idx]; transcript[idx]=((prev<<13)|(prev>>19))^x; }
                    }
                }
            }
        }
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<WORDS;i+=Threads) {
        int tile=i/16,gr=cm*(BM/16)+tile/(BN/16),gc=cn*(BN/16)+tile%(BN/16);
        if(gr<m/16 && gc<n/16) out[((size_t)gr*(n/16)+gc)*16+i%16]=transcript[i];
    }
}
template<int Lead,int TileK=64,bool Planar=false,bool PackedB=false,bool UnrollCopy=false,bool FixedCopy=false,int Stages=2,bool Wide=false,int Layout=0>
cudaError_t launch(const int8_t* a,const int8_t* b,int m,int n,int k,int r,
                   uint32_t* t,int* info,cudaStream_t stream) {
    if(m<=0 || n<=0 || m%16 || n%16 || r!=128 || k!=4096)
        return cudaErrorInvalidValue;
    constexpr int Threads=Wide?256:THREADS;
    constexpr int smem=Stages*(BM+BN)*TileK+WORDS*sizeof(unsigned);
    static bool initialized=false; // Isolated benchmarks use device0 only.
    cudaError_t err;
    if(!initialized) {
        err=cudaFuncSetAttribute(kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
        if(err!=cudaSuccess)return err;
        err=cudaFuncSetAttribute(kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
        if(err!=cudaSuccess)return err;
        initialized=true;
    }
    if(info) {
        cudaFuncAttributes attr;err=cudaFuncGetAttributes(&attr,kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>);if(err!=cudaSuccess)return err;
        int active;err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>,Threads,smem);
        if(err!=cudaSuccess)return err;
        info[0]=attr.numRegs;info[1]=Threads;info[2]=smem+attr.sharedSizeBytes;info[3]=active;info[4]=attr.localSizeBytes;
        return cudaSuccess;
    }
    kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout><<<dim3((m+BM-1)/BM,(n+BN-1)/BN),Threads,smem,stream>>>(a,b,m,n,k,r,t);
    return cudaGetLastError();
}
} // namespace rank128_copywalk_unroll_baseline



#include <cuda_runtime.h>
#include <cstdint>
namespace rank128_copywalk_unroll_fast {
constexpr int BM=256, BN=192, THREADS=384;
constexpr int WORDS=(BM/16)*(BN/16)*16;
struct Fragment { unsigned v[4]; };

template<int Layout=0>
__device__ __forceinline__ unsigned swizzle(unsigned p) {
    static_assert(Layout>=0 && Layout<=6);
    // Layout5 keeps this original swizzle, but permutes producer lanes.
    if constexpr(Layout==1) return p; // Control: no swizzle.
    if constexpr(Layout==2) return p ^ ((p >> 3) & 0x70u);
    if constexpr(Layout==3) return p ^ ((p >> 2) & 0x30u);
    if constexpr(Layout==4) return p ^ ((p >> 2) & 0x70u);
    return p ^ ((p >> 3) & 0x30u);
}
template<int TileK,bool Planar,int Layout=0>
__device__ __forceinline__ unsigned offset(int row,int col,int height) {
    unsigned p=Planar ? (col/32)*height*32+row*32+col%32 : row*TileK+col;
    return swizzle<Layout>(p);
}
template<int TileK,bool Planar,int Layout=0>
__device__ __forceinline__ Fragment load16x32(const signed char* base,int row,int col,int lane,int height) {
    // Four 8x8 matrices of 16-bit pairs. Registers are low-row/low-K,
    // high-row/low-K, low-row/high-K, high-row/high-K respectively.
    unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(row+(lane&15),col+(lane>>4)*16,height));
    Fragment x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
    return x;
}
template<int TileK,bool Planar,bool Packed,int Layout=0>
__device__ __forceinline__ Fragment loadB16x32(const signed char* base,int row,int col,int lane,int height) {
    if constexpr(!Packed) return load16x32<TileK,Planar,Layout>(base,row,col,lane,height);
    else {
        // Matrix order: Nlow/Klow, Nlow/Khigh, Nhigh/Klow, Nhigh/Khigh.
        // Each adjacent output pair is now a complete MMA B operand.
        unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(
            row+(lane&7)+(lane>>4)*8,col+((lane>>3)&1)*16,height));
        Fragment x;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
        return x;
    }
}
template<bool Packed=false>
__device__ __forceinline__ void mma(unsigned* c,const Fragment& a,const Fragment& b,int half) {
    // An 8-row B half uses registers (0,2) or (1,3).
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.v[0]),"r"(a.v[1]),"r"(a.v[2]),"r"(a.v[3]),
          "r"(b.v[Packed?2*half:half]),"r"(b.v[Packed?2*half+1:half+2]));
}
template<int TileK,bool Planar,bool UnrollCopy=false,bool FixedCopy=false,int Layout=0>
__device__ __forceinline__ void stage_copy(const signed char* a,const signed char* b,
        signed char* sa,signed char* sb,int cm,int cn,int kt,int m,int n,int k) {
    if constexpr(FixedCopy) {
        static_assert(TileK==64 && !Planar);
        // Exact 384-thread assignment: A has 2 full rounds plus 256 lanes;
        // B has 2 full rounds. No dynamically bounded copy loops.
        #pragma unroll
        for(int j=0;j<3;++j) {
            if(j<2 || threadIdx.x<256) {
                int i=threadIdx.x+j*THREADS;
                int row=i/4,col=(i%4)*16,gr=cm*BM+row;
                const signed char* src=a+(size_t)(gr<m?gr:0)*k+kt*64+col;
                unsigned dst=__cvta_generic_to_shared(sa+swizzle<Layout>(row*64+col));
                int valid=gr<m?16:0;
                asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
            }
        }
        #pragma unroll
        for(int j=0;j<2;++j) {
            int i=threadIdx.x+j*THREADS;
            int row=i/4,col=(i%4)*16,gr=cn*BN+row;
            const signed char* src=b+(size_t)(gr<n?gr:0)*k+kt*64+col;
            unsigned dst=__cvta_generic_to_shared(sb+swizzle<Layout>(row*64+col));
            int valid=gr<n?16:0;
            asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
        }
        asm volatile("cp.async.commit_group;");
        return;
    }

    static_assert(TileK==64 && Layout==6 && !Planar);
    // 384 is a multiple of128: the swizzled low7 chunk bits never change.
    int logical=int(threadIdx.x)^(int(threadIdx.x>>3)&3);
    int row=logical/4,col=(logical%4)*16;
    unsigned dstA=__cvta_generic_to_shared(sa+threadIdx.x*16);
    unsigned dstB=__cvta_generic_to_shared(sb+threadIdx.x*16);
    int grA=cm*BM+row,grB=cn*BN+row;
    unsigned offA=unsigned(grA)*unsigned(k)+unsigned(kt*64+col);
    unsigned offB=unsigned(grB)*unsigned(k)+unsigned(kt*64+col);
    #pragma unroll
    for(int j=0;j<3;++j) if(threadIdx.x+j*THREADS<BM*64/16) {
        bool valid=grA<m;
        const signed char* src=a+(valid?offA:0u);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dstA),"l"(src),"r"(valid?16:0));
        grA+=THREADS/4;offA+=unsigned(THREADS/4)*unsigned(k);dstA+=THREADS*16;
    }
    #pragma unroll
    for(int j=0;j<2;++j) {
        bool valid=grB<n;
        const signed char* src=b+(valid?offB:0u);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dstB),"l"(src),"r"(valid?16:0));
        grB+=THREADS/4;offB+=unsigned(THREADS/4)*unsigned(k);dstB+=THREADS*16;
    }
    asm volatile("cp.async.commit_group;");
}

template<int Lead,int TileK=64,bool Planar=false,bool PackedB=false,bool UnrollCopy=false,bool FixedCopy=false,int Stages=2,bool Wide=false,int Layout=0>
__global__ void __launch_bounds__(Wide?256:THREADS,1) kernel(const signed char* a,const signed char* b,int m,int n,int k,int r,unsigned* out) {
    constexpr int WarpN=Wide?96:64, Groups=WarpN/16, Threads=Wide?256:THREADS;
    // Host launch rejects every K except4096. Retain dynamic-K code in
    // unpacked controls, specialize only the packed-B experiment.
    if constexpr(PackedB) k=4096;
    extern __shared__ signed char shared[];
    signed char* sa=shared;
    signed char* sb=shared+Stages*BM*TileK;
    unsigned* transcript=reinterpret_cast<unsigned*>(shared+Stages*(BM+BN)*TileK);
    int linear=blockIdx.x+gridDim.x*blockIdx.y;
    int first=linear/(8*gridDim.x)*8;
    int cols=min(8,int(gridDim.y)-first), offset=linear%(8*gridDim.x);
    int cm=offset/cols,cn=first+offset%cols;
    int warp=threadIdx.x/32,lane=threadIdx.x%32;
    int wm=warp%4,wn=warp/4;
    // Native MMA lane ownership: row=lane/4 (+8 for c2,c3),
    // column=2*(lane%4) (+1 for c1,c3). Two halves form a16x16 tile.
    unsigned c[4][Groups][2][4]={};
    Fragment ar[4], br[Lead+1];
    #pragma unroll
    for(int p=0;p<Stages-1;++p)
        stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+p*BM*TileK,sb+p*BN*TileK,cm,cn,p,m,n,k);
    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {
        if constexpr(Stages==2) asm volatile("cp.async.wait_group 0;");
        else {
            if(kt+Stages-1<(k+TileK-1)/TileK) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int current=kt%Stages,next=(kt+Stages-1)%Stages;
        if(kt+Stages-1<(k+TileK-1)/TileK)
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);
        signed char* aa=sa+current*BM*TileK;
        signed char* bb=sb+current*BN*TileK;
        if constexpr(Lead==0 || Lead==3) {
            #pragma unroll
            for(int slice=0;slice<2;++slice) {
                Fragment aa_rows[4],bb_groups[2];
                #pragma unroll
                for(int row=0;row<4;++row) aa_rows[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,slice*32,lane,BM);
                if constexpr(Lead==3) bb_groups[0]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN,slice*32,lane,BN);
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    int slot=group%2;
                    if constexpr(Lead==0) bb_groups[slot]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+group*16,slice*32,lane,BN);
                    else if(group+1<Groups) bb_groups[(group+1)%2]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+(group+1)*16,slice*32,lane,BN);
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int row=0;row<4;++row) mma<PackedB>(c[row][group][half],aa_rows[row],bb_groups[slot],half);
                    }
                }
            }
        } else {
        #pragma unroll
        for(int row=0;row<4;++row) ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,0,lane,BM);
        #pragma unroll
        for(int p=0;p<Lead;++p) br[p]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+p*16,0,lane,BN);
        #pragma unroll
        for(int g=0;g<Groups*(TileK/32);++g) {
            constexpr int slots=Lead+1;
            int slice=g/Groups,group=g%Groups,slot=g%slots;
            if constexpr(TileK!=64) { if(kt*TileK+slice*32>=k) continue; }
            #pragma unroll
            for(int half=0;half<2;++half) {
                #pragma unroll
                for(int row=0;row<4;++row) {
                    if(half==0 && row==2 && g+Lead<Groups*(TileK/32) && (TileK==64 || kt*TileK+((g+Lead)/Groups)*32<k))
                        br[(g+Lead)%slots]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN);
                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);
                    if(slice+1<TileK/32 && (TileK==64 || kt*TileK+(slice+1)*32<k) && group==Groups-1 && half==1)
                        ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM);
                }
            }
            if constexpr(TileK!=64) {
        if(group==Groups-1 && (kt*TileK+(slice+1)*32)%128==0) {
            int rb=(kt*TileK+(slice+1)*32)/128-1;
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb%16);
                        if(rb<16) transcript[idx]=x;
                        else { unsigned prev=transcript[idx]; transcript[idx]=((prev<<13)|(prev>>19))^x; }
                    }
                }
            }
        }
            }
        }
        }
        if constexpr(TileK==64) {
        if((kt+1)%2==0) {
            int rb=(kt+1)/2-1;
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) {
                        int idx=((wm*4+row)*(BN/16)+wn*Groups+group)*16+(rb%16);
                        if(rb<16) transcript[idx]=x;
                        else { unsigned prev=transcript[idx]; transcript[idx]=((prev<<13)|(prev>>19))^x; }
                    }
                }
            }
        }
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<WORDS;i+=Threads) {
        int tile=i/16,gr=cm*(BM/16)+tile/(BN/16),gc=cn*(BN/16)+tile%(BN/16);
        if(gr<m/16 && gc<n/16) out[((size_t)gr*(n/16)+gc)*16+i%16]=transcript[i];
    }
}
template<int Lead,int TileK=64,bool Planar=false,bool PackedB=false,bool UnrollCopy=false,bool FixedCopy=false,int Stages=2,bool Wide=false,int Layout=0>
cudaError_t launch(const int8_t* a,const int8_t* b,int m,int n,int k,int r,
                   uint32_t* t,int* info,cudaStream_t stream) {
    if(m<=0 || n<=0 || m%16 || n%16 || r!=128 || k!=4096)
        return cudaErrorInvalidValue;
    constexpr int Threads=Wide?256:THREADS;
    constexpr int smem=Stages*(BM+BN)*TileK+WORDS*sizeof(unsigned);
    static bool initialized=false; // Isolated benchmarks use device0 only.
    cudaError_t err;
    if(!initialized) {
        err=cudaFuncSetAttribute(kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
        if(err!=cudaSuccess)return err;
        err=cudaFuncSetAttribute(kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
        if(err!=cudaSuccess)return err;
        initialized=true;
    }
    if(info) {
        cudaFuncAttributes attr;err=cudaFuncGetAttributes(&attr,kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>);if(err!=cudaSuccess)return err;
        int active;err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout>,Threads,smem);
        if(err!=cudaSuccess)return err;
        info[0]=attr.numRegs;info[1]=Threads;info[2]=smem+attr.sharedSizeBytes;info[3]=active;info[4]=attr.localSizeBytes;
        return cudaSuccess;
    }
    kernel<Lead,TileK,Planar,PackedB,UnrollCopy,FixedCopy,Stages,Wide,Layout><<<dim3((m+BM-1)/BM,(n+BN-1)/BN),Threads,smem,stream>>>(a,b,m,n,k,r,t);
    return cudaGetLastError();
}
} // namespace rank128_copywalk_unroll_fast




void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int copywalk_unroll_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_copywalk_unroll_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1 && (m>1048576 || n>1048576))return rank128_copywalk_unroll_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1)return rank128_copywalk_unroll_fast::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int copywalk_unroll_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=copywalk_unroll_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}

// Explicit mining ABI. Only validated R128 is supported by this trial library.
extern "C" int rank_fold(int r,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(r!=128)return cudaErrorInvalidValue;
 return copywalk_unroll_fold(1,a,b,m,n,t,info);
}
extern "C" int rank_pow(int r,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 if(r!=128)return cudaErrorInvalidValue;
 return copywalk_unroll_pow(1,a,b,m,n,key,target,t,dig,found,coord);
}
