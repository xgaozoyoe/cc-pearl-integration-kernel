// Diagnostic-only wrapper around the existing experimental Ada kernel.
#define PEARL_PURE_BENCH_ONLY
#include "fold_ops.cu"
#include <vector>
namespace {
std::vector<cudaEvent_t> starts, ends;
std::vector<cudaGraphExec_t> graphs[2];
cudaStream_t capture_stream;
bool measuring=false;
int position=0;
}
extern "C" int measure_init(int count) {
    starts.resize(count);ends.resize(count);
    graphs[0].resize(count);graphs[1].resize(count);
    auto e=cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking);
    if(e!=cudaSuccess)return e;
    for(int i=0;i<count;++i) {
        auto e=cudaEventCreate(&starts[i]);if(e!=cudaSuccess)return e;
        e=cudaEventCreate(&ends[i]);if(e!=cudaSuccess)return e;
    }
    return cudaSuccess;
}
extern "C" int measure_begin() {position=0;measuring=true;return cudaSuccess;}
extern "C" int measure_end(float* sum_ms) {
    measuring=false;*sum_ms=0;
    if(position!=int(starts.size()))return cudaErrorInvalidValue;
    for(int i=0;i<position;++i) {
        float ms;auto e=cudaEventElapsedTime(&ms,starts[i],ends[i]);
        if(e!=cudaSuccess)return e;*sum_ms+=ms;
    }
    return cudaSuccess;
}
extern "C" int measure_close() {
    for(auto& group:graphs)for(auto g:group)if(g)cudaGraphExecDestroy(g);
    cudaStreamDestroy(capture_stream);
    for(auto e:starts)cudaEventDestroy(e);
    for(auto e:ends)cudaEventDestroy(e);
    starts.clear();ends.clear();return cudaSuccess;
}
extern "C" int trial_operator(int id,const int8_t* a,const int8_t* b,
                             int m,int n,int k,int r,uint32_t* sink) {
    if(id!=20 && id!=48)return cudaErrorInvalidValue;
    if(!measuring)return dispatch(id,a,b,m,n,k,r,sink);
    if(position>=int(starts.size()))return cudaErrorInvalidValue;
    auto& exec=graphs[id==48][position];
    if(!exec) {
        auto e=cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal);
        if(e!=cudaSuccess)return e;
        e=cudaEventRecordWithFlags(starts[position],capture_stream,cudaEventRecordExternal);
        if(e!=cudaSuccess)return e;
        e=dispatch(id,a,b,m,n,k,r,sink,nullptr,capture_stream);
        if(e!=cudaSuccess)return e;
        e=cudaEventRecordWithFlags(ends[position],capture_stream,cudaEventRecordExternal);
        if(e!=cudaSuccess)return e;
        cudaGraph_t graph;e=cudaStreamEndCapture(capture_stream,&graph);
        if(e!=cudaSuccess)return e;
        e=cudaGraphInstantiate(&exec,graph,0);cudaGraphDestroy(graph);
        if(e!=cudaSuccess)return e;
    }
    auto e=cudaGraphLaunch(exec,nullptr);++position;return e;
}
extern "C" int benchmark_device_info(int* out) {
    cudaDeviceProp p;auto e=cudaGetDeviceProperties(&p,0);if(e!=cudaSuccess)return e;
    out[0]=p.multiProcessorCount;out[1]=p.maxThreadsPerMultiProcessor/32;
    out[2]=p.warpSize;out[3]=p.regsPerMultiprocessor;
    return cudaSuccess;
}
extern "C" int benchmark_lane_map(int* out) {
    using namespace cute;
    using Atom=MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using NP=Layout<Shape<_8,_4,_2>,Stride<_1,_16,_8>>;
    auto mma=make_tiled_mma(Atom{},Layout<Shape<_2,_4,_1>>{},make_tile(_,NP{},_));
    auto identity=make_identity_tensor(make_shape(_128{},_256{}));
    for(int tid=0;tid<256;++tid) {
        auto coords=mma.get_slice(tid).partition_C(identity);
        for(int tm=0;tm<size<1>(coords);++tm)
        for(int tn=0;tn<size<2>(coords);++tn)
        for(int v=0;v<size<0>(coords);++v) {
            auto mn=coords(v,tm,tn);
            int ix=((tid*4+tm)*8+tn)*4+v;
            out[2*ix]=get<0>(mn);out[2*ix+1]=get<1>(mn);
        }
    }
    return cudaSuccess;
}
