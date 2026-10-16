#include "matmul.h"
#include "matmul_cuda.h"
#include "softmax.h"
#include "softmax_cuda.h"
#include "validation.h"
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    float* data{};
    explicit Device(std::size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(float))); }
    ~Device() { cudaFree(data); }
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
int main() {
    try {
        for(auto [m,k,n] : {std::tuple{1,1,1}, {3,5,7}, {17,19,33}, {37,19,43}}) {
            std::vector<float> a(m*k),b(k*n),out(m*n);
            for(int i=0;i<m*k;++i) a[i]=(i%7-3)*.25f;
            for(int i=0;i<k*n;++i) b[i]=(i%5-2)*.5f;
            auto reference=matmul_cpu(a,b,m,k,n);
            Device da(a.size()),db(b.size()),dc(out.size());
            check(cudaMemcpy(da.data,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
            check(cudaMemcpy(db.data,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
            for(auto function : {matmul_cuda_naive,matmul_cuda_tiled,matmul_cuda_register_tiled}) {
                function(da.data,db.data,dc.data,m,k,n); check(cudaDeviceSynchronize());
                check(cudaMemcpy(out.data(),dc.data,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(!validation::close(reference,out,1e-3,1e-5)) throw std::runtime_error("GEMM mismatch");
            }
        }
        for(std::size_t cols : {1,3,31,257,2048,2049}) {
            std::vector<float> input(2*cols),out(input.size());
            for(std::size_t i=0;i<input.size();++i) input[i]=float(int(i%17)-8)*100;
            auto reference=softmax_cpu(input,2,cols);
            Device di(input.size()),dout(input.size());
            check(cudaMemcpy(di.data,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
            for(auto function : {softmax_cuda_naive,softmax_cuda_register_cached,softmax_cuda_warp_shuffle}) {
                if(cols>2048 && function!=softmax_cuda_naive) {
                    bool rejected=false;
                    try { function(di.data,dout.data,2,cols); } catch(const std::invalid_argument&) { rejected=true; }
                    if(!rejected) throw std::runtime_error("missing width guard");
                    continue;
                }
                function(di.data,dout.data,2,cols);check(cudaDeviceSynchronize());
                check(cudaMemcpy(out.data(),dout.data,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
                if(!validation::close(reference,out,1e-4,1e-5)) throw std::runtime_error("Softmax mismatch");
            }
        }
        std::cout << "GPU shape and numeric regressions PASS\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
