#include "matmul.h"
#include "softmax.h"
#include "validation.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>

void require(bool condition) { if (!condition) throw std::runtime_error("check failed"); }
template<class F> void invalid(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument");
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) return 2;
        const std::string group = argv[1];
        if (group == "comparator") {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            const float inf = std::numeric_limits<float>::infinity();
            require(validation::close({1}, {1.0001f}, .001));
            require(validation::close({1000}, {1001}, 0, .002));
            require(!validation::close({1}, {nan}, .001));
            require(!validation::close({nan}, {nan}, .001));
            require(!validation::close({inf}, {inf}, .001));
            require(!validation::close({1}, {2}, .001));
            require(!validation::close({1}, {}, .001));
            require(!validation::close({1}, {1}, -1));
            require(!validation::close({1}, {1}, nan));
        } else if (group == "shape_contract") {
            validation::softmax_shape(1, 2048, true);
            validation::softmax_shape(1, 2049, false);
            invalid([] { validation::softmax_shape(1, 2049, true); });
            invalid([] { validation::softmax_shape(0, 1, false); });
            invalid([] { validation::softmax_shape(1, 0, true); });
            invalid([] { validation::elements(std::numeric_limits<std::size_t>::max(), 2); });
            invalid([] { matmul_cpu({}, {}, 0, 0, 0); });
        } else if (group == "gemm_cpu") {
            require(validation::close(matmul_cpu({1,2,3,4,5,6}, {7,8,9,10,11,12}, 2,3,2),
                                      {58,64,139,154}, 0));
            for (auto [m,k,n] : {std::tuple{1,1,1}, {3,5,7}, {17,19,33}}) {
                std::vector<float> a(m*k), b(k*n);
                for (int i=0;i<m*k;++i) a[i]=(i%7-3)*.25f;
                for (int i=0;i<k*n;++i) b[i]=(i%5-2)*.5f;
                auto c=matmul_cpu(a,b,m,k,n);
                for (int i=0;i<m;++i) for(int j=0;j<n;++j) {
                    double expected=0;
                    for(int x=0;x<k;++x) expected+=double(a[i*k+x])*b[x*n+j];
                    require(std::abs(c[i*n+j]-expected)<1e-5);
                }
            }
            invalid([] { matmul_cpu({1},{1},2,1,1); });
        } else if (group == "softmax_cpu") {
            for (std::size_t cols : {1,3,31,257,2048,2049}) {
                std::vector<float> input(2*cols);
                for(std::size_t i=0;i<input.size();++i) input[i]=float(int(i%17)-8)*100;
                auto output=softmax_cpu(input,2,cols);
                for(std::size_t row=0;row<2;++row) {
                    double sum=0;
                    for(std::size_t col=0;col<cols;++col) {
                        float value=output[row*cols+col];
                        require(std::isfinite(value) && value>=0 && value<=1); sum+=value;
                    }
                    require(std::abs(sum-1)<1e-5);
                }
                for(float& value:input) value+=5000;
                require(validation::close(output,softmax_cpu(input,2,cols),1e-6));
            }
            invalid([] { softmax_cpu({std::numeric_limits<float>::quiet_NaN()},1,1); });
            invalid([] { softmax_cpu({std::numeric_limits<float>::infinity()},1,1); });
            invalid([] { softmax_cpu({},1,1); });
        } else return 2;
        std::cout << group << " PASS\n";
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
