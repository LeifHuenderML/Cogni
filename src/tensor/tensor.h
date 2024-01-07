#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>
#include <cassert>
#include <numeric>

namespace nn {
template<typename T>

class Tensor {
private:
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    size_t computeTotalSize(const std::vector<size_t>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    }

    void calculateStrides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    size_t index(const std::initializer_list<size_t>& idxs) const {
        assert(idxs.size() == shape.size());
        size_t idx = 0;
        size_t i = 0;
        for (auto ix : idxs) {
            idx += ix * strides[i++];
        }
        return idx;
    }

public:
    Tensor(){};

    Tensor(const std::initializer_list<size_t>& dims) : shape(dims), data(computeTotalSize(dims), T()) {
        calculateStrides();
    }

    T& operator()(const std::initializer_list<size_t>& idxs) {
        return data[index(idxs)];
    }

    const T& operator()(const std::initializer_list<size_t>& idxs) const {
        return data[index(idxs)];
    }

    void resize(const std::initializer_list<size_t>& newDims) {
        shape = std::vector<size_t>(newDims);
        data.resize(computeTotalSize(newDims));
        calculateStrides();
    }

    void fill(const T& value){
        std::fill(data.begin, data.end(), value);
    }

    
};

} // namespace nn

#endif // TENSOR_H
