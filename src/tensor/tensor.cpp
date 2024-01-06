/**
 * @file tensor.cpp
 * @author Leif Huender
 * @brief 
 * @version 0.1
 * @date 2024-01-05
 * 
 * @copyright Copyright (c) 2024 Leif Huender
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "tensor.h"

/**
 * @brief Computes the index 'I' in the 1D array (vector) of data that corresponds to a given set of indices in an n-dimensional tensor.
 * The index 'I' is calculated using the formula: 
 * I = ∑ (from k=0 to n-1) [i_k * ∏ (from j=k+1 to n-1) D_j],
 * where i_k is the index in the k-th dimension, and D_j is the size of the j-th dimension.
 * The outer loop (over k) computes the summation, accumulating the contribution of each tensor dimension to the linear index.
 * The inner loop (over j) computes the product of the sizes of all dimensions following the k-th dimension, which is then multiplied by the index in the k-th dimension (i_k).
 * This approach effectively converts a multi-dimensional index into a linear index for accessing elements in the 1D data representation of the tensor.
 * @param dims A vector of indices, one for each dimension of the tensor.
 * @return float The element from the tensor at the specified indices.
 */
size_t Tensor::index(const std::vector<size_t>& dims){
    if(totalSize(dims) != data.size()){
        throw std::runtime_error("Total size of dimensions does not match the size of the tensor");
    }
    size_t n = dims.size();
    size_t index = 0;

    for(size_t k = 0; k < n; ++k){
        size_t product = 1;
        for(size_t j = k + 1; j < n; ++j){
            product *= dimensions[j];
        }
        index += dims[k] * product;
    }
    return index;
}

/**
 * @brief returns the element at the specified dimension, 
 * no need to error check because index function handles the error checking
 * 
 * @param dims 
 * @return float 
 */
float Tensor::getElement(const std::vector<size_t>& dims){
    return data[index(dims)];
}

/**
 * @brief set the element at the specified dimension
 * no need to error check because index function handles the error checking
 * 
 * @param element 
 * @param dims 
 */
void Tensor::setElement(float element, const std::vector<size_t>& dims){
    data[index(dims)] = element;
}

/**
 * @brief fills the tensor with a vector of equally sized data
 * 
 * @param input 
 */
void Tensor::fill(std::vector<float> input){
    if(input.size() != data.size()){
        throw std::runtime_error("Input size does not match data size");
    }
    data = input;
}

/**
 * @brief returns the linear size of dims
 * 
 * @param dims 
 * @return size_t 
 */
size_t Tensor::totalSize(const std::vector<size_t>& dims){
    if(dims.size() != 0){
        size_t totalS = 1;
        for(size_t d : dims){
            totalS *= d;
        }
        return totalS;
    }
    else{
        return 0;
    }
}
/**
 * @brief Constructs a tensor given the dimensions
 * 
 * @param dims 
 */
Tensor::Tensor(std::vector<size_t> dims){
    dimensions = dims;
    data.resize(totalSize(dimensions));
}
