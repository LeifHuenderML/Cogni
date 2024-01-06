/**
 * @file tensor.h
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

#ifndef TENSOR_H
#define TENSOR_H
#include <vector>

namespace nn {
class Tensor{
private:

    std::vector<float> data;
    std::vector<size_t> dimensions;
    size_t totalSize(const std::vector<size_t>& dims);
    size_t index(const std::vector<size_t>& dims);
public:
    float getElement(const std::vector<size_t>& dims);
    void setElement(float element, const std::vector<size_t>& dims);
    void fill(std::vector<float> input);
    Tensor(std::vector<size_t> dims);
};
}

#endif