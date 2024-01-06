/**
 * @file conv1d.h
 * @author Leif Huender
 * @brief Applies a 1D convolution over an input signal composed of several input planes.
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

#ifndef CONV1D_H
#define CONV1D_H
#include "nn.h"
#include <string>
namespace nn{

class Conv1D{
private:
    int inChannels, outChannels, kernelSize, stride, padding, dilation, goups, bias, outLength, inLength;
    std::string paddingMode;
    Tensor weight;
    Tensor biasT;
    bool bias;

public:
    Conv1D(int inChannels, int outChannels, int kernelSize, int stride=1, int padding=0, std::string paddingMode="zeros", int dilation=1, int groups=1, bool bias=true);
    std::vector<float> apply (const std::vector<float>& input);
};

}
#endif