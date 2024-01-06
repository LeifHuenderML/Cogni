/**
 * @file conv1d.cpp
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

#include "conv1d.h"
using namespace nn;
Conv1D::Conv1D(int inChannels, int outChannels, int kernelSize, int stride, int padding, std::string paddingMode, int dilation, int groups, bool bias){
    

}

/**
 * @brief 
 * The mathematical formula for a 1D convolutional layer (Conv1D) typically involves sliding a filter 
 * (or kernel) over a one-dimensional input signal and computing the dot product of the filter and the input at
 *  each position. This process can be described by the following formula:

y(t)=∑k=0K−1x(t+k)⋅w(k)+b

Where:

    y(t)) is the output of the convolution at time step tt.
    x(t+k) is the input signal at time step t+kt+k.
    w(k)w(k) is the weight of the kernel at position kk.
    bb is the bias term (a constant added to each output).
    KK is the size of the kernel (the number of weights in the filter).

In this formula, the kernel is applied to each possible position in the input signal by sliding 
it across the signal. For each position, the dot product between the kernel weights and the corresponding
 segment of the input signal is calculated. This dot product is then summed with the bias term to produce the output at that position.

Note that the behavior of the convolution can be affected by factors such as:

    Stride: The number of steps the filter moves at each step. A stride greater than 1 skips input values.
    Padding: Adding zeros (or other values) to the input signal to control the size of the output or to allow the kernel to be applied at the edges of the input.
 * 
 * 
 * @param input 
 */
Tensor Conv1D::apply(const Tensor& input){

    inLength = static_cast<int>(input.dimensions[1]);
    outLength = ((inLength + 2 * padding - dilation * (kernelSize - 1) -1) / stride) + 1;
    Tensor output({input.dimensions[0], static_cast<size_t>(outChannels), static_cast<size_t>(outLength)});
    

}