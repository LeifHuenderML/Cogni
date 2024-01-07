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
#include <string>
namespace nn{
template<typename T>
class Conv1D{
private:
    int inChannels, 
        outChannels, 
        kernelSize, 
        stride, 
        padding, 
        dilation, 
        goups, 
        outLength, 
        inLength;
    std::string paddingMode;
    Tensor weight;
    Tensor biasT;
    bool bias;

public:
    Conv1D(int inChannels, 
       int outChannels, 
       int kernelSize, 
       int stride = 1, 
       int padding = 0, 
       std::string paddingMode = "zeros", 
       int dilation = 1, 
       int groups = 1, 
       bool bias = true) 
    : inChannels(inChannels), 
      outChannels(outChannels), 
      kernelSize(kernelSize), 
      stride(stride), 
      padding(padding), 
      paddingMode(paddingMode), 
      dilation(dilation), 
      groups(groups), 
      bias(bias) {

    }
    Tensor apply(Tensor& input) {
        weight.resize({outChannels, inChannels / groups, kernelSize});
        Init weightInit;
        weightInit.truncatedNormalInitialization(weight);
        
        if (bias) {
            biasT.resize({outChannels});
            biasT.fill(0);
        }
        
        outLength = ((input.shape[2] + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1;
        Tensor output({input.shape[0], outChannels, outLength});

        for(int n = 0; n < input.shape[0]; ++n) {
            for(int oc = 0; oc < outChannels; ++oc) {
                for(int i = 0; i < outLength; ++i) {
                    T sum = 0;

                    for(int k = 0; k < kernelSize; ++k) {
                        int inputIndex = i * stride + k - padding;
                        if(inputIndex >= 0 && inputIndex < input.shape[2]) {
                            for(int ic = 0; ic < inChannels; ++ic) {
                                sum += input({n, ic, inputIndex}) * weight({oc, ic, k});
                            }
                        }
                    }
                    if(bias) {
                        sum += biasT({oc});  
                    }
                    output({n, oc, i}) = sum;
                }
            }
        }
        return output;
    }
};

}
#endif