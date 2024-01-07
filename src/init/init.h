/**
 * @file init.h
 * @author Leif Huender
 * @brief helper functions for initializing weigths of a Tensor
 * for example Truncated normal initializttion and glorot ..
 * @version 0.1
 * @date 2024-01-06
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

#ifndef INIT_H 
#define INIT_H

#include <random>
#include <chrono>

namespace nn{
template<typename T>
class Init{
    void truncatedNormalInitialization(Tensor& t,T stddev, T mean = 0, T aBound = -2, T bBound = 2){
        std::normal_distribution<T> dist(mean, stddev);

        for(T& element : t){
            do{
                element = dist(generator);
            }
            while(element < aBound || element > bBound);
        }
    }

    void seedRandom(){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
    }

    Init(){
        seedRandom();
    }
    ~Init(){};
};
}
#endif