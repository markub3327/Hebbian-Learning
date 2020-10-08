#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

//  User definitions
#define error_max               (0.0001f)
#define iteration_MAX           (1000000)
#define Ni                      (0.001f)

class MachineLearning
{
    public:
        //  Variables
        uint32_t num_of_inputs, num_of_neurons, num_of_patterns;
        float **weights, *output;

        //  Functions
        MachineLearning(uint32_t _num_of_inputs, uint32_t _num_of_neurons, uint32_t _num_of_patterns);
        void initNet(int seed);
        void getOutput(float *_in);
        float linear(float _x, float _min, float _max);
        float signum(float _x);
        float sigmoida(float _x);
        float AdaptationWeights(float *_in, float *_o);
        ~MachineLearning();        
};