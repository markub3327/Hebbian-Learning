#include "MachineLearning.hpp"

MachineLearning::MachineLearning(uint32_t _num_of_inputs, uint32_t _num_of_neurons, uint32_t _num_of_patterns)
{
    // Copy attributes
    num_of_inputs = _num_of_inputs;
    num_of_neurons = _num_of_neurons;
    num_of_patterns = _num_of_patterns;

    //  init weights, output
    weights = new float*[num_of_neurons];
    output = new float[num_of_neurons];
    for (uint32_t m = 0; m < num_of_neurons; m++)
    {
        weights[m] = new float[num_of_inputs + 1];
    }
}

void MachineLearning::initNet(int seed)
{
    float k = 2.4f / (num_of_inputs + 1);

    // init rand()
    srand(seed);

    //  set psudo-random value for init NN
    for (uint32_t m = 0; m < num_of_neurons; m++)
    {
        for (uint32_t n = 0; n <= num_of_inputs; n++)
        {
            weights[m][n] = (((float)rand() / RAND_MAX) * (k + k)) - k;
        }    
    }
}

void MachineLearning::getOutput(float *_in)
{
    for (uint32_t m = 0; m < num_of_neurons; m++)
    {
        output[m] = 0.0f;
        for (uint32_t n = 0; n <= num_of_inputs; n++)
        {
            output[m] += weights[m][n] * _in[n];
        }
        output[m] = linear(output[m], -1.00f, 1.00f);
   	   //output[m] = sigmoida(output[m]);
        //output[m] = signum(output[m]);
    }
}

float MachineLearning::linear(float _x, float _min, float _max)
{
    if (_x >= _max)      return _max;
    else if (_x <= _min) return _min;
    else                 return _x;
}

float MachineLearning::signum(float _x)
{
    if (_x > 0) return 1;
    else        return 0;
}

float MachineLearning::sigmoida(float _x)
{
    return (1.0f / (1.0f + expf(-1.0f * _x)));
}

float MachineLearning::AdaptationWeights(float *_in, float *_o)
{
    float delta, e_global = 0;

    for (uint32_t m = 0; m < num_of_neurons; m++)
    {
        delta = _o[m] - output[m];

        for (uint32_t n = 0; n <= num_of_inputs; n++)
        {
            weights[m][n] += Ni * delta * _in[n];
        }

        e_global += delta * delta;
    }
    e_global = sqrtf(e_global / (float)num_of_neurons);

    return e_global;
}