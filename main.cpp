// Global libraries
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>

// My libraries
#include "MachineLearning/MachineLearning.hpp"

// Print progressBar
void DoProgress(const char *label, int step, int total);

int main(int argc, char *argv[])
{
    float **inputs, **o, E_local = 0, E_global = 0, E_global_1000last = 0, delta;
    char buffer_line[256], ch;
    uint32_t num_of_inputs, num_of_neurons, num_of_patterns;

    // open input file
    if (argc < 5)
    {
        std::cout << "./Hebbian_learning.out num_of_inputs num_of_neurons num_of_patterns file_input.txt\n";
        return -1;
    }
    else
    {
        // get inputs
        num_of_inputs = atoi(argv[1]);
        std::cout << "num_of_inputs = " << num_of_inputs << "\n";
        
        // get neurons
        num_of_neurons = atoi(argv[2]);
        std::cout << "num_of_neurons = " << num_of_neurons << "\n";

        // get patterns
        num_of_patterns = atoi(argv[3]);
        std::cout << "num_of_patterns = " << num_of_patterns << "\n\n";

        // init trained matrix
        inputs = new float*[num_of_patterns];
        o = new float*[num_of_patterns];
        for (int v = 0; v < num_of_patterns; v++)
        {
            inputs[v] = new float[num_of_inputs + 1];
            o[v] = new float[num_of_neurons];
        }
    }

    // open input file
    FILE *f_IN = fopen(argv[argc - 1], "r");
    if (f_IN == NULL)
    {
        std::cout << "File " << argv[argc - 1] << " cannot open.\n";
        return -1;
    }

    // read input file
    int v = 0;
    while (feof(f_IN) == 0)
    {
        fgets(buffer_line, sizeof(buffer_line), f_IN);
        //std::cout << "buffer_line = " << buffer_line;

        // find blank line
        if (buffer_line[0] != '\n')
        {
            inputs[v][0] = 1.0f;
            std::cout << "inputs[" << v << "][0] = 1  ";
            inputs[v][1] = atof(strtok(buffer_line, ";"));
            std::cout << "inputs[" << v << "][1] = " <<  inputs[v][1] << "  ";
            if (num_of_inputs > 1)
            {
                for (int n = 2; n <= num_of_inputs; n++)
                {
                    inputs[v][n] = atof(strtok(NULL, ";"));
                    std::cout << "inputs[" << v << "][" << n << "] = " <<  inputs[v][n] << "  ";
                }
            }

            for (int m = 0; m < num_of_neurons; m++)
            {
                o[v][m] = atof(strtok(NULL, ";"));
                std::cout << "o[" << v << "][" << m << "] = " <<  o[v][m] << "  ";
            }
            std::cout << "\n";

            v++;
        }
    }

    //  create net class
    MachineLearning *net = new MachineLearning(num_of_inputs, num_of_neurons, num_of_patterns);  

    //  init net
    net->initNet(time(NULL));

    //  set non-blocking to console stream
    //ioctlsocket(0, FIONBIO, );

    //  Training
    for (uint32_t iteration = 0; iteration <= iteration_MAX; iteration ++)
    {
		//read(0, &ch, 1);	// read Quit character
        //if (ch == 'q')  break;

        E_global = 0.0f;
        for (uint32_t v = 0; v < num_of_patterns; v++)
        {
            // get Output
            net->getOutput(inputs[v]);
            E_local = net->AdaptationWeights(inputs[v], o[v]);
            E_global += E_local * E_local;
        }
        E_global = sqrtf(E_global / (float)num_of_patterns);
        //std::cout << "E_global = " << E_global << "\n";

        // End of training
        if (E_global < error_max)   
        {
            std::cout << "\nTraining done successful.\n";
            break; 
        }
        if ((iteration % 5000) == 0)
        {
            if (fabsf(E_global - E_global_1000last) < error_max)
            {
                std::cout << "\nTraining done, error is stable.\n";
                //break;                 
            }
            E_global_1000last = E_global;
        }

        DoProgress("Training: ", iteration, iteration_MAX);
        fflush(stdout);
    }
    std::cout << "\n\n";

    // Test NN on trained vector
    for (uint32_t v = 0; v < num_of_patterns; v++)
    {     
        net->getOutput(inputs[v]);
        for (uint32_t m = 0; m < num_of_neurons; m++)
        {
            delta = o[v][m] - net->output[m];
            std::cout << "o[" << v << "][" << m << "]= " << o[v][m] << ";   "; 
            std::cout << "output[" << v << "][" << m << "]= " << net->output[m];
            if (fabsf(delta) > error_max)
            {
                std::cout << ";    delta= " << delta;
            }
            std::cout << ";    ";
        }
        std::cout << "\n";     
    }

    // Display last error
    std::cout << "\nE_global = " << E_global << "\n";
    
    // Save result to text file
    FILE *f_OUT = fopen("results.txt", "w");    
    for (uint32_t m = 0; m < num_of_neurons; m++)
    {
        for (uint32_t n = 0; n <= num_of_inputs; n++)
        {
            // write to out file    
            fprintf(f_OUT, "weights[%d][%d] = %0.3ff;    ", m, n, net->weights[m][n]);
        }
        fputs("\n\n", f_OUT);
    }
    
    fclose(f_IN);
    fclose(f_OUT);

    return 0;
}

void DoProgress(const char *label, int step, int total)
{
    static double percent_old = 0.0;

    // calc %    
    double percent = ((double)step * 100.0) / (double)total;

    if (fabs(percent - percent_old) > 0.01f)
    {
        // progress width
        const int pwidth = 72;
 
        // minus label len
        int width = pwidth - strlen(label);
        int pos = (step * width) / total;
 
        // fill progress bar with =
        std::cout << label << " [";
        for (int i = 0; i < pos - 1; i++)  
            std::cout << "=";
        std::cout << ">>";
 
        //fill progress bar with spaces
        printf( "% *c", (width - pos + 1), ']');
        printf(" %03.2f%%\r", percent);

        percent_old = percent;
    }
}