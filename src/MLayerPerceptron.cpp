#include "MLayerPerceptron.h"
#include <time.h>


MLayerPerceptron::MLayerPerceptron(int _in, int _out, int _hidden)
{
    input = _in;
    output_num = _out;
    hidden_neurons = _hidden;
    
    w_input_hidden = new double*[input];
    for (int i = 0; i<input; i++)
        w_input_hidden[i] = new double[hidden_neurons];
    
    w_hidden_output = new double*[hidden_neurons];
    for (int i = 0; i<hidden_neurons; i++)
        w_hidden_output[i] = new double[output_num];
    
    hidden = new double[hidden_neurons];
    output = new double[output_num];
    
    bias_hidden = new double[hidden_neurons];
    bias_output = new double[output_num];
    
    
    initWeights();
    
}


MLayerPerceptron::~MLayerPerceptron(void)
{
    for (int i = 0; i<input; i++)
        delete[] w_input_hidden[i];
    delete[] w_input_hidden;
    
    for (int i = 0; i<hidden_neurons; i++)
        delete[] w_hidden_output[i];
    delete[] w_hidden_output;
    
    delete[] bias_hidden;
    delete[] bias_output;
    
    delete[] hidden;
    delete[] output;
}

double MLayerPerceptron::sigmoid(double v)
{
	double r = 1/(1+powl(2.7182,-v));
	return r;
}

void MLayerPerceptron::LoadWeights(char *filename)
{
	char temp[256];
	std::fstream weightfile;
    weightfile.open(filename,std::ios::out|std::ios::in);
	for (int i = 0; i<input; i++)
		for (int j = 0; j<hidden_neurons; j++)
		{
			weightfile.getline(temp, 256);
			w_input_hidden[i][j]=strtod(temp,NULL);
		}	

	for (int i = 0; i<hidden_neurons; i++)
	{
		weightfile.getline(temp, 256);
		bias_hidden[i] = strtod(temp,NULL);
	}

	for (int i = 0; i<hidden_neurons; i++)
	{
		for (int j = 0; j<output_num; j++)
		{
			weightfile.getline(temp, 256);
			w_input_hidden[i][j]=strtod(temp,NULL);
		}
	}

	for (int i = 0; i<output_num; i++)
	{
		weightfile.getline(temp, 256);
		bias_output[i] = strtod(temp,NULL);
	}

    weightfile.close();

}

void MLayerPerceptron::WriteWeights(char* filename)
{
    std::ofstream weightfile(filename);
	if (weightfile.is_open())
	{
		for (int i = 0; i<input; i++)
			for (int j = 0; j<hidden_neurons; j++)
				weightfile << w_input_hidden[i][j]<<"\n";
        
		for (int i = 0; i<hidden_neurons; i++)
			weightfile<<bias_hidden[i]<<"\n";
        
		for (int i = 0; i<hidden_neurons; i++)
			for (int j = 0; j<output_num; j++)
				weightfile<<w_hidden_output[i][j]<<"\n";
        
		for (int i = 0; i<output_num; i++)
			weightfile<<bias_output[i]<<"\n";
        
	    weightfile.close();
	}
	
}

void MLayerPerceptron::initWeights(void)
{
	srand (uint(time(NULL)));
    
	for (int i = 0; i<input; i++)
		for (int j = 0; j<hidden_neurons; j++)
            w_input_hidden[i][j] = (rand()%1000)/ 1000.0 - 0.5;
	for (int i = 0; i<hidden_neurons; i++)
		bias_hidden[i] = (rand()%1000)/ 1000.0 - 0.5;
	for (int i = 0; i<hidden_neurons; i++)
		for (int j = 0; j<output_num; j++)
            w_hidden_output[i][j] = (rand()%1000)/ 1000.0 - 0.5;
	for (int i = 0; i<output_num; i++)
		bias_output[i] = (rand()%1000)/ 1000.0 - 0.5;
}

double* MLayerPerceptron::SendImpulse(int impulse[])
{
	for (int j = 0; j<hidden_neurons; j++)
	{
		hidden[j] = bias_hidden[j];
		for (int i = 0; i<input;i++)
			hidden[j]+=impulse[i]*w_input_hidden[i][j];
	}

    //Activation function in another loop for debugging purposes
    for (int j = 0; j<hidden_neurons; j++)
        hidden[j]=sigmoid(hidden[j]);
	
	for (int j = 0; j<output_num; j++)
	{
		output[j] = bias_output[j];
		for (int i = 0; i<hidden_neurons; i++)
			output[j]+=hidden[i]*w_hidden_output[i][j];
	}

	for (int j = 0; j<output_num; j++)
		output[j]=sigmoid(output[j]);


	return output;
}


