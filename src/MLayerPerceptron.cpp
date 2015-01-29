#include "MLayerPerceptron.h"
#include <time.h>


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
			weights1[i][j]=strtod(temp,NULL);
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
			weights2[i][j]=strtod(temp,NULL);	 
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
				weightfile << weights1[i][j]<<"\n";
		for (int i = 0; i<hidden_neurons; i++)
			weightfile<<bias_hidden[i]<<"\n";
		for (int i = 0; i<hidden_neurons; i++)
			for (int j = 0; j<output_num; j++)
				weightfile<<weights2[i][j]<<"\n";
		for (int i = 0; i<output_num; i++)
			weightfile<<bias_output[i]<<"\n";
	    weightfile.close();
	}
	
}

void MLayerPerceptron::initWeights(void)
{
	srand (time(NULL));
	for (int i = 0; i<input; i++)
		for (int j = 0; j<hidden_neurons; j++)
            weights1[i][j] = (rand()%1000)/ 1000.0 - 0.5;
	for (int i = 0; i<hidden_neurons; i++)
		bias_hidden[i] = (rand()%1000)/ 1000.0 - 0.5;
	for (int i = 0; i<hidden_neurons; i++)
		for (int j = 0; j<output_num; j++)
            weights2[i][j] = (rand()%1000)/ 1000.0 - 0.5;
	for (int i = 0; i<output_num; i++)
		bias_output[i] = (rand()%1000)/ 1000.0 - 0.5;
}

MLayerPerceptron::MLayerPerceptron(int _in, int _out, int _hidden)
{
	input = _in;
	output_num = _out;
	hidden_neurons = _hidden;

	//Memory alloc
	//Input-to-hidden weights layer
	weights1 = new double*[input];
	for (int i = 0; i<input; i++)
		weights1[i] = new double[hidden_neurons];

	//hidden-to-output weights layer
	weights2 = new double*[hidden_neurons];
	for (int i = 0; i<hidden_neurons; i++)
		weights2[i] = new double[output_num];

	hidden = new double[hidden_neurons];
	output = new double[output_num];

	bias_hidden = new double[hidden_neurons];
	bias_output = new double[output_num];


	initWeights();

}


MLayerPerceptron::~MLayerPerceptron(void)
{
	for (int i = 0; i<input; i++)
		delete[] weights1[i];
	delete[] weights1;

	for (int i = 0; i<hidden_neurons; i++)
		delete[] weights2[i];
	delete[] weights2;

	delete[] bias_hidden;
	delete[] bias_output;

	delete[] hidden;
	delete[] output;
}

double* MLayerPerceptron::Test(int impulse[])
{
	for (int j = 0; j<hidden_neurons; j++)
	{
		hidden[j] = bias_hidden[j];
		for (int i = 0; i<input;i++)
			hidden[j]+=impulse[i]*weights1[i][j];
	}

    //Activation function in another loop for debugging purposes
    for (int j = 0; j<hidden_neurons; j++)
        hidden[j]=sigmoid(hidden[j]);
	
	for (int j = 0; j<output_num; j++)
	{
		output[j] = bias_output[j];
		for (int i = 0; i<hidden_neurons; i++)
			output[j]+=hidden[i]*weights2[i][j];
	}

	for (int j = 0; j<output_num; j++)
		output[j]=sigmoid(output[j]);


	return output;
}


