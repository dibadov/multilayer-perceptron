#include "MLayerPerceptron.h"

namespace NN
{

	MultilayerPerceptron::MultilayerPerceptron(int _in, int _hidden, int _out)
	{
		input = _in;
		hidden_neurons = _hidden;
		output_num = _out;

		w_input_hidden = new float*[input];
		for (int i = 0; i<input; i++)
			w_input_hidden[i] = new float[hidden_neurons];
    
		w_hidden_output = new float*[hidden_neurons];
		for (int i = 0; i<hidden_neurons; i++)
			w_hidden_output[i] = new float[output_num];
    
		bias_hidden = new float[hidden_neurons];
		hidden_net = new float[hidden_neurons];
		hidden_z = new float[hidden_neurons];

		bias_output = new float[output_num];
		output_net = new float[output_num];
		output_z = new float[output_num];
	}


	MultilayerPerceptron::~MultilayerPerceptron(void)
	{
		for (int i = 0; i<input; i++)
			delete[] w_input_hidden[i];
		delete[] w_input_hidden;
    
		for (int i = 0; i<hidden_neurons; i++)
			delete[] w_hidden_output[i];
		delete[] w_hidden_output;
    
		delete[] bias_hidden;
		delete[] hidden_net;
		delete[] hidden_z;

		delete[] bias_output;
		delete[] output_net;
		delete[] output_z;
	}

	void MultilayerPerceptron::LoadWeights(char *filename)
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

	void MultilayerPerceptron::WriteWeights(char* filename)
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

	float* MultilayerPerceptron::SendImpulse(float impulse[])
	{
		for (int j = 0; j<hidden_neurons; j++)
		{
			hidden_net[j] = bias_hidden[j];
			for (int i = 0; i<input;i++)
				hidden_net[j]+=impulse[i]*w_input_hidden[i][j];
		}

		
		for (int j = 0; j<hidden_neurons; j++)
			hidden_z[j] = activation_hidden(hidden_net[j]);

	
		for (int j = 0; j<output_num; j++)
		{
			output_net[j] = bias_output[j];
			for (int i = 0; i<hidden_neurons; i++)
				output_net[j]+=hidden_z[i]*w_hidden_output[i][j];
		}

		for (int j = 0; j<output_num; j++)
			output_z[j]= activation_output(output_net[j]);

		return output_z;
	}

	void MultilayerPerceptron::SetHiddenActivation(ActivationFunction Activation)
	{
		HiddenActivation = Activation;
		if (Activation == ActivationFunction::Linear)
			activation_hidden = LinearActivation;
		else if (Activation == ActivationFunction::Sigmoid)
			activation_hidden = SigmoidActivation;
	}

	void MultilayerPerceptron::SetOutputActivation(ActivationFunction Activation)
	{
		OutputActivation = Activation;
		if (Activation == ActivationFunction::Linear)
			activation_output = LinearActivation;
		else if (Activation == ActivationFunction::Sigmoid)
			activation_output = SigmoidActivation;
	}


	float SigmoidActivation(float v)
	{
		float r = 1 / (1 + powl(2.7182, -v));
		return r;
	}

	float LinearActivation(float v)
	{
		return v;
	}

}