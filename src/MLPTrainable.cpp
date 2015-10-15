#include "MLPTrainable.h"

MLPTrainable::MLPTrainable(int _in, int _out, int _hidden, double _learning_rate):MLayerPerceptron(_in, _out, _hidden)
{
	learning_rate = _learning_rate;

	hiddenerr = new double[hidden_neurons];
	outerr = new double[output_num];
	last_error = new double[output_num];
}

MLPTrainable::~MLPTrainable(void)
{
    delete[] hiddenerr;
    delete[] outerr;
    delete[] last_error;
}

void MLPTrainable::PropagateError(int impulse[], double targetval[])
{
	SendImpulse(impulse);
	CalculateUnitErrors(targetval);
	AdjustWeights(impulse);
}


void MLPTrainable::CalculateUnitErrors(double targetval[])
{
    for (int i = 0; i<output_num; i++)
        last_error[i]=targetval[i]-output[i];
    
    //Outer unit error:
	for (int i = 0; i<output_num; i++)
        outerr[i]= (targetval[i]-output[i])*output[i]*(1-output[i]);
    
    double sum;
    for(int i = 0; i<hidden_neurons; i++)
	{
        //Calculate sum of outer neuron errors multiplied by weights
        sum = 0;
        for (int j = 0; j<output_num; j++)
            sum += w_hidden_output[i][j]*outerr[j];
        hiddenerr[i]=hidden[i]*(1-hidden[i])*sum;
	}
}



void MLPTrainable::AdjustWeights(int impulse[])
{
	for(int i = 0; i<hidden_neurons; i++)
		for (int j = 0; j<output_num; j++)
			w_hidden_output[i][j]+=learning_rate*outerr[j]*hidden[i];
    
	for (int i = 0; i<hidden_neurons; i++)
		bias_hidden[i] += learning_rate*hiddenerr[i]*1;
    
	for (int i = 0; i<output_num; i++)
		bias_output[i]+=learning_rate*outerr[i]*1;

	for(int i = 0; i<input; i++)
		for(int j = 0; j<hidden_neurons; j++)
			w_input_hidden[i][j]+=learning_rate*hiddenerr[j]*impulse[i];
	
}


