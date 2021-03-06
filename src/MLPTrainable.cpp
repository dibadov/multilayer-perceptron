#include "MLPTrainable.h"
#include <time.h>
#include <iostream>


namespace NN
{

	MLPTrainable::MLPTrainable(int _in, int _hidden, int _out, float _learning_rate) :MultilayerPerceptron(_in, _hidden, _out)
	{
		learning_rate = _learning_rate;

		hiddenerr = new float[hidden_neurons];
		outerr = new float[output_num];
		last_error = new float[output_num];

		InitWeights();
	}

	MLPTrainable::~MLPTrainable(void)
	{
		delete[] hiddenerr;
		delete[] outerr;
		delete[] last_error;
	}

	void MLPTrainable::InitWeights(void)
	{
		srand(uint(time(NULL)));

		for (int i = 0; i<input; i++)
			for (int j = 0; j<hidden_neurons; j++)
				w_input_hidden[i][j] = (rand() % 500) / 1000.0f - 0.25f;

		for (int i = 0; i<hidden_neurons; i++)
			bias_hidden[i] = (rand() % 500) / 1000.0f - 0.25f;

		for (int i = 0; i<hidden_neurons; i++)
			for (int j = 0; j<output_num; j++)
				w_hidden_output[i][j] = (rand() % 500) / 1000.0f - 0.25f;

		for (int i = 0; i<output_num; i++)
			bias_output[i] = (rand() % 500) / 1000.0f - 0.25f;
	}

	void MLPTrainable::PropagateError(float impulse[], float targetval[])
	{
		SendImpulse(impulse);
		CalculateUnitErrors(targetval);
		AdjustWeights(impulse);
	}

	void MLPTrainable::SetHiddenActivation(ActivationFunction Activation)
	{
		if (Activation == ActivationFunction::Linear)
			deactivation_hidden = LinearDeactivation;
		else if (Activation == ActivationFunction::Sigmoid)
			deactivation_hidden = SigmoidDeactivation;
		else if (Activation == ActivationFunction::Tanh)
			deactivation_hidden = TanhDeactivation;

		MultilayerPerceptron::SetHiddenActivation(Activation);
	}

	void MLPTrainable::SetOutputActivation(ActivationFunction Activation)
	{
		if (Activation == ActivationFunction::Linear)
			deactivation_output = LinearDeactivation;
		else if (Activation == ActivationFunction::Sigmoid)
			deactivation_output = SigmoidDeactivation;
		else if (Activation == ActivationFunction::Tanh)
			deactivation_output = TanhDeactivation;

		MultilayerPerceptron::SetOutputActivation(Activation);
	}

	void MLPTrainable::CalculateUnitErrors(float targetval[])
	{
		for (int i = 0; i < output_num; i++)
			last_error[i] = targetval[i] - output_z[i];

		for (int i = 0; i < output_num; i++)
			outerr[i] = (targetval[i] - output_z[i])*deactivation_output(output_net[i]);

		for (int i = 0; i < hidden_neurons; i++)
		{
			float sum = 0;
			for (int j = 0; j < output_num; j++)
				sum += w_hidden_output[i][j] * outerr[j];
			hiddenerr[i] = deactivation_hidden(hidden_net[i])*sum;
		}
	}


	void MLPTrainable::AdjustWeights(float impulse[])
	{
		for (int i = 0; i < output_num; i++)
			bias_output[i] += learning_rate*outerr[i] * 1;

		for (int i = 0; i < hidden_neurons; i++)
			for (int j = 0; j < output_num; j++)
				w_hidden_output[i][j] += learning_rate * outerr[j] * hidden_z[i];

		for (int i = 0; i < hidden_neurons; i++)
			bias_hidden[i] += learning_rate * hiddenerr[i] * 1;

		for (int i = 0; i < input; i++)
			for (int j = 0; j < hidden_neurons; j++)
				w_input_hidden[i][j] += learning_rate * hiddenerr[j] * impulse[i];
	}


	float SigmoidDeactivation(float v)
	{
		float r = SigmoidActivation(v)*(1 - SigmoidActivation(v));
//		float r = (1 - v)*v;
		return r;
	}

	float LinearDeactivation(float v)
	{
		return 1;
	}

	float TanhDeactivation(float v)
	{
		float r = 1 - TanhActivation(v)*TanhActivation(v);
		return r;
	}
}