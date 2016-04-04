#include <math.h>
#include <stdlib.h>
#include <fstream>

namespace NN
{

	float SigmoidActivation(float v);
	float LinearActivation(float v);
	float TanhActivation(float v);


	enum ActivationFunction
	{
		Linear,
		Sigmoid,
		Tanh
	};

	class MultilayerPerceptron
	{
	private:
		

		float (*activation_hidden)(float) = LinearActivation;
		float (*activation_output)(float) = LinearActivation;

	protected:
		int hidden_neurons;
		int input;
		int output_num;

		// Weight matrices
		float **w_input_hidden;
		float **w_hidden_output;

		float *bias_hidden;
		float *hidden_net;
		float *hidden_z;

		float *bias_output;
		float *output_net;
		float *output_z;

		ActivationFunction HiddenActivation = Linear;
		ActivationFunction OutputActivation = Linear;

	public:
		MultilayerPerceptron(int _in, int _hidden, int _out);
		virtual ~MultilayerPerceptron(void);

		void WriteWeights(char *filename);
		void LoadWeights(char *filename);

		float *SendImpulse(float impulse[]);

		ActivationFunction GetHiddenActivation(void) { return HiddenActivation; }
		ActivationFunction GetOutputActivation(void) { return OutputActivation; }

		virtual void SetHiddenActivation(ActivationFunction Activation);
		virtual void SetOutputActivation(ActivationFunction Activation);

	};

}