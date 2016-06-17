#include "MLayerPerceptron.h"

#ifndef MLPTrainable_h
#define MLPTrainable_h

namespace NN
{

	float SigmoidDeactivation(float v);
	float LinearDeactivation(float v);
	float TanhDeactivation(float v);

	class DLL MLPTrainable :
		public MultilayerPerceptron
	{
	private:
		float learning_rate;
		float *hiddenerr;
		float *outerr;
		float *last_error;

		float(*deactivation_hidden)(float) = LinearDeactivation;
		float(*deactivation_output)(float) = LinearDeactivation;

		void CalculateUnitErrors(float targetval[]);
		void AdjustWeights(float impulse[]);

		void InitWeights(void);

	public:
    
		MLPTrainable(int _in, int _hidden, int _out, float _learning_rate);
		virtual ~MLPTrainable(void);
    
		void PropagateError(float impulse[], float targetval[]);
    
		float* GetLastError(void){return last_error;}

		void SetHiddenActivation(ActivationFunction Activation) override;
		void SetOutputActivation(ActivationFunction Activation) override;
	};

}

#endif