#include "MLayerPerceptron.h"
class MLPTrainable :
    public MLayerPerceptron
{
private:
	double learning_rate;
	double *hiddenerr;
	double *outerr;
    double *last_error;
	void CalculateUnitError(double targetval[]);
	void AdjustWeights(int impulse[]);

public:
    MLPTrainable(int _in, int _out, int _hidden, double _learning_rate);
	void PropagateError(int impulse[], double targetval[]);
    double* GetLastError(void){return last_error;}
    virtual ~MLPTrainable(void);
};

