#include "MLayerPerceptron.h"
class MLPTrainable :
    public MLayerPerceptron
{
private:
	double learning_rate;
	double *hiddenerr;
	double *outerr;
    double *last_error;
    
	void CalculateUnitErrors(double targetval[]);
	void AdjustWeights(double impulse[]);

public:
    
    MLPTrainable(int _in, int _out, int _hidden, double _learning_rate);
    virtual ~MLPTrainable(void);
    
	void PropagateError(double impulse[], double targetval[]);
    
    double* GetLastError(void){return last_error;}
};

