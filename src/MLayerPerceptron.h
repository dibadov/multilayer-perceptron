#include <math.h>
#include <stdlib.h>
#include <fstream>


class MLayerPerceptron
{
private:
    double sigmoid(double v);
    
    void initWeights(void);
    
protected:
	int hidden_neurons;
	int input;
	int output_num;

    // Weight matrices
	double **w_input_hidden;
	double **w_hidden_output;
    
	double *bias_hidden;
	double *bias_output;
	double *hidden;
	double *output;
    
public:
    MLayerPerceptron(int _in, int _out, int _hidden);
    virtual ~MLayerPerceptron(void);
    
    void WriteWeights(char *filename);
    void LoadWeights (char *filename);
    
	double *SendImpulse(double impulse[]);
    
};

