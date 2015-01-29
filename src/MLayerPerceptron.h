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


	double **weights1;
	double **weights2;
	double *bias_hidden;
	double *bias_output;
	double *hidden;
	double *output;
public:
    MLayerPerceptron(int _in, int _out, int _hidden);
    void WriteWeights(char *filename);
    void LoadWeights(char *filename);
	double *Test(int impulse[]);
    virtual ~MLayerPerceptron(void);

};

