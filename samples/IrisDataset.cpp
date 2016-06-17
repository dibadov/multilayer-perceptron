#include <iostream>
#include "TrainingSuperviser.h"
#include "TrainingStopper.h"
#include "Utils.h"

using namespace std;

void TrainIris(char *file, const int &setsnum, const int &endtraining)
{
    int hidden_neurons = 2;
    int epochs_stop_criterion = 100;
 
    FILE * fp;
    fp = fopen(file, "r");
    
    float *s_len = new float[setsnum];
    float *s_width = new float[setsnum];
    float *p_len = new float[setsnum];
    float *p_width = new float[setsnum];
    int *i_class = new int[setsnum];
    
    float x, y;
    for (int i = 0; i < setsnum; i++)
    {
        fscanf(fp, "%f,%f,%f,%f,%i", &s_len[i], &s_width[i], &p_len[i], &p_width[i], &i_class[i]);
        
    }
    fclose(fp);
    
    Normalize(s_len, setsnum);
    Normalize(s_width, setsnum);
    Normalize(p_len, setsnum);
    Normalize(p_width, setsnum);
    
    float setosa[] = { 1.0f, 0.0f, 0.0f };
    float versicolor[] = { 0.0f, 1.0f, 0.0f };
    float virginica[] = { 0.0f, 0.0f, 1.0f };
    
    NN::MLPTrainable M(4, hidden_neurons, 3, 0.2f);
    M.SetHiddenActivation(NN::ActivationFunction::Sigmoid);
    M.SetOutputActivation(NN::ActivationFunction::Sigmoid);
    
    NN::TrainingSuperviser TS(4,3);
    float *input = new float[4];
    float *classlabel;
    for (int i = 0; i < setsnum; i++)
    {
        input[0] = s_len[i];
        input[1] = s_width[i];
        input[2] = p_len[i];
        input[3] = p_width[i];
        
        switch (i_class[i])
        {
            case 1:
                classlabel = setosa;
                break;
            case 2:
                classlabel = versicolor;
                break;
            case 3:
                classlabel = virginica;
                break;
            default:
                classlabel = setosa;
                break;
        }
        
        TS.AddSample(NN::Datasample(4, input), NN::Datasample(3, classlabel));
    }
    
    std::ofstream mse_log("iris_mse.log");
    std::ofstream testhits_log("iris_testhits.log");
    std::ofstream trainhits_log("iris_trainhits.log");
    
    TS.iterative_log = &cout;
    TS.mse_log = &mse_log;
    TS.hits_train_log = &trainhits_log;
    TS.hits_test_log = &testhits_log;
    
    
    
    NN::EpochsStopper stopper(100);
    TS.SetNetwork(&M);
    TS.TrainNetwork(&stopper, 100);
    mse_log.close();
    
    delete[] input;
    delete[] s_len;
    delete[] p_len;
    delete[] s_width;
    delete[] p_width;
    delete[] i_class;    
}
