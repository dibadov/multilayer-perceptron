#include <iostream>
#include "TrainingSuperviser.h"
#include "TrainingStopper.h"

using namespace std;

void XOR()
{
    NN::TrainingSuperviser TS(2, 1);
    NN::MLPTrainable M(2, 2, 1, 0.25);
    M.SetHiddenActivation(NN::ActivationFunction::Sigmoid);
    M.SetOutputActivation(NN::ActivationFunction::Sigmoid);
    TS.SetNetwork(&M);
    
    float i0[2] = { 0.f, 0.f };
    float i1[2] = { 0.f, 1.f };
    float i2[2] = { 1.f, 0.f };
    float i3[2] = { 1.f, 1.f };
    
    float o0 = 0;
    float o1 = 1;
    float o2 = 1;
    float o3 = 0;
    
    NN::Datasample input0(2, i0), output0(1, &o0);
    NN::Datasample input1(2, i1), output1(1, &o1);
    NN::Datasample input2(2, i2), output2(1, &o2);
    NN::Datasample input3(2, i3), output3(1, &o3);
    
    TS.AddSample(input0, output0);
    TS.AddSample(input1, output1);
    TS.AddSample(input2, output2);
    TS.AddSample(input3, output3);
    
    NN::MSEStopper stopper(0.01f);
    
    TS.iterative_log = &cout;
    TS.TrainNetwork(&stopper, 4);
    
    cout << *(M.SendImpulse(i0)) << endl;
    cout << *(M.SendImpulse(i1)) << endl;
    cout << *(M.SendImpulse(i2)) << endl;
    cout << *(M.SendImpulse(i3)) << endl;
    
    
}
