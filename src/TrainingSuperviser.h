#include "MLPTrainable.h"
#include "TrainingStopper.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <utility>

#ifndef TrainingSuperviser_h
#define TrainingSuperviser_h

namespace NN
{

	struct Datasample
	{
		float *parameters;
		int number;

		Datasample(const int & params_number, float *params)
		{
			number = params_number;
			parameters = new float[params_number];

			for (int i = 0; i < params_number; i++)
				parameters[i] = params[i];
		}
	};

	class TrainingSuperviser
	{
	private:
		std::vector<std::pair<Datasample, Datasample>> samples;

		int parameters_num;
		int label_values_num;
		int train_samples_num;

		bool own_perceptron = false;

		MLPTrainable *perceptron;
		TrainingStoppper *stopper;

		int GetRangeHits(const int &start, const int &end);
		bool IsHit(const int &params_num, float *result, float *class_label);

	public:
		TrainingSuperviser(const int &_parameters_num, const int &_label_values_num);

		int AddSample(Datasample sample, Datasample classlabel);
		int SetNetwork(MLPTrainable * const _perceptron);

		void CreateNetwork(const int &hidden_neurons, const float &learning_rate);

		void TrainNetwork(TrainingStoppper * const _stopper, const int &_train_samples_num);

		std::ostream *iterative_log;
		std::ostream *mse_log;
		std::ostream *hits_train_log;
		std::ostream *hits_test_log;

		~TrainingSuperviser();
	};

}
#endif