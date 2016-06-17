#include "TrainingSuperviser.h"

namespace NN
{
	int TrainingSuperviser::GetRangeHits(const int &start, const int &end)
	{
		int hits = 0;

		float *result;
		for (int i = start; i < end; i++)
		{
			result = perceptron->SendImpulse(samples[i].first.parameters);
			if (IsHit(samples[i].second.number, result, samples[i].second.parameters))
				hits++;
		}
		return hits;
	}

	bool TrainingSuperviser::IsHit(const int &params_num, float * result, float * class_label)
	{
		int subhits = 0;
		for (int i = 0; i < params_num; i++)
		{
			if (abs(result[i] - class_label[i]) <= 0.5f)
				subhits++;
		}
		if (subhits == params_num)
			return true;
		return false;
	}

	TrainingSuperviser::TrainingSuperviser(const int & _parameters_num, const int & _label_values_num) :parameters_num(_parameters_num), label_values_num(_label_values_num)
	{
		iterative_log = NULL;
		mse_log = NULL;
		hits_train_log = NULL;
		hits_test_log = NULL;
	}

	int TrainingSuperviser::AddSample(const Datasample sample, const Datasample classlabel)
	{
		if (sample.number != parameters_num)
			return 1;
		if (classlabel.number != label_values_num)
			return 2;
		samples.push_back(std::make_pair(sample, classlabel));

		return 0;
	}

	int TrainingSuperviser::SetNetwork(NN::MLPTrainable * const _perceptron)
	{
		if (_perceptron->GetInputNeuronsNum() != parameters_num)
			return 1;
		if (_perceptron->GetOutputNeuronsNum() != label_values_num)
			return 2;

		perceptron = _perceptron;

		return 0;
	}

	void TrainingSuperviser::CreateNetwork(const int & hidden_neurons, const float &learning_rate)
	{
		perceptron = new NN::MLPTrainable(parameters_num, hidden_neurons, label_values_num, learning_rate);
		own_perceptron = true;
	}

	void TrainingSuperviser::TrainNetwork(TrainingStoppper * const _stopper, const int &_train_samples_num)
	{
		train_samples_num = _train_samples_num;
		stopper = _stopper;

		float mse;
		int train_hits = 0, test_hits = 0;

		int epochs = 0;

		do
		{
			mse = 0;
			for (int i = 0; i < train_samples_num; i++)
			{
				perceptron->PropagateError(samples[i].first.parameters, samples[i].second.parameters);

				for (int j = 0; j < samples[i].second.number; j++)
					mse += pow(perceptron->GetLastError()[j], 2);
			}

			mse /= train_samples_num;
			train_hits = GetRangeHits(0, train_samples_num);
			if (train_samples_num < samples.size())
			{
				test_hits = GetRangeHits(train_samples_num, samples.size());
			}

			if (iterative_log)
			{
				*(iterative_log) << "MSE: " << mse << " Train hits:" << train_hits;
				if (train_samples_num < samples.size())
					*(iterative_log) << " Test hits" << test_hits;
				*(iterative_log) << std::endl;
			}
			if (mse_log)
			{
				*(mse_log) << mse << std::endl;
			}
			if (hits_train_log)
			{
				*(hits_train_log) << train_hits << std::endl;
			}
			if (hits_test_log && train_samples_num < samples.size())
			{
				*(hits_test_log) << test_hits << std::endl;
			}
			epochs++;
		} while (!(stopper->ShouldStopTraining(mse, train_hits, test_hits, epochs)));
	}


	TrainingSuperviser::~TrainingSuperviser()
	{
		if (own_perceptron)
			delete perceptron;
	}
}