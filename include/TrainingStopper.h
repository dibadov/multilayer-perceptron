#ifndef TrainingStoppper_h
#define TrainingStoppper_h

namespace NN
{
	class DLL TrainingStoppper
	{
	public:
		virtual bool ShouldStopTraining(float mse, int train_hits, int test_hits, int epochs) = 0;
	};

	class DLL EpochsStopper : public TrainingStoppper
	{
		int max_epochs;
	public:
		EpochsStopper(const int &_max_epochs) :max_epochs(_max_epochs) {};

		virtual bool ShouldStopTraining(float mse, int train_hits, int test_hits, int epochs)
		{
			if (epochs >= max_epochs)
				return true;
			return false;
		}
	};

	class DLL MSEStopper : public TrainingStoppper
	{
		float min_mse;
	public:
		MSEStopper(const float &_min_mse) :min_mse(_min_mse) {};

		virtual bool ShouldStopTraining(float mse, int train_hits, int test_hits, int epochs)
		{
			if (mse <= min_mse)
				return true;
			return false;
		}
	};
}
#endif