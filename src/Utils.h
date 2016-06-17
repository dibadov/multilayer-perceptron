#include <math.h>
#include "Defines.h"

#ifndef Utils_h
#define Utils_h

void DLL Normalize(int *source, float *result, const int &setsnum)
{
	float mean = 0;
	for (int i = 0; i < setsnum; i++)
		mean += source[i];
	mean /= setsnum;

	float deviation = 0;
	for (int i = 0; i < setsnum; i++)
		deviation += pow(source[i] - mean, 2);
	deviation /= setsnum;
	deviation = sqrt(deviation);

	for (int i = 0; i < setsnum; i++)
		result[i] = (source[i] - mean) / deviation;
}


void DLL Normalize(float *source, const int &setsnum)
{
	float mean = 0;
	for (int i = 0; i < setsnum; i++)
		mean += source[i];
	mean /= setsnum;

	float deviation = 0;
	for (int i = 0; i < setsnum; i++)
		deviation += pow(source[i] - mean, 2);
	deviation /= setsnum;
	deviation = sqrt(deviation);

	for (int i = 0; i < setsnum; i++)
		source[i] = (source[i] - mean) / deviation;
}


#endif