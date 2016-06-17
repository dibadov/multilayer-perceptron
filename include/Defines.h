#ifndef Defines_h
#define Defines_h

	#ifdef _WIN32
		typedef unsigned int uint;

		#ifdef MAKEDLL
			#  define DLL __declspec(dllexport)
		#else
			#  define DLL __declspec(dllimport)
		#endif
	#endif

#endif
