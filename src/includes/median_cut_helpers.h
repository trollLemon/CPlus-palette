/* *
 * median_cut_helpers.h
 *
 * This file contains declarations for 
 * helper funcitons for median cut.
 *
 *
 *
 * */

#ifndef HELPERS
#define HELPERS


/* *
 * Returns true if the inputed number is a power of 2.
 *
 * This function will be used by the client code to determine 
 * if the user-inputed size of the palette is a power of two.
 *
 * If the size isn't a power of two, we need to round up to 
 * the next power of two since medain cut only works with powers of 2
 * */
bool isPowerOfTwo(int x);

/* *
 * Returns the number rounded up to the next power of two
 * */
int powerOfTwoSize(int size);
#endif
