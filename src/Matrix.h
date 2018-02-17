#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <vector>
#include <iostream>

class Matrix
{
public:
	Matrix();
	Matrix(int height, int width);
	Matrix(std::vector<std::vector<double>> const &array);

	Matrix multiply(double const &value); // scalar multiplication

	Matrix add(Matrix const &m) const;			// addition
	Matrix subtract(Matrix const &m) const; // subtraction
	Matrix multiply(Matrix const &m) const; // hadamard product

	Matrix dot(Matrix const &m) const; // dot product
	Matrix transpose() const;					 // transposed matrix

	Matrix applyFunction(double (*function)(double)) const; // to apply a function to every element of the matrix

	void print(std::ostream &flux) const; // pretty print of the matrix

private:
	std::vector<std::vector<double>> array;
	int height;
	int width;
};

std::ostream &operator<<(std::ostream &flux, Matrix const &m); // overloading << operator to print easily

#endif