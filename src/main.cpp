#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#include "Matrix.h"

using namespace std;

// params
Matrix X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;
double learningRate;

double random(double x)
{
	return (double)(rand() % 10000 + 1) / 10000 - 0.5;
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double sigmoidePrime(double x)
{
	return exp(-x) / (pow(1 + exp(-x), 2));
}

double stepFunction(double x)
{
	if (x > 0.9)
	{
		return 1.0;
	}
	if (x < 0.1)
	{
		return 0.0;
	}
	return x;
}

void init(int inputNeuron, int hiddenNeuron, int outputNeuron, double rate)
{
	learningRate = rate;

	W1 = Matrix(inputNeuron, hiddenNeuron);
	W2 = Matrix(hiddenNeuron, outputNeuron);
	B1 = Matrix(1, hiddenNeuron);
	B2 = Matrix(1, outputNeuron);

	W1 = W1.applyFunction(random);
	W2 = W2.applyFunction(random);
	B1 = B1.applyFunction(random);
	B2 = B2.applyFunction(random);
}

Matrix computeOutput(vector<double> input)
{
	X = Matrix({input}); // row matrix
	H = X.dot(W1).add(B1).applyFunction(sigmoid);
	Y = H.dot(W2).add(B2).applyFunction(sigmoid);
	return Y;
}

void learn(vector<double> expectedOutput)
{
	Y2 = Matrix({expectedOutput}); // row matrix

	// Error E = 1/2 (expectedOutput - computedOutput)^2
	// Then, we need to calculate the partial derivative of E with respect to W1,W2,B1,B2

	// compute gradients
	dJdB2 = Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
	dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
	dJdW2 = H.transpose().dot(dJdB2);
	dJdW1 = X.transpose().dot(dJdB1);

	// update weights
	W1 = W1.subtract(dJdW1.multiply(learningRate));
	W2 = W2.subtract(dJdW2.multiply(learningRate));
	B1 = B1.subtract(dJdB1.multiply(learningRate));
	B2 = B2.subtract(dJdB2.multiply(learningRate));
}

void loadTraining(const char *filename, vector<vector<double>> &input, vector<vector<double>> &output)
{
	int trainingSize = 946;
	input.resize(trainingSize);
	output.resize(trainingSize);

	ifstream file(filename);
	if (file)
	{
		string line;
		int n;

		for (int i = 0; i < trainingSize; i++) // load 946 examples
		{
			for (int h = 0; h < 32; h++) // 'images' are 32*32 pixels
			{
				getline(file, line);
				for (int w = 0; w < 32; w++)
				{
					input[i].push_back(atoi(line.substr(w, 1).c_str()));
				}
			}
			getline(file, line);
			output[i].resize(10);								 // output is a vector of size 10
			n = atoi(line.substr(0, 1).c_str()); // get the number that is represented by the array
			output[i][n] = 1;										 // set index that represent the number to 1, other are automatically 0 because of the resize()
		}
	}
	file.close();
}

int main(int argc, char *argv[])
{
	srand(time(NULL)); // to generate random weights

	// learning digit recognition (0,1,2,3,4,5,6,7,8,9)
	std::vector<std::vector<double>> inputVector, outputVector;
	loadTraining("training", inputVector, outputVector); // load data from file called "training"

	// 32*32=1024 input neurons (images are 32*32 pixels)
	// 15 hidden neurons (experimental)
	// 10 output neurons (for each image output is a vector of size 10, full of zeros and a 1 at the index of the number represented)
	// 0.7 learning rate (experimental)
	init(1024, 15, 10, 0.7);

	// train on 30 iterations
	// could be more but to my surprise it is very slow... I did the same program in Java and it was a lot faster, so I probably messed up somewhere...
	for (int i = 0; i < 30; i++)
	{
		for (int j = 0; j < inputVector.size() - 10; j++) // skip the last 10 examples to test the program at the end
		{
			computeOutput(inputVector[j]);
			learn(outputVector[j]);
		}
		cout << "#" << i + 1 << "/30" << endl;
	}

	// test
	cout << endl
			 << "expected output : actual output" << endl;
	for (int i = inputVector.size() - 10; i < inputVector.size(); i++) // testing on last 10 examples
	{
		// as the sigmoid function never reaches 0.0 nor 1.0
		// it can be a good idea to consider values greater than 0.9 as 1.0 and values smaller than 0.1 as 0.0
		// hence the step function.
		for (int j = 0; j < 10; j++)
		{
			cout << outputVector[i][j] << " ";
		}
		cout << ": " << computeOutput(inputVector[i]).applyFunction(stepFunction) << endl;
	}
}