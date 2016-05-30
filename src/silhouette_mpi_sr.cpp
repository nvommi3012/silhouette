#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include "math.h"
#include <mpi.h>

#include "timer.c"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> doubleMatrix;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> floatMatrix;

/*--------------------------------------------------------------------------------------------------*/
/*					FILE READ						    */
/*--------------------------------------------------------------------------------------------------*/

void parseTxtFile (string ffile, string lfile, int rows, int cols, int nol, doubleMatrix* inputData, vector<int>* dataSizes)
{
	vector<vector<double> > data (4);

	string line, labelLine;
	string delimiter = " ";
	ifstream dataFile (ffile.c_str());
	ifstream labelFile(lfile.c_str());
	string::size_type sz;
	if (dataFile.is_open() && labelFile.is_open())
	{
		while (getline(dataFile, line) && getline(labelFile, labelLine))
		{
			size_t pos = 0;
			string token;
			while ((pos = line.find(delimiter)) != string::npos)
			{
				token = line.substr(0, pos);
				data[atoi(labelLine.c_str())].push_back(atof(token.c_str()));
    			line.erase(0, pos + delimiter.length());
			}
			data[atoi(labelLine.c_str())].push_back(atof(line.c_str()));
		}
		dataFile.close();
	}
	else cout << "Unable to open file";	

	dataSizes->push_back(data[0].size()/cols);
	for (int i = 1; i < nol; i++)
	{
		data[0].insert(data[0].end(), data[i].begin(), data[i].end());
	    dataSizes->push_back(data[i].size()/cols);
	}

	int k = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			(*inputData)(i, j) = data[0][k];
			k++;
		}
	}
}

/*--------------------------------------------------------------------------------------------------*/
/*			    DISSIMILARITY WITHIN CLUSTER					    */
/*--------------------------------------------------------------------------------------------------*/

doubleMatrix average_dissimilarity_withincluster(doubleMatrix ip_matrix)
{
	doubleMatrix op_matrix;
	doubleMatrix temp;
	temp.resize(ip_matrix.rows(), ip_matrix.rows());
	int i, j;
	for (i = 0; i < (int)ip_matrix.rows(); i++)
	{
		for (j = 0; j < (int)ip_matrix.rows(); j++)
		{
			temp(i, j) = sqrt(((ip_matrix.row(i) - ip_matrix.row(j)).array().square()).sum());
		}
	}
	op_matrix.resize(ip_matrix.rows(), 1);
	op_matrix = temp.rowwise().sum()/(double)(temp.rows() - 1);

	//cout << op_matrix << endl << endl;

	return(op_matrix);
}

/*--------------------------------------------------------------------------------------------------*/
/*			    DISSIMILARITY BETWEEN 2 DIFF CLUSTERS				    */
/*--------------------------------------------------------------------------------------------------*/

doubleMatrix average_dissimilarity_twoclusters(doubleMatrix ip_matrix, doubleMatrix other_matrix)
{
	doubleMatrix op_matrix;
	doubleMatrix temp;
	temp.resize(ip_matrix.rows(), other_matrix.rows());
	int i, j;
	for (i = 0; i < (int)ip_matrix.rows(); i++)
	{
		for (j = 0; j < (int)other_matrix.rows(); j++)
		{
			temp(i, j) = sqrt(((ip_matrix.row(i) - other_matrix.row(j)).array().square()).sum());
		}
	}
	op_matrix.resize(ip_matrix.rows(), 1);
	op_matrix = temp.rowwise().mean();
	
	//cout << op_matrix << endl << endl;

	return(op_matrix);
}


/*--------------------------------------------------------------------------------------------------*/
/*			    			MAIN						    */
/*--------------------------------------------------------------------------------------------------*/


int main(int argc, char* argv[])
{
	int rank = 0;
   	int np = 0;
   	char hostname[MPI_MAX_PROCESSOR_NAME+1];
   	int namelen = 0;
	MPI_Init (&argc, &argv);
    	MPI_Status stat;
    	MPI_Comm_rank (MPI_COMM_WORLD, &rank); /* Get process id */
    	MPI_Comm_size (MPI_COMM_WORLD, &np); /* Get number of processes */
    	MPI_Get_processor_name (hostname, &namelen); /* Get hostname of node */
    	printf ("Hello, world! [Host:%s -- Rank %d out of %d]\n", hostname, rank, np);
	
	stopwatch_init ();
  	struct stopwatch_t* timer = stopwatch_create ();
 	assert (timer);
	long double time = 0;

	int rows = 100, cols = 9, nol = 3;
	string dataFile = "temp.txt";
	string labelFile = "label.txt";	
	vector<int> dataSizes;
	doubleMatrix inputData(rows, cols);
	
	parseTxtFile(dataFile, labelFile, rows, cols, nol, &inputData, &dataSizes);

	//cout << inputData << endl << endl;

	std::vector<int> intervals;
	intervals.push_back(1);
	int var = 0;
	for (int i = 0; i < dataSizes.size(); i++)
	{
		intervals.push_back(dataSizes[i] + var);
		var += dataSizes[i];
		if (i < dataSizes.size() - 1)
		{
			intervals.push_back(var + 1);
		}		
	}

	// for (int x = 0; x < intervals.size(); x++)
	// {
	// 	cout << intervals[x] << endl;
	// }

	stopwatch_start (timer);

	doubleMatrix X;
	X.resize(inputData.rows(), inputData.cols());
	X = inputData.cast<double>();

	doubleMatrix temp;
	doubleMatrix t;
	doubleMatrix t1;
	doubleMatrix a_local;
	doubleMatrix a;
	a.resize(100,1);
	int size_local[1];
	int size[np-1];

	if(rank > 0)
	{
	int offset = (rank-1) * (intervals.size()/(np-1));
	int pkt_size = intervals.size()/(np-1);
	for (int i = offset; i < offset + pkt_size; i = i + 2)
	{
		temp.resize(((intervals[i + 1] - intervals[i]) + 1), X.cols());
		temp = X.block((intervals[i] - 1), 0, ((intervals[i + 1] - intervals[i]) + 1), X.cols());
		if (i == offset)
		{
			a_local.resize(temp.rows(), 1);
			t.resize(temp.rows(), 1);
			a_local = average_dissimilarity_withincluster(temp);
			t = a_local;
		}
		else
		{
			a_local.resize(a_local.rows() + temp.rows(), 1);
			t1.resize(temp.rows(), 1);
			t1 = average_dissimilarity_withincluster(temp);
			a_local << t,t1;
			t.resize(a_local.rows() + temp.rows(), 1);
			t = a_local;
		}
	}
	}

	if(rank > 0)
	{
		//Send the size of the packet and data
		size_local[0] = a_local.size();
		MPI_Send(&size_local,1, MPI_INT, 0, 1 , MPI_COMM_WORLD);
		MPI_Send(a_local.data(),a_local.size(), MPI_DOUBLE, 0, 0 , MPI_COMM_WORLD);
	}

	if(rank == 0)
	{
		//receive the packet size - offset the pointer - receive the data//
		MPI_Recv(&size[0], 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&size[1], 1, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&size[2], 1, MPI_INT, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(a.data(), size[0], MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&a.data()[size[0]], size[1], MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&a.data()[size[0]+size[1]], size[2], MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}	

	
	doubleMatrix adtc;
	doubleMatrix b;
	b.resize(100,1);
	doubleMatrix b_local;
	doubleMatrix temp1;
	doubleMatrix temp2;
	doubleMatrix t2;
	doubleMatrix t3;
	doubleMatrix t4;
	doubleMatrix t5;
	int b_size_local[1];
	int b_size[np-1];

	if (rank > 0)
	{
	
	int offset = (rank-1) * (intervals.size()/(np-1));
	int pkt_size = intervals.size()/(np-1);
	for (int i = offset; i < offset + pkt_size; i = i + 2)
	{
		int ctr = 0;
		for (int j = 0; j < intervals.size(); j = j + 2)
		{
			if (i == j)
			{
				ctr = 1;
				continue;
			}
			temp1.resize(((intervals[i + 1] - intervals[i]) + 1), X.cols());
			temp2.resize(((intervals[j + 1] - intervals[j]) + 1), X.cols());
			temp1 = X.block((intervals[i] - 1), 0, ((intervals[i + 1] - intervals[i]) + 1), X.cols());
			temp2 = X.block((intervals[j] - 1), 0, ((intervals[j + 1] - intervals[j]) + 1), X.cols());
			if (j == 0)
			{
				adtc.resize(temp1.rows(), 1);
				t2.resize(temp1.rows(), 1);
				adtc = average_dissimilarity_twoclusters(temp1, temp2);
				t2 = adtc;				
			}
			else if (j == 2 && ctr == 1)
			{
				adtc.resize(temp1.rows(), 1);
				t2.resize(temp1.rows(), 1);
				adtc = average_dissimilarity_twoclusters(temp1, temp2);
				t2 = adtc;
			}
			else
			{
				adtc.resize(adtc.rows(), adtc.cols() + 1);
				t3.resize(temp1.rows(), 1);
				t3 = average_dissimilarity_twoclusters(temp1, temp2);
				adtc << t2,t3;
				t2.resize(adtc.rows(), adtc.cols() + 1);
				t2 = adtc;
			}
		}
		if (i == offset)
		{
			b_local.resize(adtc.rows(), 1);
			t4.resize(adtc.rows(), 1);
			b_local = adtc.rowwise().minCoeff();
			t4 = b_local;
		}
		else
		{
			b_local.resize(b.rows() + adtc.rows(), 1);
			t5.resize(adtc.rows(), 1);
			t5 = adtc.rowwise().minCoeff();
			b_local << t4,t5;
			t4.resize(b_local.rows() + adtc.rows(), 1);
			t4 = b_local;
		}
	}
	
	} //end of if loop



	if(rank > 0)
	{
		//Send the size of the packet and data
		b_size_local[0] = b_local.size();
		MPI_Send(&b_size_local,1, MPI_INT, 0, 3 , MPI_COMM_WORLD);
		MPI_Send(b_local.data(),b_local.size(), MPI_DOUBLE, 0, 2 , MPI_COMM_WORLD);
	}

	if(rank == 0)
	{
		//receive the packet size - offset the pointer - receive the data//
		MPI_Recv(&b_size[0], 1, MPI_INT, 1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&b_size[1], 1, MPI_INT, 2, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&b_size[2], 1, MPI_INT, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(b.data(), b_size[0], MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&b.data()[b_size[0]], b_size[1], MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&b.data()[b_size[0]+b_size[1]], b_size[2], MPI_DOUBLE, 3, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}




	//if (rank == 0)
		//cout << b << endl << endl;

	if (rank == 0)
	{
		doubleMatrix SIL;
		doubleMatrix c;
		c.resize(X.rows(), 2);
		SIL.resize(X.rows(), 1);
		c << a, b;
		SIL = (b-a).array()/(c.rowwise().maxCoeff()).array();
		cout << SIL << endl;
		time = stopwatch_stop (timer);
		printf("\ntime for implementation: %Lg seconds\n", time);
	}
	return 0;
}
