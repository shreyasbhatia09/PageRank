//Function Used is x^2
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <ctime>
//constants
#define d 0.85
#define e  2.718281828459
#define lim 10
using namespace std;

const double err = 1e-9;
const int node = 400727;
const int edges = 3200440;
const int arraySize = 3200450;
time_t start,stop,fread_time;

float f(float a)
{
	return a*a;
}
struct node
{
	int from;
	int to;
};
int f(const void * a,const void *  b)
{
	const struct node *_a = (struct node *)a;
	const struct node *_b = (struct node *)b;
	return  (_a->to - _b->to);
}

__global__ void addKernel(const int *start, const int *end,const int * out,double *curr,double *prev, double *sum_page)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < node)
	{
		double temp = 0;
		for (int j = start[i]; j <= end[i]; j++)
		{
			temp+=prev[out[j]]*(prev[i]*prev[i])/sum_page[out[j]];
		}
		temp *= d;
		temp += (1 - d);
		curr[i] = temp;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t PageRankCuda(int *start_index, int * end_index,int* out_node,double* current_pagerank, double* previous_pagerank,struct node *arr,double *sum_page)
{
	ofstream myfile;
	myfile.open("D:\\output_spam_func.txt");
	cout<<"File Created"<<endl;
	int *dev_start = 0;
	int *dev_end = 0;
	int *dev_out = 0;

	double *dev_curr = 0;
	double *dev_prev = 0;
	double *dev_sumpage=0;
	cudaError_t cudaStatus;

	int iteration = 0;
	
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for vectors (node,incoming, prev_rank, curr_rank)    .
	cudaStatus = cudaMalloc((void**)&dev_start, node* sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_end, node * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_out, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_curr, node * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_prev, node * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_sumpage, node * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy data to device
	cudaStatus = cudaMemcpy(dev_start, start_index, node * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_end, end_index, node * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_out, out_node, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	int threads=512;
	dim3 block(threads,1, 1);
	dim3 grid( 1+(node/threads ), 1, 1);

next_iteration:

	memset(sum_page,0,sizeof(double)*node);
	for(int i=0;i<edges;i++)
	{
		sum_page[arr[i].from]+= ( f(current_pagerank[arr[i].to]) );
	}


	cudaStatus = cudaMemcpy(dev_sumpage, sum_page, node * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_prev, current_pagerank, node * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}	
	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<< grid, block >>>(dev_start, dev_end, dev_out,dev_curr, dev_prev,dev_sumpage);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "pageRankKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		getchar();
		goto Error;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pageRankKernel!\n", cudaStatus);
		getchar();
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(current_pagerank, dev_curr, node * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
Error:
	cout <<"Iteration:" <<iteration++<<endl;

	//Find out mean and maximum difference
	double mx = 0.0;
	double total=0.0;
	for (int i = 0; i < node; i++)
	{
		
		//Calculate maximum difference for convergence
		if (current_pagerank[i] - previous_pagerank[i] > mx)
			mx = current_pagerank[i] - previous_pagerank[i];
		
		//Update previous pageRank
		previous_pagerank[i] = current_pagerank[i];
	}	
	//Check whether the maximum difference is less than the threshold
	if (mx < err || iteration==500)
	{
		// Write to output file
		myfile << " Iteration:" << iteration<< endl;
		for (int i = 0; i < node; i++)
			myfile  << current_pagerank[i] << endl;
		
		// Free the device arrays
		cudaFree(dev_curr);
		cudaFree(dev_prev);
		cudaFree(dev_start);
		cudaFree(dev_end);
		cudaFree(dev_out);
		time(&stop);
		cout<<endl<<"The file time is "<<difftime(fread_time, start)<<" secs"<<endl;	
		cout<<"The execution time is "<<difftime(stop, fread_time)<<" secs"<<endl;
		cout<<"The total time is "<<difftime(stop, start)<<" secs"<<endl;
		
		return cudaStatus;
	}
	else
		goto next_iteration;
}

int main()
{
	time(&start);
	// Begin pageRank
	cout << "Begin!" << endl;
	
	// Read Graph
	ifstream graph;
	//Read input file
	graph.open("C://Users//Dell//Desktop//datasets//Amazon0312.txt");// Enter dataset path
	
	// Allocate memory to the arrays 
	double *current_pagerank =(double *) malloc(node*sizeof(double));
	double *previous_pagerank = (double *)malloc( node*sizeof(double));
	double *sum_page = (double *)malloc( node*sizeof(double));
	int *end_index =(int*)malloc(node*sizeof(int));
	int *start_index = (int*)malloc(node*sizeof(int));
	int *out_node = (int*)malloc( arraySize*sizeof(int));
	struct node *arr = (struct node *)malloc(arraySize*sizeof(struct node));
	memset(start_index, -1, sizeof(int)*node);
	memset(sum_page,0,sizeof(double)*node);
	for (int i = 0; i < node; i++)
		current_pagerank[i] = previous_pagerank[i] = 1.0 ;
	
	//Start reading from the txt file
	int linesToSkip=4;
	string skip;
	while(linesToSkip--)
		getline(graph,skip);
	int i = 0;
	while (!graph.eof())
	{
		graph >> arr[i].from >> arr[i].to;
		i++;
	}

	cout << "File Ready. Graph Ready" << endl;
	
	time(&fread_time);
	
	// Sort according to the incoming nodes
	qsort( arr,edges,sizeof(struct node),f);
	
	cout<<"Sorting done"<<endl;

	//Make out node array and initialise the startindex and ending index 
	for (i = 0; i < edges; i++)
	{
		out_node[i] = arr[i].from;
		int temp = arr[i].to;
		if (start_index[temp] == -1)
			start_index[temp] = i;
		end_index[temp] = i;
	}

	// Add vectors in parallel.
	cout<<"Calling Pagerank function"<<endl;
	cudaError_t cudaStatus= PageRankCuda(start_index,end_index,out_node,current_pagerank, previous_pagerank,arr,sum_page);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");	
        return 1;
    }
	
	cout << "Program Ends, Press Enter to exit" << endl;
	getchar();
	
	// Free the arrays 
	free(current_pagerank);
	free(previous_pagerank);
	free(end_index);
	free(start_index);
	free(out_node);
	free(arr);
	free(sum_page);
    return 0;
}