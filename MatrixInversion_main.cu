// ГЛАВНАЯ ПРОГРАММА
// ОБРАЩЕНИЕ МАТРИЦ методом модифицированного Грамма-Шмидта

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
//#include "shrUtils.h"
#include "math.h"
#include <cuda_runtime_api.h>

#if !defined(_Complex_Type_)
#define _Complex_Type_

struct __align__(8) complex {
	__device__ __host__ complex(){}
	__device__ __host__ complex( float r, float i ) : x(r), y(i) {}
	//__device__ __host__ complex operator+( complex &other );
	__device__ __host__	complex operator+( complex &other ) const {return complex( x + other.x, y + other.y );};
	__device__ __host__	complex operator+( const complex &other ) const {return complex( x + other.x, y + other.y );};
	__device__ __host__ complex operator-( complex &other ) const {return complex( x - other.x, y - other.y );};
	__device__ __host__ complex operator-() const {return complex( -x , -y);};
	__device__ __host__ complex operator*( complex &other ) const {return complex( x*other.x-y*other.y, x*other.y+y*other.x);};
	__device__ __host__ complex operator*( const complex &other ) const {return complex( x*other.x-y*other.y, x*other.y+y*other.x);};
	__device__ __host__ complex operator*( float other ) const {return complex( x*other, y*other);};
	__device__ __host__ complex operator~() const {return complex( x ,- y);};
	__device__ __host__ float norm() const {return x*x+y*y;};
	__device__ __host__ void zero(){x=0.0f;y=0.0f;return;};
   float x, y;
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////
// определения функций ЦПУ 
#define IDX2C(i,j,ld) (((i)*(ld))+(j))
//int DevicePropertymain( int argc, const char** argv);
int divMaxint(int A, int B);
void randMatrixComplex(complex* Y,const int M,const int N,const float low, const float hi);
void SeqMatrixComplex(complex* Y,const int M,const int N);
void printMatrixComplex(const complex* Y,const int M,const int N);
void printMatrixComplexDiag(const complex* Y,const int M,const int N);
complex dotSqrtRecipRealHost(complex* Y,int N);
void MatVecMulComplexHost(const complex* Y,const int N,const int M, complex* L);
void subMatrixVecScalHost(complex* Y, complex* L, const int N, const int M);
void RowScalarMul_Host(complex* L, const int M,const int k);
void VecMulVechSumMatr_Host(complex* L,const int M, const int k);
void AMulAh_Host(const complex* Y,complex* R,const  int N,const  int M);
void LhMulL_Host(const complex* L,complex* R,const  int M);
void AMulB_Host(const complex* R,const complex* Rinv,complex* tmp, const  int M, const  int N,const  int P );
float absEyeMatr_Host(const complex* Y, const int M);

///////////////////////////////////////////////////////////////////////////////////
// Определения функций ГПУ
__global__ void MatVecMulComplex(const complex* Y,const  int N,const  int p,complex* c, complex* L);
__global__ void sumMatrixRowShort(const complex* Matr,const  int lead, complex* vec);
__global__ void sumMatrixColShort(const complex* Matr,const  int lead, complex* vec);
__global__ void subMatrixVecScal(complex* Y,complex* c,complex* L,const  int p,const  int k);
__global__ void subMatrixVecScal1(complex* Y,complex* c,complex* L,const  int p,const  int k,const int M);
__global__ void subMatrixVecScal2(complex* Y,complex* c,complex* L,const  int p,const  int k,const int M);
__global__ void VecMulVechSumMatr(complex* L, const  int M,const  int k);
__global__ void RowScalarMul(complex* L, const int M,const int k);
__global__ void SetmatL(complex* L,complex* c, int k, int M);
__global__ void mult(complex* Y,complex* z,int p,int N);
__device__ __host__ complex AhMulB(complex A, complex B);

////////////////////////////////////////////////////////////////////////////////////////////////////

//MAIN PROGRAMM

#define rnum 4
#define CPUtest 0

/* Main */
int main(int argc, char** argv)
{   int N=1024*64; // =65535 ~5*10^4 число столбцов (отсчетов) ---> частота полосы 1МГц // должно быть кратно 1024 но не меньше 4096
	int M=192; // число строк (каналов)
	dim3 blockn; // конфигурация блоков
	dim3 threadn; // конфигурация тредов
	size_t blocksize; // количество доп. shared памяти
	int NM=N*M; // размер матрицы Y
	complex* h_Y; // матрица наблюдений Y в ЦПУ (N*M)
    complex* d_Y; // матрица наблюдений Y на ГПУ (N*M)
	complex* h_R; // корреляционная матрица R (M*M)
	complex* h_Rinv; // корреляционная матрица Rinv (M*M)
	complex* h_L; // матрица L треугольная на ЦПУ, хранится по столбцам (M*(M+1)/2)
	complex* d_L; // матрица L треугольная на ГПУ, хранится по столбцам	(M*(M+1)/2)
	complex* d_c; // вектор скалярных произведений с на ГПУ (М)
	complex* h_Lg; //	Матрица L треугольная, хранится по столбцам (M*(M+1)/2) - для хранения матрици переданной с ГПУ
	complex* h_tmp; // временное хранилище на ЦПУ
	complex* d_tmp;	// временное хранилище на ГПУ
	clock_t cpuStart, cpuFinish; 
	cudaEvent_t start, stop; // события для таймера
	//cudaError_t err; // ошибка
	float time; // время в миллисекундах
	int o; // индекс начала столбца треугольной матрицы
	//--------------------------------------------------------------------------------------------------------------------------
	// Создаем события тайймера
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//DevicePropertymain(  argc,  (const char**) argv) ;
	// Инициализация
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	int THRpBL=deviceProp.maxThreadsPerBlock;
	printf("MatrixInversion start..%d Threads per block\n",THRpBL);
	cudaEventRecord( start, 0 ); // старт таймера
	// Выделение памяти ЦПУ
	h_Y = (complex*)malloc(NM * sizeof(h_Y[0]));
	h_R = (complex*)malloc(M*M * sizeof(h_R[0]));
	h_Rinv = (complex*)malloc(M*M * sizeof(h_Rinv[0]));
	h_tmp = (complex*)malloc(NM * sizeof(h_tmp[0]));
	h_L = (complex*)malloc(M*M * sizeof(h_L[0]));
	h_Lg = (complex*)malloc(M*M * sizeof(h_Lg[0]));
	cudaMalloc((void**)&d_Y,NM*sizeof(d_Y[0]));
	cudaMalloc((void**)&d_L,M*M*sizeof(d_L[0]));
	cudaMalloc((void**)&d_c,M*sizeof(d_c[0]));
	cudaMalloc((void**)&d_tmp,M*8*sizeof(d_tmp[0]));
	//-----------------------------------------------------------------------------------------------------------------------------
	cudaEventRecord( stop, 0 );//остановка таймера
	cudaEventSynchronize( stop );//не забываем синхронизироваться
	cudaEventElapsedTime( &time, start, stop ); //определение времени
	printf("GPU Initialaze				%12.0f ms\n",time); //печать времени
	// ввод данных 
	srand((int)(clock()));
	randMatrixComplex(h_Y,M,N,-1000,1000);
	//SeqMatrixComplex(h_Y,M,N);
	//-----------------------------------------------------------------------------------------------------------------------------
			//warm up
		blockn=dim3(100,50);threadn=dim3(THRpBL,1);  
		for(int r=0;r<1;r++){mult<<<blockn,threadn>>>(&d_Y[0],d_tmp,100,100);}
	cpuStart=clock();
	// Передача матрицы из основной памяти в ГПУ 
	cudaEventRecord( start, 0 );
	cudaMemcpy(d_Y,h_Y,NM*sizeof(h_Y[0]),cudaMemcpyHostToDevice); 
	cudaEventRecord( stop, 0 );//остановка таймера
	cudaEventSynchronize( stop );//не забываем синхронизироваться
	cudaEventElapsedTime( &time, start, stop ); //определение времени
	printf("GPU Load data  				%12.0f ms\n",time); //печать времени
//==============================================================================================================================
	cudaEventRecord( start, 0 );
	// РАСЧЕТЫ НА ГПУ	
	int qx,qy,p;
	for(int k=0;k<M;k++) //Цикл расчетов
	{
		o=M*k-(k*(k-1))/2; // индекс начала столбца треугольной матрицы
		qx=THRpBL/4;
		p=N/qx;
		blockn=dim3(N/(p*qx),M-k);threadn=dim3(qx,1); blocksize=(qx+qx)*sizeof(float);//параметры сетки
		MatVecMulComplex<<<blockn,threadn,blocksize>>>(&d_Y[k*N],N,p,&d_c[0],&d_L[o]); 
		if(k<M-1) //если не последняя итерация
		{	// Находим скалярные произведения по блочно на ГПУ
			//float lm=floorf(512.0f/(float)(M-k-1));
			//int ln=(int)pow(2.0f,floorf(log(lm)/log(2.0f)));
			// Находим проекцию на ГПУ
			//p=32;
			//blockn=dim3(N/(p*THRpBL),M-k-1);threadn=dim3(THRpBL,1);blocksize=THRpBL*sizeof(complex);//параметры сетки 
			//subMatrixVecScal<<<blockn,threadn,blocksize>>>(&d_Y[k*N],d_c,&d_L[0],p,k);
			qy=16;
			qx=THRpBL/qy;
			//qx=128/qy;
			//qx=64;
			p=32;
			//printf("qX %d   qY %d   blockX %d  blockY %d   p %d \n",qx,qy,N/(p*qx),(int)ceilf((float)(M-k-1)/(float)qy),p);
			blockn=dim3(N/(p*qx),(int)ceilf((float)(M-k-1)/(float)qy));threadn=dim3(qx,qy);blocksize=(qx*p+qy+1)*sizeof(complex);//параметры сетки 
			subMatrixVecScal1<<<blockn,threadn,blocksize>>>(&d_Y[k*N],d_c,&d_L[0],p,k,M);
			//p=1; qx=128; qy=1;
			//blockn=dim3(N/(p*qx),(int)ceilf((float)(M-k-1)/(float)(qy*rnum)));threadn=dim3(qx,qy);blocksize=(qx*p+qy+1)*sizeof(complex);//параметры сетки 
			//subMatrixVecScal2<<<blockn,threadn,blocksize>>>(&d_Y[k*N],d_c,&d_L[0],p,k,M);

			blockn=dim3(1,1); threadn=dim3(1,M-k);
			SetmatL<<<blockn,threadn>>>(d_L,d_c, k,M);
			if(k>0){blockn=dim3(k,1); threadn=dim3(1,M-k-1);VecMulVechSumMatr<<<blockn,threadn>>>(d_L, M,k);}
			//printf("%s\n",cudaGetErrorString(cudaGetLastError())); 
		}
		if(k>0) //Если не первая итерация
		{	// обновляем подстроку
			blockn=dim3(1,1); threadn=dim3(k,1);	
			RowScalarMul<<<blockn,threadn>>>(d_L,M,k);
		}
	}
	// передаем матрицу L  в ЦПУ
	cudaEventRecord( stop, 0 );//остановка таймера
	cudaEventSynchronize( stop );//не забываем синхронизироваться
	cudaEventElapsedTime( &time, start, stop ); //определение времени
	printf("GPU Compute  				%12.0f ms\n",time); //печать времени
	cudaEventRecord( start, 0 );
	cudaMemcpy(h_Lg,d_L,(M*M)*sizeof(d_L[0]),cudaMemcpyDeviceToHost);
	cpuFinish=clock();
	cudaEventRecord( stop, 0 );//остановка таймера
	cudaEventSynchronize( stop );//не забываем синхронизироваться
	cudaEventElapsedTime( &time, start, stop ); //определение времени
	printf("GPU Save data  				%12.0f ms\n\n",time); //печать времени
	//printf("\nGPU Process  				%12.0f ms  or %12.0f ms \n",time,(double)(cpuFinish - cpuStart) / CLOCKS_PER_SEC*1000); //печать времени
	//printf("%s\n",cudaGetErrorString(cudaGetLastError())); 
	
//==============================================================================================================================

#if CPUtest
cpuStart=clock();
	AMulAh_Host(h_Y,h_R,N,M); // Находим корреляционную матрицу R (для проверки)
cpuFinish=clock();
printf("CPU Create R				%12.0f ms\n",(double)(cpuFinish - cpuStart) / CLOCKS_PER_SEC*1000);
#endif

// РАСЧЕТЫ НА ЦПУ
#if CPUtest==2
cpuStart=clock();
	for(int k=0;k<M;k++) //Цикл расчетов
	{
		o=M*k-(k*(k-1))/2; // индекс начала столбца треугольной матрицы
		h_L[o]=dotSqrtRecipRealHost(&h_Y[k*N],N); // расчет первого элемента на ЦПУ
		if(k<M-1) //если не последняя итерация
		{			
			MatVecMulComplexHost(&h_Y[k*N],N,M-k,&h_L[o]); //скалярные произведения на ЦПУ
			subMatrixVecScalHost(&h_Y[k*N],&h_L[o],N,M-k);// Находим проекцию на ЦПУ
			// Перемножаем матрицы L итеративно на ГПУ
			if(k>0) //Если не первая итерация
			{	//обновляем подматрицу
				VecMulVechSumMatr_Host(h_L,M,k);// Перемножаем матрицы L итеративно на ЦПУ
			}
		}
		if(k>0) //Если не первая итерация
		{	// обновляем подстроку
			RowScalarMul_Host(h_L,M,k);
		}
	}
	cpuFinish=clock();
	printf("CPU Process					%12.0f ms\n",(double)(cpuFinish - cpuStart) / CLOCKS_PER_SEC*1000);
#endif	
	//Собственно можно считать что конец
	// если очень надо ищем обратную матрицу в явном виде
#if CPUtest
	cpuStart=clock();	
	LhMulL_Host(h_Lg,h_Rinv,M); // При необходимости также перенести на ГПУ
	cpuFinish=clock();
	printf("CPU Transpose Mult.			%12.0f ms\n",(double)(cpuFinish - cpuStart) / CLOCKS_PER_SEC*1000);	
							//printMatrixComplexDiag(&h_tmp[M],1,M-1);
							//printMatrixComplexDiag(&h_L[M],1,M-1);	
	//Проверка перемножением матрицы R и Rinv
	cpuStart=clock();
	AMulB_Host(h_R,h_Rinv,h_tmp,M,M,M);
	float nrmR;
	nrmR = absEyeMatr_Host(h_tmp, M);
	cpuFinish=clock();
	printf("CPU Proverka 				%12.0f ms\n",(double)(cpuFinish - cpuStart) / CLOCKS_PER_SEC*1000);
	printf("Error %12.4e\n\n",nrmR);
	//printMatrixComplex(h_tmp,M,M);
#endif	
	// освобождаем ресурсы 
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	free(h_Y);
	free(h_R);
	free(h_Rinv);
	free(h_L);
	free(h_Lg);
	cudaFree(d_Y);
	cudaFree(d_L);
	cudaFree(d_c);
	cudaFree(d_tmp);
	
	// ждем ввода
    if (argc > 1) {
        if (!strcmp(argv[1], "-noprompt") ||
            !strcmp(argv[1], "-qatest") ) 
        {
            return EXIT_SUCCESS;
        }
    } 
    else
    {
        printf("\nPress ENTER to exit...\n");
    //    getchar();
    }

	//cudaThreadExit();
    return EXIT_SUCCESS;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Функции ГПУ, ядра

//-----------------------------------------------------------------------------------------------------------------------------------------------
// умножение сопряженного на несопряженное чисел
__device__ __host__ complex AhMulB(complex A, complex B)
	{
	complex res;
	res.x=A.x*B.x+A.y*B.y;
	res.y=A.x*B.y-A.y*B.x;
	return res;
	}
//----------------------------------------------------------------------------------------------------------------------------
// скалярное произведение поблочно
__global__ void MatVecMulComplex(const complex* Y,const  int N,const  int p, complex* c,complex* L)
	{
	extern __shared__ float data1[];
	
	float* sumax=(float*)&data1[0];
	float* sumay=(float*)&data1[blockDim.x];
	int indg=threadIdx.x+blockDim.x*blockIdx.x;
	int stride=blockDim.x*gridDim.x;
	int indy=indg+p*gridDim.x*blockDim.x*(threadIdx.y+blockDim.y*blockIdx.y);
	//int threadIdx.x=threadIdx.x;
	sumax[threadIdx.x]=0.0f;
	sumay[threadIdx.x]=0.0f;
	#pragma unroll 
	for(int i=0;i<p;i++)
	{
		sumax[threadIdx.x]=sumax[threadIdx.x]+Y[indg+i*stride].x*Y[indy+i*stride].x+Y[indg+i*stride].y*Y[indy+i*stride].y;
		sumay[threadIdx.x]=sumay[threadIdx.x]+Y[indg+i*stride].x*Y[indy+i*stride].y-Y[indg+i*stride].y*Y[indy+i*stride].x;
		}
	__syncthreads();
int indc=blockIdx.x+gridDim.x*(threadIdx.y+blockDim.y*blockIdx.y);	
#pragma unroll 	
for(int s=blockDim.x/2;s>32;s>>=1)
		{
		if(threadIdx.x<s)
		{	int ss=threadIdx.x+s;
			sumax[threadIdx.x]= sumax[threadIdx.x]+sumax[ss];
			sumay[threadIdx.x]= sumay[threadIdx.x]+sumay[ss];
		
		}
		__syncthreads();
		}
	if(threadIdx.x<32)
		{
			sumax[threadIdx.x]=sumax[threadIdx.x]+sumax[threadIdx.x+32];
			sumay[threadIdx.x]=sumay[threadIdx.x]+sumay[threadIdx.x+32];
			sumax[threadIdx.x]=sumax[threadIdx.x]+sumax[threadIdx.x+16];
			sumay[threadIdx.x]=sumay[threadIdx.x]+sumay[threadIdx.x+16];
			sumax[threadIdx.x]=sumax[threadIdx.x]+sumax[threadIdx.x+8];
			sumay[threadIdx.x]=sumay[threadIdx.x]+sumay[threadIdx.x+8];
			sumax[threadIdx.x]=sumax[threadIdx.x]+sumax[threadIdx.x+4];
			sumay[threadIdx.x]=sumay[threadIdx.x]+sumay[threadIdx.x+4];
			sumax[threadIdx.x]=sumax[threadIdx.x]+sumax[threadIdx.x+2];
			sumay[threadIdx.x]=sumay[threadIdx.x]+sumay[threadIdx.x+2];
			sumax[threadIdx.x]=sumax[threadIdx.x]+sumax[threadIdx.x+1];
			sumay[threadIdx.x]=sumay[threadIdx.x]+sumay[threadIdx.x+1];
		}
	
	 if(threadIdx.x==0) 
		{
		 if (indc==0)
			{
			L[indc].x=1.0f/sqrtf(sumax[threadIdx.x]);
			L[indc].y=0.0f;
			c[indc].x=1.0f/sqrtf(sumax[threadIdx.x]);
			c[indc].y=0.0f;
			
			//__threadfence();
			//atomicExch(&go,5);

			}
//__threadfence();
//__syncthreads();
		// while(go==0){};		
		 if (indc!=0)
			 {
			c[indc].x=sumax[threadIdx.x];c[indc].y=sumay[threadIdx.x];
		//L[indc].x=-sumax[threadIdx.x]*(c[0].x*c[0].x);L[indc].y=-sumay[threadIdx.x]*(c[0].x*c[0].x);
			L[indc].x=sumax[threadIdx.x];L[indc].y=sumay[threadIdx.x];
			 }
		}		
	return;
	}

//------------------------------------------------------------------------------------------------------------------------------------
// построчное сложение
__global__ void sumMatrixRowShort(const complex* Matr, const  int lead, complex* vec)
	{ 
	int ind=lead*threadIdx.y;
	complex sum;
	sum.zero();
	for(int i=0;i<lead;i++)
		{sum=sum+Matr[ind+i];}
	vec[threadIdx.y]=sum;	
	return;
	}
//
__global__ void sumMatrixColShort(const complex* Matr,const  int lead, complex* vec)
	{ 

	int ind=threadIdx.x;
	complex sum;
	sum.zero();
	for(int i=0;i<lead;i++)
		{sum=sum+Matr[ind+i*blockDim.x];}
	vec[ind]=sum;	
	return;
	}

//----------------------------------------------------------------------------------------------------------------------------------
// Находим поекцию A=A-l*g , заодно расчитываем очередной столбец матрицы L
__global__ void subMatrixVecScal(complex* Y,complex* c,complex* L,const  int p,const  int k)
{
	int indc=threadIdx.y+blockDim.y*blockIdx.y+1;
	__shared__ complex cc;
	__shared__ float cc0;
	__shared__ complex c2;
	int a;
	a=threadIdx.x+threadIdx.y;
	if(a==0)
	{	cc=c[indc];
		cc0=c[0].x;
		c2=-cc*(cc0*cc0);
	}
	__syncthreads();
	int indy=p*gridDim.x*blockDim.x*indc;
	indc=threadIdx.x+blockDim.x*blockIdx.x;
	complex b;

	for(int i=0;i<p;i++)
	{	a=indc+i*gridDim.x*blockDim.x;
		b=Y[a]*c2;
		a=a+indy;
		Y[a]=Y[a]+b;		
	}
	return;
}

__global__ void subMatrixVecScal1(complex* Y,complex* c,complex* L,const  int p,const  int k,const int M)
{
	int indc=threadIdx.y+blockDim.y*blockIdx.y;		
	if (indc<M-k-1)
		{
		extern __shared__ float data2[];
		float* YvecX=(float*) &data2[0];
		float* YvecY=(float*) &data2[blockDim.x*p];
		float* CvecX=(float*) &data2[2*blockDim.x*p];
		float* CvecY=(float*) &data2[2*blockDim.x*p+blockDim.y];
		float* C0=(float*) &data2[2*blockDim.x*p+2*blockDim.y];
		int indg=threadIdx.x+blockDim.x*blockIdx.x;
		int indy=indg+(indc+1)*p*blockDim.x*gridDim.x;
		if(threadIdx.y==0)	
			{
			//#pragma unroll 
			for (int w=0;w<p;w++)
				{
				YvecX[threadIdx.x+blockDim.x*w]=Y[indg+w*blockDim.x*gridDim.x].x;
				YvecY[threadIdx.x+blockDim.x*w]=Y[indg+w*blockDim.x*gridDim.x].y;
				} 
			}
		if(threadIdx.x==0) 
			{
			CvecX[threadIdx.y]=c[indc+1].x;
			CvecY[threadIdx.y]=c[indc+1].y;
		if(threadIdx.y==0){C0[0]=c[0].x;}	
			}
		
		__syncthreads();

		#pragma unroll
		for(int w=0;w<p;w++)
			{
			Y[indy+w*blockDim.x*gridDim.x].x=Y[indy+w*blockDim.x*gridDim.x].x-(YvecX[threadIdx.x+blockDim.x*w]*CvecX[threadIdx.y]-YvecY[threadIdx.x+blockDim.x*w]*CvecY[threadIdx.y])*(C0[0]*C0[0]);
			Y[indy+w*blockDim.x*gridDim.x].y=Y[indy+w*blockDim.x*gridDim.x].y-(YvecX[threadIdx.x+blockDim.x*w]*CvecY[threadIdx.y]+YvecY[threadIdx.x+blockDim.x*w]*CvecX[threadIdx.y])*(C0[0]*C0[0]);
			//Y[indy+w*blockDim.x*gridDim.x].x=Y[indy+w*blockDim.x*gridDim.x].x+(YvecX[threadIdx.x+blockDim.x*w]*CvecX[threadIdx.y]-YvecY[threadIdx.x+blockDim.x*w]*CvecY[threadIdx.y])*C0[0];
			//Y[indy+w*blockDim.x*gridDim.x].y=Y[indy+w*blockDim.x*gridDim.x].y+(YvecX[threadIdx.x+blockDim.x*w]*CvecY[threadIdx.y]+YvecY[threadIdx.x+blockDim.x*w]*CvecX[threadIdx.y])*C0[0];
			
			}



		}
	return;
}


__global__ void subMatrixVecScal2(complex* Y,complex* c,complex* L,const  int p,const  int k,const int M)
{
complex Cvec[rnum];
complex Out[rnum];
complex G;
complex X;
float c0;
int indg=threadIdx.x+blockDim.x*blockIdx.x;
int indc=rnum*(threadIdx.y+blockDim.y*blockIdx.y);
c0=c[0].x;
G=Y[indg];
#pragma unroll
for(int i=0;i<__min(rnum,M-k-1-indc-1);i++)
	{Cvec[i]=c[indc+i+1];	
	Out[i]=Y[indg+blockDim.x*gridDim.x*(indc+i+1)];
	
	}
for(int i=0;i<__min(rnum,M-k-1-indc-1);i++)
	{	
	Out[i]=Out[i]-G*Cvec[i]*(c0*c0);
	}
for(int i=0;i<__min(rnum,M-k-1-indc-1);i++)
	{	
	Y[indg+blockDim.x*gridDim.x*(indc+i+1)]=Out[i];
	}
	

	return;
}



__global__ void SetmatL(complex* L,complex* c, int k,int M)
	{int o=M*k-(k*(k-1))/2;
	if (threadIdx.y==0)
		{L[o]=c[0];}
	else
		{L[o+threadIdx.y]=-c[threadIdx.y]*(c[0].x*c[0].x);}
	return;
	}


//----------------------------------------------------------------------------------------------------------------------------------
// обновляем подматрицу

__global__ void VecMulVechSumMatr(complex* L, const  int M,const  int k)
	{
	int o=M*k-(k*(k-1))/2;
	int h;
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int j=threadIdx.y;
	//if(i<k)
	//{
		h=M*i-(i*(i-1))/2+k-i+1;
		L[h+j]=L[h+j]+L[h-1]*L[o+j+1];
	//}
	return;
	}


// умножаем подстроку на скаляр,для k = номер текущего столбца
__global__ void RowScalarMul(complex* L, const int M,const int k)
{
	int o=M*k-(k*(k-1))/2;//индекс начала столбца k
	int i=threadIdx.x;
	int h;
	h=M*i-(i*(i-1))/2+k-i;//индекс начала столбца i
	L[h]=L[h]*L[o].x;
	return;
}
	
__global__ void warm(int N)
	{return;}

__global__ void mult(complex* Y,complex* z,int p,int N)
	{complex r;
r.zero();
for(int i=0; i<p;i++)
	{r=r+AhMulB(Y[i],Y[i+N]);}
z[0]=r;
	return;
	}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Функции ЦПУ


// Заполняем матрицу случайными числами
void randMatrixComplex(complex* Y,const int M,const int N,const float low, const float hi)
{
	for (int i=0;i<M*N;i++)
	{Y[i]=complex((float)rand() / RAND_MAX  * (hi - low) + low, (float)rand() / RAND_MAX  * (hi - low) + low);}
	return;
}

// Заполняем матрицу неслучайными числами
void SeqMatrixComplex(complex* Y,const int M,const int N)
{
	for (int i=0;i<M*N;i++)
	{Y[i]=complex((float)i,(float)i);}
	return;
}

// Печать комплексной матрицы
void printMatrixComplex(const complex* Y,const int M,const int N)
{
	for (int i=0;i<M;i++)
	{
		for (int j=0;j<N;j++)
		{printf("%12.4e%+12.4ei  ",Y[IDX2C(i,j,N)].x,Y[IDX2C(i,j,N)].y);}
		printf("\n");
	}
	printf("\n");
	return;
}

// Печать комплексной диагональной матрицы
void printMatrixComplexDiag(const complex* Y,const int M,const int N)
{
	int o=0;
	for (int i=0;i<M;i++)
	{
		for (int j=0;j<N-i;j++)
		{
			printf("%12.4e%+12.4ei  ",Y[o].x,Y[o].y);
			o++;
		}
		printf("\n");
	}
	printf("\n");
	return;
}

//Находим длину вектора
complex dotSqrtRecipRealHost(complex* Y,int N)
{
	complex sum;
	sum.zero();
	for(int i=0;i<N;i++)
		{sum.x+=Y[i].norm();}
	sum.x=1/sqrt(sum.x);
	return sum;
}

// Находим скалярные произведения первой строки и последующих
void MatVecMulComplexHost(const complex* Y,const int N,const int M, complex* L)
{
	complex sum;
	for (int j=1;j<M;j++) // цикл с 1 т.к. перая строка уже посчитана в другом месте
	{
		sum.zero();
		for(int i=0;i<N;i++)
			{sum=sum+AhMulB(Y[i],Y[i+j*N]);}
		L[j]=sum;		
	}
	return;
}

// Находим поекцию A=A-l*g , заодно расчитываем очередной столбец матрицы L
void subMatrixVecScalHost(complex* Y, complex* L, const int N, const int M)
{
	for(int j=1;j<M;j++)
	{	for(int i=0;i<N;i++)
		{	Y[i+j*N]=Y[i+j*N]-Y[i]*L[j]*(L[0].x*L[0].x);}
	L[j]=-L[j]*L[0].x*L[0].x;
	}
	return;
}

// умножаем подстроку на скаляр,для k = номер текущего столбца
void RowScalarMul_Host(complex* L, const int M,const int k)
{
	int o=M*k-(k*(k-1))/2;//индекс начала столбца k
	int h;
	for(int i=0;i<k;i++)
	{	h=M*i-(i*(i-1))/2+k-i;//индекс начала столбца i
		L[h]=L[h]*L[o].x;
	}
	return;
}

// обновляем подматрицу
void VecMulVechSumMatr_Host(complex* L,const int M, const int k)
{
	int o=M*k-(k*(k-1))/2;//индекс начала столбца k
	int h;
	for(int i=0;i<k;i++)
	{	h=M*i-(i*(i-1))/2+k-i+1;//индекс для строки
		for(int j=0;j<M-k-1;j++)
		{L[h+j]=L[h+j]+L[h-1]*L[o+j+1];}
	}
	return;
}

// Произведение матрицы на  сопряженную Y*Yh
void AMulAh_Host(const complex* Y,complex* R,const  int N,const  int M)
{
	complex sum;
	for( int k=0;k<M;k++)
	{
		for( int i=0;i<M;i++)
		{	sum.zero();
			for( int j=0;j<N;j++)
			{sum=sum+Y[k*N+j]*~Y[i*N+j];}
			R[k*M+i]=sum;
		}
	}
	return;
}

// Произведение сопряженной треугольной матрицы на себя
void LhMulL_Host(const complex* L,complex* R,const  int M)
{
	int oi=0;
	int ok=0;
	complex sum;
	for( int k=0;k<M;k++)
	{	ok=M*(k+1)-((k+1)*(k))/2-1;////индекс конца столбца k
		for( int i=0;i<M;i++)
		{	oi=M*(i+1)-((i+1)*(i))/2-1;//индекс конца столбца i
			sum.zero();
			for( int j=0;j<__min(M-k,M-i);j++) // только min(M-k,M-i) не нулевые
			{	sum=sum+~L[ok-j]*L[oi-j];}
			R[k*M+i]=sum;
		}
	}
	return;
}

//Произведение двух матриц
void AMulB_Host(const complex* A,const complex* B,complex* C, const  int M, const  int N,const  int P )
{
	complex sum;
	for( int k=0;k<M;k++)
	{
		for( int i=0;i<N;i++)
		{
			sum.zero();
			for( int j=0;j<P;j++)
			{	sum=sum+A[k*P+j]*B[j*P+i];}
			C[k*N+i]=sum;
		}
	}
	return;
}


float absEyeMatr_Host(const complex* Y, const int M)
{
	float sum=0.0f;
	int p=0;
	for(int j=0;j<M-1;j++)
	{	complex Z;
		Z=Y[p]-complex(1.0f,0.0f);
		sum=sum+Z.norm();
		p++;
		for(int i=0;i<M;i++)
		{	sum=sum+Y[p].norm();
			p++;
			//printf("%4d %12.4e\n",p,sum);
		}
	}
	return sum;
}
/*

int divMaxint(int A, int B)
	{int r=(int)ceilf((float)A/(float)B);return r;}

/////////////////////////

// utilities and system includes
//#include <shrUtils.h>

// CUDA-C includes
//#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
DevicePropertymain( int argc, const char** argv) 
{
    shrSetLogFileName ("deviceQuery.txt");
    shrLog("%s Starting...\n\n", argv[0]); 
    shrLog(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");
        
    int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		shrLog("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
		shrLog("\nFAILED\n");
		shrEXIT(argc, argv);
	}

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        shrLog("There is no device supporting CUDA\n");

    int dev;
	int driverVersion = 0, runtimeVersion = 0;     
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                shrLog("There is no device supporting CUDA.\n");
            else if (deviceCount == 1)
                shrLog("There is 1 device supporting CUDA\n");
            else
                shrLog("There are %d devices supporting CUDA\n", deviceCount);
        }
        shrLog("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    #if CUDART_VERSION >= 2020
        // Console log
		cudaDriverGetVersion(&driverVersion);
		shrLog("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		shrLog("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
    #endif
        shrLog("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		char msg[256];
		sprintf(msg, "  Total amount of global memory:                 %llu bytes\n", (unsigned long long) deviceProp.totalGlobalMem);
		shrLog(msg);
    #if CUDART_VERSION >= 2000
        shrLog("  Multiprocessors x Cores/MP = Cores:            %d (MP) x %d (Cores/MP) = %d (Cores)\n", 
			deviceProp.multiProcessorCount,
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    #endif
        shrLog("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem); 
        shrLog("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        shrLog("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        shrLog("  Warp size:                                     %d\n", deviceProp.warpSize);
        shrLog("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        shrLog("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        shrLog("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        shrLog("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
        shrLog("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        shrLog("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    #if CUDART_VERSION >= 2000
        shrLog("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 2020
        shrLog("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        shrLog("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
        shrLog("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        shrLog("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
			                                                            "Default (multiple host threads can use this device simultaneously)" :
		                                                                deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
		                                                                deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
    #endif
    #if CUDART_VERSION >= 3000
        shrLog("  Concurrent kernel execution:                   %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 3010
        shrLog("  Device has ECC support enabled:                %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
    #endif
    #if CUDART_VERSION >= 3020
		shrLog("  Device is using TCC driver mode:               %s\n", deviceProp.tccDriver ? "Yes" : "No");
    #endif
	}

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name 
    shrLog("\n");    
   	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";        
       
       
	char cTemp[10];
    
    // driver version
    sProfileString += ", CUDA Driver Version = ";
    #ifdef WIN32
	    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, driverVersion%100);    
    #else
	    sprintf(cTemp, "%d.%d", driverVersion/1000, driverVersion%100);	
    #endif
    sProfileString +=  cTemp;
    
    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
    #ifdef WIN32
	    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
    #else
	    sprintf(cTemp, "%d.%d", runtimeVersion/1000, runtimeVersion%100);
    #endif
    sProfileString +=  cTemp;  
    
    // Device count      
    sProfileString += ", NumDevs = ";
    #ifdef WIN32
        sprintf_s(cTemp, 10, "%d", deviceCount);
    #else
        sprintf(cTemp, "%d", deviceCount);
    #endif
    sProfileString += cTemp;
    
    // First 2 device names, if any
    for (dev = 0; dev < ((deviceCount > 2) ? 2 : deviceCount); ++dev) 
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += ", Device = ";
        sProfileString += deviceProp.name;
    }
    sProfileString += "\n";
    shrLogEx(LOGBOTH | MASTER, 0, sProfileString.c_str());

    // finish
    shrLog("\n\nPASSED\n");
    //shrEXIT(argc, argv);
	return 0;
}
*/