/****
 * 2D-Matrix-Matrix Multiplication using MPI
 * Task-1
 ****/

#include <stdio.h>
#include<stdlib.h>
#include<mpi.h>

#define SIZE 1024			/* SIZE should be a multiple of number of nodes*/
#define DEBUG 0			/* Set it to 1 - To view detailed Output */
#define MASTER 1
#define SLAVE 2

MPI_Status status;

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];
static double b_transpose[SIZE][SIZE];

static void initialization()
{
	int i,j; 
	for(i = 0;i<SIZE;i++)
	{
		for(j = 0;j<SIZE;j++)
		{
			a[i][j] = 1.0;
            if (i >= SIZE/2) a[i][j] = 2.0;
            b[i][j] = 1.0;
            if (j >= SIZE/2) b[i][j] = 2.0;
			
			c[i][j] = 0.0; /*Initially*/
			b_transpose[j][i] = b[i][j]; /*Since it is easy to send rows than columns */
		}
	}
}

static void output()
{
	int i, j;
	printf("\n Final Output is:\n");
	for(i = 0;i<SIZE;i++)
	{
		for(j = 0;j<SIZE;j++)
		{
			printf("%7.2f", c[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char **argv)
{
	int prank, np;
	int cols; 																				/* number of columns per worker */
	int rows; 																				/* number of rows per worker */
	int mtype; 																				/* message type */
	int dest, src, offset1, offset2, new_offset1;
	double start_t, end_t;
	int i, j ,k;
	
	MPI_Init(&argc,&argv);											
	MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	
	if(np==1)                                                        /* Number of nodes is 1 */ 
	{
		initialization();
		printf("\n SIZE = %d, number of  nodes = %d\n",SIZE,np);
		start_t = MPI_Wtime();
		
		for (i = 0; i < SIZE; i++)
			for (j = 0; j < SIZE; j++) {
				c[i][j] = 0.0;
				for (k = 0; k < SIZE; k++)
					c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		end_t = MPI_Wtime();
		
		if(DEBUG)
			output();
			
			printf("\n Execution time on %d nodes is %f\n", np, end_t-start_t);
	}

	if(np==2)  /* Number of nodes is 2*/
	{
		if(prank == 0)
		{
			initialization();
			printf("\n SIZE = %d, number of  nodes = %d\n",SIZE,np);
			start_t = MPI_Wtime();
			mtype = MASTER;
			rows = (SIZE/np)*2;
			offset1 = rows;
			cols = SIZE/2;
			offset2 = cols;
			new_offset1 = 0;
			
			for(dest = 1;dest <np;dest++)
			{
				MPI_Send(&new_offset1, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&offset2, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&a[0][0], SIZE*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&b_transpose[offset2][0], offset2*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
			}
			/*Master calculation part*/
			for(i = 0; i<rows;i++)
			{
				for(j = 0; j<cols;j++)
				{
					for(k = 0;k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
					}
				}
			}
			
			/*Collect results from slave*/
			mtype = SLAVE;
			for(src = 1;src<np;src++)
			{
				MPI_Recv(&new_offset1, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&rows, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&offset2, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
		        for(i = new_offset1;i<rows;i++)
				{
					MPI_Recv(&c[i][offset2], cols, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);

				}					
			
			}
			
			end_t = MPI_Wtime();
			
			if(DEBUG)
				output();
			
			printf("\n Execution time on %d nodes is %f\n", np, end_t-start_t);			
		}
		if(prank >0)
		{
			mtype = MASTER;
			MPI_Recv(&new_offset1, 1, MPI_INT, 0, mtype,MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset2, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[0][0], SIZE*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b_transpose[offset2][0], offset2*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/*Slaves calculation part*/
			
			for(i = new_offset1;i<rows;i++)
			{
				for(j = offset2;j<SIZE;j++)
				{
					for(k = 0;k<SIZE;k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
					}
				}
			}
			if(DEBUG){
					printf("\n Rank is %d Calculated matrix part is:\n",prank);
					for(i = new_offset1;i<rows;i++)
						for(j=offset2;j<SIZE;j++)
							printf("%7.2f",c[i][j]);
						printf("\n");
				}
			
			mtype = SLAVE;
			MPI_Send(&new_offset1, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset2, 1,MPI_INT, 0, mtype, MPI_COMM_WORLD);
			for(i = new_offset1;i<rows;i++)
			{
				MPI_Send(&c[i][offset2], offset2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}
		}
		
	}
	
	if(np>=4)  /*Number of nodes greater than or equal to 4*/
	{
		if (prank == 0) 										/*Master Job */
		{
			initialization(); 
			printf("\nSIZE = %d, number of nodes = %d\n", SIZE, np);
			start_t = MPI_Wtime();
			mtype = MASTER;
			rows = (SIZE/np)*2;																	/* Number of rows in matrix A for each node*/
			offset1 = rows;
			cols = SIZE/2; 																		/* Number of cols in matrix B which is same as number of rows in B-Transpose for each node*/
			offset2 = cols;
			new_offset1 = 0;
		
			for(dest = 1; dest <np/2; dest++) /*Sending data for first set of slaves*/
			{
				MPI_Send(&offset1, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&a[offset1][0], rows*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD); 	  /* STart element in A matrix */
				MPI_Send(&b_transpose[0][0], cols*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD); /* number of cols in matrix b is equal to number of rows in b_transpose */
				offset1 = offset1 + rows;
			}
				
			for(dest = np/2;dest<np;dest++) /*Sending data for second set of slaves*/
			{
				MPI_Send(&new_offset1, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&offset2, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
				MPI_Send(&a[new_offset1][0], rows*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD); /* Starting point in A matrix */ 
				MPI_Send(&b_transpose[offset2][0], cols*SIZE, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD); /* number of cols in matrix b is equal to number of rows in b_transpose */
				new_offset1 = new_offset1 + rows;
			}
			
			/* Master Calculation Part */

			for(i = 0; i<rows; i++)
			{
				for(j = 0; j< cols; j++)
				{
					for(k = 0; k <SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
					}
				}
			}
			
						
			/* collect part-1 results from slaves */

			mtype = SLAVE;
			for(src = 1; src < np/2; src++)
			{
				MPI_Recv(&offset1, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&rows, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&cols, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				for(i = offset1;i<offset1+rows;i++)
				{
					MPI_Recv(&c[i][0], cols, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
				}
				
				if(DEBUG){
				printf("\n Received this part from process %d \n",src);
				for(i = offset1;i<offset1+rows;i++)
					for(j=0;j<cols;j++)
						printf("%7.2f",c[i][j]);
					printf("\n");
				}
			} 
			if(DEBUG){
				printf("\n Rank 0 calculated\n");
				for(i = 0; i<rows; i++)
					for(j = 0; j< cols; j++)
						printf("%7.2f",c[i][j]);
					printf("\n");
			}
			mtype = SLAVE;
			for(src = np/2;src<np;src++)
			{
				MPI_Recv(&new_offset1, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&rows, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&offset2, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				MPI_Recv(&cols, 1, MPI_INT, src, mtype, MPI_COMM_WORLD, &status);
				for(i = new_offset1;i<new_offset1+rows;i++)
				{
					MPI_Recv(&c[i][offset2], cols, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
				}
				if(DEBUG){
					printf("\n Received this part from process %d \n",src);
					for(i = new_offset1;i<rows;i++)
						for(j=offset2;j<SIZE;j++)
							printf("%7.2f",c[i][j]);
						printf("\n");
				}
			}
			 
			if(DEBUG){
				printf("\n Rank 0 calculated\n");
				for(i = 0; i<rows; i++)
					for(j = 0; j< cols; j++)
						printf("%7.2f",c[i][j]);
					printf("\n");
			}
		
		end_t = MPI_Wtime();

		if(DEBUG)
		output(); 

		printf("\n Execution time on %d nodes is: %f ", np, end_t-start_t);	
	}

		else if(prank>0 && prank<np/2) /* First set of slaves */ 
		{																												/* Receive data from Master */
																													
			mtype = MASTER;
			MPI_Recv(&offset1, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[offset1][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b_transpose[0][0], cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
	
			/* slaves calculation part-1 */

			for(i = offset1; i<offset1+rows;i++)
			{
				for(j=0;j< cols;j++)
				{
					for(k=0;k< SIZE;k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
					}
				}
			}
		
			if(DEBUG){
				printf("\n Rank is %d Calculated matrix part is:\n",prank);
				for(i = offset1;i<offset1+rows;i++)
					for(j=0;j<cols;j++)
						printf("%7.2f",c[i][j]);
					printf("\n");
			}
			/*Now sending  part-1 of the results back to the master */
			
			mtype = SLAVE;
			MPI_Send(&offset1, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			for(i = offset1;i<offset1+rows;i++)
			{
				MPI_Send(&c[i][0], cols, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}
			}
	
		else if(prank>=np/2) /*Second set of slaves */ 
		{
			/*Receiving data from master*/ 
	
			mtype = MASTER;
			MPI_Recv(&new_offset1, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset2, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[new_offset1][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b_transpose[offset2][0], cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Slaves calculations part-2 */ 
		
			for(i=new_offset1;i<new_offset1 + rows;i++)
			{
				for(j=offset2;j<SIZE;j++)
				{
					for(k = 0; k<SIZE;k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b_transpose[j][k];
					}
				}
			}
			
			if(DEBUG){
					printf("\n Rank is %d Calculated matrix part is:\n",prank);
					for(i = new_offset1;i<rows;i++)
						for(j=offset2;j<SIZE;j++)
							printf("%7.2f",c[i][j]);
						printf("\n");
				}
			
			
			/* Sending Part-2 of the results back to the master */ 
	
			mtype = SLAVE;
			MPI_Send(&new_offset1, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset2, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			for(i = new_offset1;i<new_offset1+rows;i++)
			{
				MPI_Send(&c[i][offset2], cols, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}
		}
	}

MPI_Finalize();
return 0;

}