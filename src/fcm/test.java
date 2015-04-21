package fcm;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import Jama.Matrix;
public class test {
	//private static DecimalFormat df=new DecimalFormat("0.00");
	public static void main(String [] args){
		Matrix inputdata = null;
		try{
			FileReader in=new FileReader("./src/fcm/test1.txt");
			BufferedReader bReader=new BufferedReader(in);
			inputdata=Matrix.read(bReader);
			}catch(Exception ex){
				ex.printStackTrace();
			}
		 int inputrows=inputdata.getRowDimension();
		 int inputcols=inputdata.getColumnDimension();
		 Matrix imatrix=Matrix.identity(inputrows, inputcols);
		 double [] zero=new double[inputrows];
		 double [] cross=new double[inputrows];
		 Matrix D=Matrix.identity(inputrows, inputcols);//D为对角矩阵
		 for(int i=0;i<inputrows;i++){
			 Matrix initial=new Matrix(zero,inputrows);
			 initial.set(i, 0, 1);
			 cross[i]=inputdata.times(initial).norm1();
		 }
		 for(int j=0;j<inputrows;j++){
		     D.set(j, j, cross[j]);
		 }
		 //L为拉普拉斯矩阵
		 Matrix L=D.minus(inputdata);
		 //eig为初始特征向量矩阵
		 Matrix ieig=L.eig().getV();
		 Matrix teigMatrix=L.eig().getD();
		 ArrayList<Integer> list=new ArrayList<Integer>();
		 //筛选特征向量
		 for(int k=0;k<inputrows;k++){
			 if(teigMatrix.get(k, k)!=0.0){
				 list.add(k);
			 }
		 }
		 int [] selectEig=new int[list.size()];
		 for(int m=0;m<selectEig.length;m++){
			 selectEig[m]=list.get(m);
		 }
		 Matrix eig=ieig.getMatrix(0,inputrows-1,selectEig);
		 FCM(eig,2,2,100,0.00001);
	}
	//data 矩阵m*n,有m个具有n维特征的样本；cluster_n：聚类个数；Uindex:目标函数中隶属度矩阵的指数;
	//maxIterate_n:最大迭代次数；udegreechange:隶属度变化最小，迭代终止条件；
	public static  void FCM(Matrix data,int cluster_n,double Uindex,int maxIterate_n,double udegreechange){
		int data_rows_n=data.getRowDimension();
		int data_cols_n=data.getColumnDimension();
		//1.初始化隶属度矩阵
		double [][] arrayU=new double[data_rows_n][cluster_n];
		initialMOC(arrayU);
		Matrix U=new Matrix(arrayU);
		double [][]tempdata=data.getArray();
		double sum=0;//更新隶属度矩阵计算分母的和
		double objfcn1=0;
		double objfcn2=0;
		double temp=0;
		double sum1=0;
		double sum2=0;
		double [][]centroid=new double[cluster_n][data_cols_n];
		//初始化聚类中心
		for(int i=0;i<cluster_n;i++){
			for(int j=0;j<data_cols_n;j++){
				temp=0;sum1=0;sum2=0;
				for(int k=0;k<data_rows_n;k++){
					temp=Math.pow(U.get(k, i), Uindex);
					sum1+=temp*data.get(k, j);
					sum2+=temp;
				}
				centroid[i][j]=sum1/sum2;
			}
		}
		while(true)
		{//前后目标函数的差值是否达到阀值
			objfcn1=objectFunction(data, U, Uindex, centroid);
			//2.更新隶属度矩阵
			for(int x=0;x<data_rows_n;x++){
				for(int y=0;y<cluster_n;y++){
					if(disfcn(tempdata[x],centroid[y])==0){
						U.set(x, y, 1);
					}
					else{
						sum=0;
						for(int z=0;z<cluster_n;z++){
							sum+=Math.pow(disfcn(tempdata[x],centroid[y])/disfcn(tempdata[x],centroid[z]),2/(Uindex-1));
						}
						U.set(x, y, 1/sum);
					}
				}
			}
			//3.更新聚类中心
			for(int i=0;i<cluster_n;i++){
				for(int j=0;j<data_cols_n;j++){
					temp=0;sum1=0;sum2=0;
					for(int k=0;k<data_rows_n;k++){
						temp=Math.pow(U.get(k,i), Uindex);
						sum1+=temp*data.get(k, j);
						sum2+=temp;
					}
					centroid[i][j]=sum1/sum2;
				}
			}
			//4.计算目标函数差值
			objfcn2=objectFunction(data, U, Uindex, centroid);
			if(Math.abs(objfcn1-objfcn2)<udegreechange){
				break;
			}
			else{
				continue;
			}
		}
		//输出打印测试
		U.print(1, 10);
		Matrix outputCentroid=new Matrix(centroid);
		outputCentroid.print(1, 5);
	}
	//初始化隶属度矩阵
	public static void initialMOC(double [][]a){
     	 int sample_n=a.length;
     	 int cluster_n=a[0].length;
     	 double sum=0;
     	 /*for(int i=0;i<sample_n;i++){
     		 a[i][0]=Math.random();
     		 for(int j=0;j<cluster_n-1;j++){
     			 sum+=a[i][j];
     			 a[i][j+1]=Math.random()*(1-sum);
     		 }
     		 a[i][cluster_n-1]=1-sum;
     		 sum=0;
     	 }*/
     	 for(int i=0;i<sample_n;i++){
     		 for(int j=0;j<cluster_n;j++){
     			 a[i][j]=Math.random();
     		 }
     	 }
     	 for(int i=0;i<sample_n;i++){
     		 for(int j=0;j<cluster_n;j++){
     			 sum+=a[i][j];
     		 }
     		 for(int k=0;k<cluster_n;k++){
     			 a[i][k]/=sum;
     		 }
     		 sum=0;
     	 }
      }
	//计算目标函数
	public static double objectFunction(Matrix data,Matrix MOC,double Uindex,double [][]centroid){
		double objfcn=0;
		int data_rows=data.getRowDimension();
		int MOC_cols=MOC.getColumnDimension();
		double [][]sampledata=data.getArray();
		for(int j=0;j<data_rows;j++){
			for(int k=0;k<MOC_cols;k++){
				objfcn=Math.pow(MOC.get(j, k),Uindex)*Math.pow(disfcn(sampledata[j], centroid[k]), 2);
			}
		}
		return objfcn;
	}
	//计算两个数据点的距离
	public static double disfcn(double[]x1,double[]x2){
		double result=0;
		double sum=0;
		for(int i=0;i<x1.length;i++){
			sum+=Math.pow((x1[i]-x2[i]), 2);
		}
		result=Math.sqrt(sum);
		return result;
	}
	public static void outputArray(int[] selectEig){
		for(int i=0;i<selectEig.length;i++){
			System.out.print(selectEig[i]+" ");
		}
	}
}
