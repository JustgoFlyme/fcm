package fcm;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;

import Jama.Matrix;
public class test {
	public static Matrix inputdata = null;
	//private static DecimalFormat df=new DecimalFormat("0.00");
	public static void main(String [] args){
		//�����ڽӾ���
		try{
			FileReader in=new FileReader("./src/fcm/test.txt");
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
		 Matrix D=Matrix.identity(inputrows, inputcols);//DΪ�ԽǾ���
		 for(int i=0;i<inputrows;i++){
			 Matrix initial=new Matrix(zero,inputrows);
			 initial.set(i, 0, 1);
			 cross[i]=inputdata.times(initial).norm1();
		 }
		 for(int j=0;j<inputrows;j++){
		     D.set(j, j, cross[j]);
		 }
		 //LΪ������˹����
		 Matrix L=D.minus(inputdata);
		 L.print(1, 0);
		 //eigΪ��ʼ������������
		 Matrix ieig=L.eig().getV();
		 Matrix teigMatrix=L.eig().getD();
		 ArrayList<Integer> list=new ArrayList<Integer>();
		 //ȥ������ֵ0��Ӧ����������
		 for(int k=0;k<inputrows;k++){
			 if(teigMatrix.get(k, k)>0.1){
				 list.add(k);
			 }
		 }
		 teigMatrix.print(1, 5);
		 int [] selectEig=new int[list.size()];
		 for(int m=0;m<selectEig.length;m++){
			 selectEig[m]=list.get(m);
		 }
		 Arrays.sort(selectEig);
		 int [] direct=new int[3];
		 System.arraycopy(selectEig, 0, direct, 0, 3);
		 outputArray(direct);
		 Matrix eig=ieig.getMatrix(0,inputrows-1,direct);
		 FCM(eig,3,2,100,0.000001);
	}
	//data ����m*n,��m������nά������������cluster_n�����������Uindex:Ŀ�꺯���������Ⱦ����ָ��;
	//maxIterate_n:������������udegreechange:�����ȱ仯��С��������ֹ������
	public static  void FCM(Matrix data,int cluster_n,double Uindex,int maxIterate_n,double udegreechange){
		int data_rows_n=data.getRowDimension();
		int data_cols_n=data.getColumnDimension();
		//1.��ʼ�������Ⱦ���
		double [][] arrayU=new double[data_rows_n][cluster_n];
		initialMOC(arrayU);
		Matrix U=new Matrix(arrayU);
		double [][]tempdata=data.getArray();
		double sum=0;//���������Ⱦ�������ĸ�ĺ�
		double objfcn1=0;
		double objfcn2=0;
		double temp=0;
		double sum1=0;
		double sum2=0;
		double [][]centroid=new double[cluster_n][data_cols_n];
		//��ʼ����������
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
		{//ǰ��Ŀ�꺯���Ĳ�ֵ�Ƿ�ﵽ��ֵ
			objfcn1=objectFunction(data, U, Uindex, centroid);
			//2.���������Ⱦ���
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
			//3.���¾�������
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
			//4.����Ŀ�꺯����ֵ
			objfcn2=objectFunction(data, U, Uindex, centroid);
			if(Math.abs(objfcn1-objfcn2)<udegreechange){
				break;
			}
			else{
				continue;
			}
		}
		//�����ӡ����
		U.print(1, 10);
		Matrix moc=U.copy();
		for(int m=0;m<moc.getRowDimension();m++){
			for(int n=0;n<moc.getColumnDimension();n++){
				if(moc.get(m, n)>0.4){
					moc.set(m, n, 1);
				}
				else{
					moc.set(m, n, 0);
				}
			}
		}
		System.out.print("��������Ч����"+EQ(inputdata,moc));
		Matrix outputCentroid=new Matrix(centroid);
		outputCentroid.print(1, 5);
	}
	//��ʼ�������Ⱦ���
	public static void initialMOC(double [][]a){
     	 int sample_n=a.length;
     	 int cluster_n=a[0].length;
     	 double sum=0;
     	 for(int i=0;i<sample_n;i++){
     		 a[i][0]=Math.random();
     		 for(int j=0;j<cluster_n-1;j++){
     			 sum+=a[i][j];
     			 a[i][j+1]=Math.random()*(1-sum);
     		 }
     		 a[i][cluster_n-1]=1-sum;
     		 sum=0;
     	 }
     	/*
     	 //��һ�ַ�����ʼ�������Ⱦ���
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
     	 }*/
      }
	//����Ŀ�꺯��
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
	//�����������ݵ�ľ���
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
	//�������Ч��
	public static double EQ(Matrix Adjacent,Matrix MOC){
		int nodeNum=0;
		int vclusterNum=0;
		int wclusterNum=0;
		int vdegree=0;
		int wdegree=0;
		double sum=0;
		int edgeNum=0;
		ArrayList al=new ArrayList();
		for(int x=0;x<Adjacent.getRowDimension();x++){
			for(int y=0;y<Adjacent.getColumnDimension();y++){
				edgeNum+=Adjacent.get(x,y);
			}
		}
		edgeNum/=2;
		for(int i=0;i<MOC.getColumnDimension();i++){
			//���������ڲ��ڵ�
			for(int j=0;j<MOC.getRowDimension();j++){
				if(MOC.get(j, i)==1){
					nodeNum+=1;
					al.add(j);
				}	
			}
			for(int k=0;k<nodeNum;k++){
				for(int l=0;l<nodeNum;l++){
					//����ڵ�v��w�����������ĸ���
					for(int m=0;m<MOC.getColumnDimension();m++){
						vclusterNum+=(int)MOC.get((int) al.get(k), m);
						wclusterNum+=(int)MOC.get((int)al.get(l), m);
					}
					//����ڵ�v��w�Ķ�
					for(int n=0;n<Adjacent.getColumnDimension();n++){
						vdegree+=Adjacent.get((int)al.get(k),n);
						wdegree+=Adjacent.get((int)al.get(l),n);
					}
					sum+=(1/((double)vclusterNum*wclusterNum))*(Adjacent.get((int) al.get(k), (int)al.get(l))-vdegree*wdegree/((double)2*edgeNum));
					vclusterNum=0;wclusterNum=0;vdegree=0;wdegree=0;
					System.out.println("��ǰֵΪ��"+sum);
				}
			}
			nodeNum=0;
		}
		return sum/(2*edgeNum);
	}
}
