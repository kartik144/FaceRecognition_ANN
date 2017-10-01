import java.io.*;
import java.util.*;

class Sunglasses
{
	int nh=3,no=1,ni=960;	// variables to store number of nodes in hidden layer,output layer and input layer respectively
	float x[]=new float[ni+1]; // array to store the inputs(pixels)
	float h[]=new float[nh+1]; // array to store the hidden layer values
	float o[]=new float[no]; // array to store the (calculated) output values
	float t[]=new float[no]; // array to store the target output values
	float Whx[][]=new float [nh+1][ni+1]; // 2-D arry to store the weights of all inputs for each hidden layer node
	float Woh[][]=new float[no][nh+1]; //2-D array to store the weights of all 
	float learningRate=0.3f; 
	float momentum=0.3f;
	float maxIter=10000; //max number of epochs 
	float minError=0.005f;
	float delK[]=new float[no]; // array to  store error in each node of hidden layer with respect to output layer
	float delH[]=new float[nh+1]; // array to store error in each node of input layer weight with respect to hidden layer 
	float w1[][]=new float [no][nh+1]; // array to store the prevoius change in weights of Woh
	float w2[][]=new float [nh+1][ni+1]; // array to store the prevois change in weights of Whx
	
	void randomWeights()
	{
		for(int i=0;i<nh+1;i++)
		for(int j=0;j<ni+1;j++)
			Whx[i][j]=((float)(Math.random()*100)-50)/100f;
			
		for(int i=0;i<no;i++)
		for(int j=0;j<nh+1;j++)
			Woh[i][j]=((float)(Math.random()*100)-50)/100f;
	}//function to assign random weights to Whx and Woh
	
	float sigmoid(float f)
	{
		return (float)(1/(1+Math.exp(-f)));
	}// function to return sigmoid of the value passed
	
	String getFileName(String s)
	{
		int i1=0;
		
		for(int i=s.length()-1;i>=0;i--)
			if(s.charAt(i)=='/')
			{
				i1=i+1;
				break;
			}
			
		String file_name="";
		
		for(int i=i1;i<s.length();i++)
		file_name=file_name+s.charAt(i);
		
		return file_name;
	}// function to return the name of the file from its path which is passed as parameter
	
	void getTargetValue(String s)
	{
		if(s.charAt(s.length()-7)=='s')
			t[0]=1;
		else
			t[0]=0;
	}//function to get the target output value from the file path passed as parameter
	
	void feedForward()
	{
		float sum=0;
		h[0]=1f;//threshold
		for(int i=1;i<nh+1;i++)
		{
			sum=0;
			for(int j=0;j<ni+1;j++)
				sum=sum+Whx[i][j]*x[j];

			h[i]=sigmoid(sum);
		}//calculates the values at hidden layer nodes

		for(int i=0;i<no;i++)
		{
			sum=0;

			for(int j=0;j<nh+1;j++)
				sum=sum+Woh[i][j]*h[j];

			o[i]=sigmoid(sum);
		}// calculates the values at output layer nodes
	}// function to calculte the values at various nodes 

	void calculateError()
	{ 	
		float sum=0;
		
		for(int k=0; k<no ;k++)
		{
			delK[k]=o[k]*(1-o[k])*(t[k]-o[k]); 
		}

		for(int k=0;k<nh+1;k++)
		{
			sum=0;

			for(int i=0;i<no;i++)
				sum+= Woh[i][k]*delK[i];

			delH[k]=h[k]*(1-h[k])*sum;
		}
	}// function to calculte error

	void updateWeights()
	{
	
		float delOH=0f,delHX=0f;

		for(int i=0;i<no;i++)
		{
			for(int j=0;j<nh+1;j++)
			{
				delOH=learningRate*delK[i]*h[j] - momentum*w1[i][j];
				Woh[i][j]=Woh[i][j]+delOH;
				w1[i][j]=delOH;// storing the value of change in weights for next iteration
			}
		}

		for(int i=1;i<nh+1;i++)
		{
			for(int j=0;j<ni+1;j++)
			{
				delHX=learningRate*delH[i]*x[j]-momentum*w2[i][j];
				Whx[i][j]=Whx[i][j]+delHX;
				w2[i][j]=delHX;// storing the value of change in weights for next iteration
			}

		}
	}// function to update weights according to error in output

	float calculateOverallError()
	{
		float overallError=0.0f;

		for(int i=0;i<no;i++)
		overallError=0.5f*(float)(Math.pow((t[i]-o[i]),2));


		return overallError;
	}//calculates RMS error for each input file

	void backPropogate()
	{
		feedForward();
		calculateError();
		updateWeights();

	}//method to calculte the error and backpropogate it to update weights
	
	public static void main(String [] args)throws IOException //main function
	{
		BufferedReader k=new BufferedReader(new InputStreamReader(System.in));
		
		Sunglasses obj=new Sunglasses();
		
		//Training the network starts 
		
		obj.randomWeights();
		float er=100f;//to store the RMS error for one epoch
		int iter=1;// to store number of epocs performed
		while(er>obj.minError && iter<obj.maxIter)
		{
			er=0;
			FileReader fr = new FileReader("list/straightrnd_train.list");//opening the training set
			BufferedReader br=new BufferedReader(fr);
			String s;
			
		
			while((s=br.readLine())!=null)//reading each instance of data set
			{
				obj.x[0]=1;// threshold
				
				FileInputStream fileInputStream = new FileInputStream(s.substring(1,s.length()));
			   	DataInputStream dis = new DataInputStream(fileInputStream);//reading the image file
			    
			  	for (int c=1;c<961; c++ ) 
					obj.x[c] = dis.readUnsignedByte()/255.0f;//storing the pixel grayscale values
	
				obj.getTargetValue(s);//to obtain the target output value for the data instance
				obj.backPropogate(); //calls the method which runs the backpropogation algorith
				er=er+obj.calculateOverallError();// total RMS error is updated
			}
		
			System.out.println(iter+" "+er);//Uncomment this line to veiw progress of Gradient Descent
			
			iter++;
		}
		
		//Training Network ends
		
		System.out.println("Classification\t\tFile Name");
		
		// -------TESTING begins here
		FileReader fr=new FileReader("list/straightrnd_test2.list");//opening the testing set file
		BufferedReader br=new BufferedReader(fr);
		String s;
		float acc=0.0f;// stores the accuracy
		int size=0;// stores size of testing set to calculate accuracy
		
		while((s=br.readLine())!=null)//reads each input image of testing set
		{
			obj.x[0]=1;//threshold
			String file_name=obj.getFileName(s);//stores name of the image
			
			FileInputStream fileInputStream = new FileInputStream(s.substring(1,s.length()));//reading the image file
		   	DataInputStream dis = new DataInputStream(fileInputStream);
		    
		  	for (int c=1;c<961; c++ ) 
				obj.x[c] = dis.readUnsignedByte()/255.0f;
			
			obj.feedForward();// feed forward to get the generated output
			obj.getTargetValue(s);// to get target output
			
			
			if(obj.o[0]>0.5)//classifies as wearing or not wearing sunglasses
				System.out.print("Sunglasses\t\t");
			else 
				System.out.print("No Sunglasses\t\t");
			
			System.out.println(file_name);// outputs the file name which has been classified
			
			if((obj.o[0]>0.5 && obj.t[0]==1) || (obj.o[0]<0.5 && obj.t[0]==0))
			acc++;
			
			size++;
		}
		
		System.out.println("Accuracy="+((acc/size)*100)+"%");//Prints accuracy
	}//end of main
	
	
}
