package pagerank;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;

/*
 * VERY IMPORTANT 
 * 
 * Each time you need to read/write a file, retrieve the directory path with conf.get 
 * The paths will change during the release tests, so be very carefully, never write the actual path "data/..." 
 * CORRECT:
 * String initialVector = conf.get("initialRankVectorPath");
 * BufferedWriter output = new BufferedWriter(new FileWriter(initialVector + "/vector.txt"));
 * 
 * WRONG
 * BufferedWriter output = new BufferedWriter(new FileWriter(data/initialVector/vector.txt"));
 */

public class PageRank {

	/**
	 * Initialize vector for pageRank
	 * @param directoryPath
	 * @param n
	 * @throws IOException
	 */
	public static void createInitialRankVector(String directoryPath, long n) throws IOException 
	{
		File dir = new File(directoryPath);
		if(! dir.exists())
			FileUtils.forceMkdir(dir);
		String tgtFileName = dir+"/part-r-00000";
		BufferedWriter bw = new BufferedWriter(new FileWriter(tgtFileName));
		for (int i=0;i<n;i++) {
			bw.write((i+1)+" "+(1.0/((double)n))+"\n");
		}
		bw.close();
	}
	
	public static boolean checkConvergence(String initialDirPath, String iterationDirPath, double epsilon) throws IOException
	{
		List<Double> initialVector = new ArrayList<Double>();
		List<Double> iterationVector = new ArrayList<Double>();
		
		//Read initialDir
		InputStream ips=new FileInputStream(initialDirPath+"/part-r-00000"); 
		InputStreamReader ipsr = new InputStreamReader(ips);
		BufferedReader br = new BufferedReader(ipsr);
		String ligne;
		while ((ligne=br.readLine())!=null){
			String[] splits  = ligne.toString().split("\\s+");
			//Stock value in the list
			initialVector.add(Double.parseDouble(splits[1]));
		}
		br.close(); 
		//Read iterationDir
		ips=new FileInputStream(iterationDirPath+"/part-r-00000"); 
		ipsr = new InputStreamReader(ips);
		br = new BufferedReader(ipsr);
		while ((ligne=br.readLine())!=null){
			String[] splits  = ligne.toString().split("\\s+");
			//Stock value in the list
			iterationVector.add(Double.parseDouble(splits[1]));
		}
		br.close(); 
		
		Double sum = 0.0;
		
		for (int i = 0 ; i < initialVector.size(); i++) {
			Double abs = Math.abs(initialVector.get(i) - iterationVector.get(i));
			sum += abs;
		}
		
		if (sum < epsilon) {
			return true;
		} else {
			return false;
		}
	}
	
	public static void avoidSpiderTraps(String vectorDirPath, long nNodes, double beta) 
	{
		String vFile = vectorDirPath +"/part-r-00000";
		String destFileName = vectorDirPath +"/part-r-00001";
		try {
			BufferedReader br = new BufferedReader(new FileReader(vFile));
			BufferedWriter bw = new BufferedWriter(new FileWriter(destFileName));
			String line = br.readLine();
			while (line != null) {
				String[] data = line.split("\\s+");
				double val = Double.parseDouble(data[1]);
				val *= beta;
				val += (1.0-beta)/((double)nNodes);
				bw.write(data[0]+" "+val);
				bw.newLine();
				line = br.readLine();
			}
			br.close();
			bw.close();
			FileUtils.forceDelete(new File(vFile));
			File f = new File(destFileName);
			f.renameTo(new File(vFile));
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * PageRank job
	 * @param conf
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ClassNotFoundException
	 */
	public static void iterativePageRank(Configuration conf) 
			throws IOException, InterruptedException, ClassNotFoundException
	{
		Job job = Job.getInstance(conf);
		job.setJobName("pagerank job");
		String initialVector = conf.get("initialVectorPath");
		String currentVector = conf.get("currentVectorPath");
		String processedGraph = conf.get("processedGraphPath");
		String stochasticMatrix = conf.get("stochasticMatrixPath");
		
		FileUtils.deleteQuietly(new File(initialVector));
		FileUtils.deleteQuietly(new File(processedGraph));
		FileUtils.deleteQuietly(new File(stochasticMatrix));

		String finalVector = conf.get("finalVectorPath"); 
		/*here the testing system will search for the final rank vector*/
		
		Double epsilon = conf.getDouble("epsilon", 0.1);
		Double beta = conf.getDouble("beta", 0.8);

		// Remove dead ends
		RemoveDeadends.job(conf);
		// Build Stochastic matrix
		GraphToMatrix.job(conf);
		// Initialize vector
		createInitialRankVector(initialVector, conf.getLong("numNodes",0));
		
		boolean convergence = false;
		
		while (!convergence) {
			MatrixVectorMult.job(conf);
			avoidSpiderTraps(currentVector, conf.getLong("numNodes",0), beta);
			convergence = checkConvergence(initialVector, currentVector, epsilon);
			FileUtils.deleteQuietly(new File(initialVector));
            FileUtils.moveFile(new File(currentVector+"/part-r-00000"), new File(initialVector+"/part-r-00000"));
			FileUtils.deleteQuietly(new File(currentVector));
		}
 
		FileUtils.copyDirectory(new File(initialVector), new File(finalVector));
	}
}
