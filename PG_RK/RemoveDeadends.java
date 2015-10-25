package pagerank;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class RemoveDeadends {

	enum myCounters{ 
		NUMNODES;
	}
	
	static class Map extends Mapper<LongWritable, Text, Text, Text> {
		
		protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
			{
				String[] data = value.toString().split("[ \t]");
				Text pred = new Text(data[0]);
				Text succ = new Text(data[1]);
				context.write(pred, new Text("S "+ succ));
				context.write(succ, new Text("P "+ pred));
			}
		}
	

	static class Reduce extends Reducer<Text, Text, Text, Text> {
		
		protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			/*
			 * Count number of successors
			 */
			List<String> succs = new ArrayList<String>();
			List<String> preds = new ArrayList<String>();
			for (Text value: values) {
				String [] data = value.toString().split("[ \t]");
				if ("P".equals(data[0])) {
					preds.add(data[1]);
				}
				else {
					succs.add(data[1]);
				}
			}
			if (succs.size() > 0) {
				for (String pred:preds) {
					context.write(new Text(pred), key);
				}
				context.getCounter(myCounters.NUMNODES).increment(1);
			}
			
			
		}

}

	public static void job(Configuration conf) throws IOException, ClassNotFoundException, InterruptedException{
		
		
		boolean existDeadends = true;
		
		/* You don't need to use or create other folders besides the two listed below.
		 * In the beginning, the initial graph is copied in the processedGraph. After this, the working directories are processedGraphPath and intermediaryResultPath.
		 * The final output should be in processedGraphPath. 
		 */
		
		FileUtils.copyDirectory(new File(conf.get("graphPath")), new File(conf.get("processedGraphPath")));
		String intermediaryDir = conf.get("intermediaryResultPath");
		String currentInput = conf.get("processedGraphPath");
		
		long nNodes = conf.getLong("numNodes", 0);
		conf.setLong("numNodes", 0);
		
		while(existDeadends)
		{
			
			Job job = Job.getInstance(conf);
			job.setJobName("deadends job");
			/* TO DO : configure job and move in the best manner the output for each iteration
			 * you have to update the number of nodes in the graph after each iteration,
			 * use conf.setLong("numNodes", nNodes);
			*/
			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Text.class);

			job.setMapperClass(Map.class);
			job.setReducerClass(Reduce.class);

			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);

			FileInputFormat.setInputPaths(job, new Path(currentInput));
			FileOutputFormat.setOutputPath(job, new Path(intermediaryDir));
			
			job.waitForCompletion(true);

			Counter counter = job.getCounters().findCounter(myCounters.NUMNODES);
			existDeadends = (counter.getValue()!=nNodes);
			nNodes = counter.getValue();

			//Delete currentInput
			FileUtils.deleteDirectory(new File(currentInput));
			//Copy currentInput in currentInput
			FileUtils.copyDirectory(new File(intermediaryDir), new File(currentInput));
			FileUtils.deleteDirectory(new File(intermediaryDir));
		}
		conf.setLong("numNodes", nNodes);
		FileUtils.copyDirectory(new File(currentInput), new File(conf.get("graphNoDeadendsPath")));
	}
	
}