package model;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 监督学习的一阶HMM
 * @author outsider
 */
public class SupervisedHMMOneFile {
	private double[] pi;
	private double[][] A;
	private double[][] B;
	//定义一个无穷小
	public double infinity = (double) -Math.pow(2, 31);
	public SupervisedHMMOneFile() {
		pi = new double[4];
		A = new double[4][4];
		B = new double[4][65536];
	}
	
	public void train(SequenceData data) {
		for(int i = 0; i < data.xSeqs.size(); i++) {
			int[] xSeq = data.xSeqs.get(i);
			int[] ySeq = data.ySeqs.get(i);
			for(int j = 0; j < xSeq.length -1; j++) {
				pi[ySeq[j]]++;
				A[ySeq[j]][ySeq[j+1]]++;
				B[ySeq[j]][xSeq[j]]++;
			}
			//最后一个没统计到的节点
			pi[ySeq[ySeq.length -1]]++;
			B[ySeq[ySeq.length-1]][xSeq[xSeq.length-1]]++;
		}
		//概率取对数。
		logProba(data.allSeqLen);
	}
	//概率取对数
	public void logProba(int seqLen) {
		double a = Math.log(seqLen);
		for(int i = 0; i < pi.length; i++) {
			pi[i] = Math.log(pi[i]) - a;
		}
		for(int i = 0; i < A.length; i++) {
			double sum = Math.log(sum(A[i]));
			for(int j = 0; j < A[0].length; j++) {
				if(A[i][j] == 0)
					A[i][j] = infinity;
				else 
					A[i][j] = Math.log(A[i][j]) - sum;
			}
		}
		for(int i = 0; i < B.length; i++) {
			double sum = Math.log(sum(B[i]));
			for(int j = 0; j < B[0].length; j++) {
				if(B[i][j] == 0)
					B[i][j] = infinity;
				else 
					B[i][j] = Math.log(B[i][j]) - sum;
			}
		}
	}
	
	public int[] predict(String sen) {
		char[] chs = sen.toCharArray();
		return viterbi(chs);
	}
	public String[] predictAndSplit(String sen) {
		char[] chs = sen.toCharArray();
		int[] tags = viterbi(chs);
		return decode(tags, sen);
	}
	
	/**
	 * 解码为分词结果
	 * 0 1 2 3
	 * B M E S
	 * @param predict
	 * @param sentence
	 * @return
	 */
	public String[] decode(int[] predict, String sentence) {
		List<String> res = new ArrayList<>();
		char[] chars = sentence.toCharArray();
		for(int i = 0; i < predict.length;i++) {
			if(predict[i] == 0 || predict[i] == 1) {
				int a = i;
				while(predict[i] != 2) {
					i++;
					if(i == predict.length) {
						break;
					}
				}
				int b = i;
				if(b == predict.length) {
					b--;
				}
				res.add(new String(chars,a,b-a+1));
			} else {
				res.add(new String(chars,i,1));
			}
		}
		String[] s = new String[res.size()];
		return res.toArray(s);
	}
	/**
	 * 维特比算法
	 * @param x
	 * @return
	 */
	public int[] viterbi(char[] x) {
		double[][] delta = new double[x.length][4];
		int[][] track = new int[x.length][4];
		//delta初始值
		for(int i = 0; i < 4; i++) {
			delta[0][i] = pi[i] + B[i][x[0]];
		}
		for(int t = 1; t < x.length; t++) {
			for(int i = 0; i < 4; i++) {
				delta[t][i] = B[i][x[t]];
				double max = -Double.MAX_VALUE;
				for(int j = 0; j < 4;j++) {
					double tmp = delta[t-1][j] + A[j][i];
					if(tmp > max) {
						max = tmp;
						track[t][i] = j;
					}
				}
				delta[t][i] += max;
			}
		}
		int T = x.length-1;
		//回溯找到最优路径
		int[] tags = new int[x.length];
		double p = delta[T][0];
		for(int i = 1; i < 4; i++) {
			if(delta[T][i] > p) {
				p = delta[T][i];
				tags[T] = i;
			}
		}
		for(int i = T-1; i >=0; i--) {
			tags[i] = track[i+1][tags[i+1]];
		}
		return tags;
	}
	
	//求和
	public double sum(double[] arr) {
		double sum = 0;
		for(double a : arr) {
			sum += a;
		}
		return sum;
	}
	
	public static void main(String[] args) {
		SequenceData data = loadPKUSegData();
		SupervisedHMMOneFile hmm = new SupervisedHMMOneFile();
		hmm.train(data);
		List<String> testLines = readLines("data/novel.txt", "utf-8");
		for(String line : testLines) {
			String[] words = hmm.predictAndSplit(line);
			System.out.println(String.join("/", words));
		}
	}
	public static class SequenceData{
		public List<int[]> xSeqs;
		public List<int[]> ySeqs;
		public int allSeqLen;
	}
	public static SequenceData loadPKUSegData() {
		List<String> lines =readLines("data/pku_training_crf.utf8", "utf-8");
		List<int[]> xSeqs = new ArrayList<int[]>(lines.size());
		List<int[]> ySeqs = new ArrayList<int[]>(lines.size());
		Map<String, Integer> tag2Int = new HashMap<String, Integer>();
		tag2Int.put("B", 0);
		tag2Int.put("M", 1);
		tag2Int.put("E", 2);
		tag2Int.put("S", 3);
		int[] x = new int[lines.size()];
		int[] y = new int[lines.size()];
		int c = 0;
		for(String line : lines) {
			String[] wn = line.split("\t");
			x[c] = wn[0].toCharArray()[0];
			y[c] = tag2Int.get(wn[1]);
			c++;
		}
		xSeqs.add(x);
		ySeqs.add(y);
		SequenceData sequenceData = new SequenceData();
		sequenceData.xSeqs = xSeqs;
		sequenceData.ySeqs = ySeqs;
		sequenceData.allSeqLen = xSeqs.size();
		return sequenceData;
	}
	public static List<String> readLines(String path, String encoding){
		if(encoding == null)
			encoding = "gbk";
		List<String> res = new ArrayList<String>();
		try {
			InputStreamReader in = new InputStreamReader(new FileInputStream(path), encoding);
			BufferedReader reader = new BufferedReader(in);
			
			String line = null;
			while((line = reader.readLine()) != null) {
				res.add(line);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return res;
	}
}
