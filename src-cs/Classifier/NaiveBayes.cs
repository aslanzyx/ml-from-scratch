using System;
using Main;

namespace Main.Classifier
{
    class NaiveBayes
    {
        private double[,] prob;
        private double[] cond;

        public void fit(Boolean[,] x, int[] y)
        {
            // find and catalgory labels
            int n = x.GetLength(0);
            int d = x.GetLength(1);
            int[] count = new int[n];
            int[,] num = new int[n, d, n];

            // catalgory labels
            for (int i = 0; i < n; i++)
            {
                count[y[i]]++;
                for (int j = 0; j < d; j++)
                {
                    if (x[i, j])
                    {
                        num[y[i], j]++;
                    }
                }
            }
            cond = new double[n];
            prob = new double[n, d];
            int c;
            for (int i = 0; i < n; i++)
            {
                if (count[i] != 0)
                {
                    cond[i] = (double)count[i] / n;
                    for (int j = 0; j < d; j++)
                    {
                        prob[i, j] = num[i, j] / count[i];
                    }
                }
            }
        }

        public int[] Predict(double[,] x)
        {
            // TODO
        }
    }
}
