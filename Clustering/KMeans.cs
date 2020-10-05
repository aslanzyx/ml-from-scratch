using System;
using Main;

namespace Main.Clustering
{
    class KMeans
    {
        private int k;
        private double[,] means;

        public KMeans(int k, double[,] means)
        {
            this.k = k;
            this.means = means;
        }

        public void fit(double[,] x)
        {
            int n = x.GetLength(0);
            int d = x.GetLength(1);
            int k = means.GetLength(0);
            // make prediction
            int[] yPred = Predict(x);
            double[,] sums = new double[k, d];
            int[] size = new int[c];
            // calculate the means
            for (int i = 0; i < n; i++)
            {
                int label = yPred[i];
                for (int j = 0; j < d; j++)
                {
                    sums[label, j] += x[i, j];
                    size[label]++;
                }
            }
            // update the mean
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < d; j++)
                {
                    means[c, j] = sums[c, j] / size[c];
                }
            }
        }

        public int[] Predict(double[,] x)
        {
            int n = x.GetLength(0);
            double retval = new int[n];
            // For each example
            for (int i = 0; i < n; i++)
            {
                // Compute the distance for each mean point
                // And get the index for the shortest distance
                double idx = 0;
                double minDist = Util.distance(x[i], means[0]);
                for (int c = 1; c < k; c++)
                {
                    double dist = Util.distance(x[i], means[c]);
                    minDist = dist < minDist ? dist : minDist;
                    idx = dist < minDist ? c : idx;
                }
                retval[i] = idx;
            }
            return retval;
        }
    }
}
