using System;
using Main;

namespace Main.Classifier
{
    class KNN
    {
        private int k;
        private double[,] x;
        private int[] y;
        public KNN(int k)
        {
            this.k = k;
        }

        public void fit(double[,] x, int[] y)
        {
            this.x = x;
            this.y = y;
        }

        public int[] Predict(double[,] x)
        {
            int t = this.x.GetLength(0);
            int n = x.GetLength(0);
            int d = x.GetLength(1);
            int[] retval = new int[n];
            // for each data example
            for (int i = 0; i < n; i++)
            {
                // for each point in the model
                double distances = new double[t];
                for (int c = 0; c < t; c++)
                {
                    distances[c] = Util.Distance(x[i], this.x[c]);
                }
                int[] nearest = Util.MinArg(distances, k);
                retval[i] = Util.Mode(nearest);
            }
            return retval;
        }
    }
}
