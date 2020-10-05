using System;

namespace Data
{
    class Dataset
    {
        private double[,] X;
        private int[] y;
        private string[] features;
        private string[] labels;

        public Dataset(double[,] X, double[] y = null, string[] features = null, string[] labels) 
        {
            this.X = X;
            this.y = y;
            this.features = features;
            this.labels = labels;
        }

        public FeatureNum()
        {
            return X.GetLength(1);
        }

        public double this[int i, int j]
        {
            get
            {
                X[i, j];
            }
            set
            {
                X[i, j] = value;
            }
        }

        public bool Supervisable()
        {
            return y != null;
        }

        public string GetFeatureName(int i)
        {
            return features[i];
        }

        public int GetFeatureSize()
        {
            return features.Length;
        }

        public double[] GetExample(int i)
        {
            return dataset[i];
        }

        public double[] GetFeature(int i)
        {
            double[] retval = new double[dataset.GetLength(0)];
            for (int j = 0; j < dataset.GetLength(0); j++)
            {
                retval[j] = dataset[j, i];
            }
            return retval;
        }

        public void AddData()
        {

        }

        public Dataset[] SplitDataset(int fold)
        {
            Dataset[] retval = new Dataset[fold];
            int size = X.GetLength(0) / fold;

            for (int i = 0; i < fold; i++)
            {
                double[,] X = new double[size, features.Length];
                for (int j = 0; j < (i + 1) * fold; j++)
                {
                    X[]
                }
                retval = new Dataset()

            }
        }
    }

    public class DimensionNotMatchException : Exception
    {
        public DimensionNotMatchException(string message)
        {
            base(message);
        }
    }

    enum SplittingMethod
    {
        StripOff
    }
}