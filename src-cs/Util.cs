using System;

namespace Main
{
    static class Util
    {
        public static double Distance(double[] r, double[] s)
        {
            double[] shift = new double[r.Length];
            for (int i = 0; i < s.Length; i++)
            {
                shift[i] = Math.Abs(r[i] - s[i]);
            }
            return Norm(shift, NormType.L2);
        }
        public static double Norm(double[] r, NormType type)
        {
            switch (type)
            {
                case NormType.L1:
                    return norm1(r);
                case NormType.L2:
                    return norm2(r);
                case NormType.LINF:
                    return normINF(r);
                default:
                    return norm2(r);
            }
        }
        private static double norm1(double[] r)
        {
            double retval = 0;
            for (int i = 0; i < r.Length; i++)
            {
                retval += r[i];
            }
            return retval;
        }

        private static double norm2(double[] r)
        {
            double retval = 0;
            for (int i = 0; i < r.Length; i++)
            {
                retval += Math.Pow(r[i], 2);
            }
            return retval;
        }

        private static double normINF(double[] r)
        {
            double retval = 0;
            foreach (double i in r)
            {
                retval = i > retval ? i : retval;
            }
            return retval;
        }

        public int[] MinArg(double[] arr, int k)
        {
            // Stub
            return null;
        }

        public int Mode(int[] arr)
        {
            // Stub
            return -1;
        }
    }

    enum NormType
    {
        L1, L2, LINF
    }
}