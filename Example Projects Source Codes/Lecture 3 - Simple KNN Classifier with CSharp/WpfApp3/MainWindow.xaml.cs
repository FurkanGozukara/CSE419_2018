using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WpfApp3
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {

            double[] StudentA = new double[] { 9, 32, 97 };   //Final Grade A
            double[] StudentB = new double[] { 12, 65, 86.1 };  //Final Grade B
            double[] StudentC = new double[] { 19, 54, 45.1 };  //Final Grade C
            double[] StudentA2 = new double[] { 11, 34, 98 };   //Final Grade A
            double[] StudentB2 = new double[] { 13, 63, 87.1 };  //Final Grade B
            double[] StudentC2 = new double[] { 13, 46, 55.1 };  //Final Grade C
            double[] StudentC3 = new double[] { 11, 36, 55.1 };  //Final Grade C                                                 //------------------------

            List<double[]> TrainingSet = new List<double[]>();
            TrainingSet.Add(StudentA);
            TrainingSet.Add(StudentA2);
            TrainingSet.Add(StudentB);
            TrainingSet.Add(StudentB2);
            TrainingSet.Add(StudentC);
            TrainingSet.Add(StudentC2);
            TrainingSet.Add(StudentC3);

            //------------------------

            //------new Student-------
            List<double> newPlayer = new List<double>();
            newPlayer.Add(Convert.ToDouble(txt1.Text));
            newPlayer.Add(Convert.ToDouble(txt2.Text));
            newPlayer.Add(Convert.ToDouble(txt3.Text));

            Dictionary<string, List<double>> dicClassDistances = new Dictionary<string, List<double>>();

            double[] newSudent = (double[])newPlayer.ToArray();
            //------------------------

            //compare to all students
            double distance = 0.0;

            double dblMinDist = double.MaxValue;
            string srWhichClassBelongs = "";

            for (int i = 0; i < TrainingSet.Count; i++)
            {

                string whichClass = "Grade A";

                switch (i)
                {
                    case 2:
                    case 3:
                        whichClass = "Grade B";
                        break;
                    case 4:
                    case 5:
                    case 6:
                        whichClass = "Grade C";
                        break;
                }

          

                double dblAvgDist = 0;

                //Test the Euclidean Distance calculation between two data points
                distance = EuclideanDistance(newSudent, TrainingSet[i]);
                lstBox1.Items.Add("Euclidean Distance New Student : " + distance + " to "+ whichClass);
                dblAvgDist += distance;

                distance = ChebyshevDistance(newSudent, TrainingSet[i]);
                lstBox1.Items.Add("Chebyshev Distance New Student : " + distance + " to " + whichClass);
                dblAvgDist += distance;

                distance = ManhattanDistance(newSudent, TrainingSet[i]);
                lstBox1.Items.Add("Manhattan Distance New Student : " + distance + " to " + whichClass);

                dblAvgDist += distance;

                dblAvgDist = dblAvgDist / 3;

                distance = ManhattanDistance(newSudent, TrainingSet[i]);
                lstBox1.Items.Add("Avg distance : " + dblAvgDist + " to " + whichClass);

                //if (dblAvgDist< dblMinDist)
                //{
                //    srWhichClassBelongs = whichClass;
                //    dblMinDist = dblAvgDist;
                //}

                if (dicClassDistances.ContainsKey(whichClass))
                {
                    dicClassDistances[whichClass].Add(dblAvgDist);
                }
                else
                {
                    dicClassDistances.Add(whichClass, new List<double> { dblAvgDist });
                }
            }

            dblMinDist = double.MaxValue;
            srWhichClassBelongs = "";
            int k = 2;
            foreach (var item in dicClassDistances)
            {
                double dblLoopAvg = item.Value.OrderBy(pr => pr).ToList().GetRange(0, k).Sum(pr => pr)/k;

                List<double> tempList = item.Value.OrderBy(pr => pr).ToList();
                tempList = tempList.GetRange(0, k);
                double dblTempAvg = tempList.Average();

                if(dblMinDist> dblTempAvg)
                {
                    srWhichClassBelongs = item.Key;
                    dblMinDist = dblTempAvg;
                }
            }

            lstBox1.Items.Add("new student belongs to class "+ srWhichClassBelongs+" with lowest avg distance: "+ dblMinDist);

        }

        public static double EuclideanDistance(double[] X, double[] Y)
        {
            int count = 0;

            double distance = 0.0;

            double sum = 0.0;


            if (X.GetUpperBound(0) != Y.GetUpperBound(0))
            {
                throw new System.ArgumentException("the number of elements in X must match the number of elements in Y");
            }
            else
            {
                count = X.Length;
            }

            for (int i = 0; i < count; i++)
            {
                sum = sum + Math.Pow(Math.Abs(X[i] - Y[i]), 2);
            }

            distance = Math.Sqrt(sum);

            return distance;
        }

        /// <summary>
        /// Calculates the Minkowski Distance Measure between two data points
        /// </summary>
        /// <param name="X">An array with the values of an object or datapoint</param>
        /// <param name="Y">An array with the values of an object or datapoint</param>
        /// <returns>Returns the Minkowski Distance Measure Between Points X and Points Y</returns>
        public static double ChebyshevDistance(double[] X, double[] Y)
        {
            int count = 0;

            if (X.GetUpperBound(0) != Y.GetUpperBound(0))
            {
                throw new System.ArgumentException("the number of elements in X must match the number of elements in Y");
            }
            else
            {
                count = X.Length;
            }
            double[] sum = new double[count];

            for (int i = 0; i < count; i++)
            {
                sum[i] = Math.Abs(X[i] - Y[i]);
            }
            double max = double.MinValue;
            foreach (double num in sum)
            {
                if (num > max)
                {
                    max = num;
                }
            }
            return max;
        }
        /// <summary>
        /// Calculates the Manhattan Distance Measure between two data points
        /// </summary>
        /// <param name="X">An array with the values of an object or datapoint</param>
        /// <param name="Y">An array with the values of an object or datapoint</param>
        /// <returns>Returns the Manhattan Distance Measure Between Points X and Points Y</returns>
        public static double ManhattanDistance(double[] X, double[] Y)
        {
            int count = 0;

            double distance = 0.0;

            double sum = 0.0;


            if (X.GetUpperBound(0) != Y.GetUpperBound(0))
            {
                throw new System.ArgumentException("the number of elements in X must match the number of elements in Y");
            }
            else
            {
                count = X.Length;
            }

            for (int i = 0; i < count; i++)
            {
                sum = sum + Math.Abs(X[i] - Y[i]);
            }

            distance = sum;

            return distance;
        }
    }
}
