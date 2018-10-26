using Accord.Math.Optimization.Losses;
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
using Accord.Controls;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics;
using Accord.Statistics.Kernels;
using System.IO;

namespace Lecture_6
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

        private void btnXor_Click(object sender, RoutedEventArgs e)
        {
            double[][] inputs =
            {
                /* 1.*/ new double[] { 0, 0 },
                /* 2.*/ new double[] { 1, 0 }, 
                /* 3.*/ new double[] { 0, 1 }, 
                /* 4.*/ new double[] { 1, 1 },
            };

            int[] outputs =
            { 
                /* 1. 0 xor 0 = 0: */ 0,
                /* 2. 1 xor 0 = 1: */ 1,
                /* 3. 0 xor 1 = 1: */ 1,
                /* 4. 1 xor 1 = 0: */ 0,
            };

            // Create the learning algorithm with the chosen kernel
            var smo = new SequentialMinimalOptimization<Gaussian>()
            {
                Complexity = 100 // Create a hard-margin SVM 
            };

            // Use the algorithm to learn the svm
            var svm = smo.Learn(inputs, outputs);

            // Compute the machine's answers for the given inputs
            bool[] prediction = svm.Decide(inputs);

            // Compute the classification error between the expected 
            // values and the values actually predicted by the machine:
            double error = new AccuracyLoss(outputs).Loss(prediction);

            Console.WriteLine("Error: " + error);

            // Show results on screen 
            ScatterplotBox.Show("Training data", inputs, outputs);
            ScatterplotBox.Show("SVM results", inputs, prediction.ToZeroOne());
        }

        class hersinifbilgi
        {
            public int irClass = 0;
            public int irCount = 1;
        }

        class perInput
        {
            public double[] features;
            public int irClass = 0;
        }

        private void btniris_Click(object sender, RoutedEventArgs e)
        {
            List<perInput> lstInputs = new List<perInput>();

            int irListSize = 150;
            double[][] inputs = new double[irListSize][];

            int irCounter = 0;
            foreach (var item in File.ReadLines("iris.txt"))
            {
                inputs[irCounter] = new double[4];
                inputs[irCounter][0] = double.Parse(item.Split(',')[0]);
                inputs[irCounter][1] = double.Parse(item.Split(',')[1]);
                inputs[irCounter][2] = double.Parse(item.Split(',')[2]);
                inputs[irCounter][3] = double.Parse(item.Split(',')[3]);
                perInput tempinput = new perInput();
                tempinput.features = inputs[irCounter];
                lstInputs.Add(tempinput);
                irCounter++;
            }

            Dictionary<string, hersinifbilgi> dicClasses =
                new Dictionary<string, hersinifbilgi>();

            int irFirstClass = 0;
            irCounter = 0;
            foreach (var item in File.ReadLines("iris.txt"))
            {
                hersinifbilgi temp = new hersinifbilgi();
                temp.irClass = irFirstClass;
            
                if (!dicClasses.ContainsKey(item.Split(',').Last()))
                {
                    dicClasses.Add(item.Split(',').Last(), temp);
                    lstInputs[irCounter].irClass = irFirstClass;
                    irFirstClass++;
                }
                else
                {
                    lstInputs[irCounter].irClass = irFirstClass-1;
                    dicClasses[item.Split(',').Last()].irCount++;
                }
                irCounter++;
            }


            int irTrainPercent = 90;

            int trainingSetSize = irListSize * irTrainPercent / 100;

            List<perInput> lstInputTraining = new List<perInput>();
            List<perInput> lstInputTest = new List<perInput>();

            foreach (var item in dicClasses)
            {
                var selection = lstInputs.Where(pr => pr.irClass == item.Value.irClass).Select(pr => pr).ToList();

                var inputSelect = selection.GetRange(0, selection.Count * irTrainPercent / 100);

                var outputSelection = selection.Where(pr => !inputSelect.Contains(pr)).ToList();

                lstInputTraining.AddRange(inputSelect);
                lstInputTest.AddRange(outputSelection);
            }

            double[][] trainingSet = new double[trainingSetSize][];
            int[] trainingOutput = new int[trainingSetSize];

            irCounter = 0;
            foreach (var item in lstInputTraining)
            {
                trainingSet[irCounter] = item.features;
                trainingOutput[irCounter] = item.irClass;
                irCounter++;
            }

            int testSetSize = irListSize - trainingSetSize;
            double[][] TestSet = new double[testSetSize][];
            int[] testOutput = new int[testSetSize];

            irCounter = 0;
            foreach (var item in lstInputTest)
            {
                TestSet[irCounter] = item.features;
                testOutput[irCounter] = item.irClass;
                irCounter++;
            }

            // Create a one-vs-one multi-class SVM learning algorithm 
            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearNewtonMethod()
                {
                   
                }
            };

            // The following line is only needed to ensure reproducible results. Please remove it to enable full parallelization
            teacher.ParallelOptions.MaxDegreeOfParallelism = 4; // (Remove, comment, or change this line to enable full parallelism)

            // Learn a machine
            var machine = teacher.Learn(trainingSet, trainingOutput);

            // Obtain class predictions for each sample
            int[] predicted = machine.Decide(trainingSet);

            double error = new ZeroOneLoss(trainingOutput).Loss(predicted);


            int[] predictedTest = machine.Decide(TestSet);

            double errorTest = new ZeroOneLoss(testOutput).Loss(predictedTest);

        }
    }
}
