using Accord.Imaging;
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
using System.IO;
using System.Drawing;
using Accord;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.MachineLearning;

namespace Lecture_13_Image
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

        //http://accord-framework.net/docs/html/T_Accord_Imaging_BagOfVisualWords.htm

        private void btnLearn_Click(object sender, RoutedEventArgs e)
        {
            Accord.Math.Random.Generator.Seed = 0;


            var bow = BagOfVisualWords.Create(new BinarySplit(6));

            // Since we are using generics, we can setup properties 
            // of the binary split clustering algorithm directly:
            bow.Clustering.ComputeProportions = true;
            bow.Clustering.ComputeCovariances = true;

            List<string> lstfiles = Directory.GetFiles("numbers").ToList();

            lstfiles.Sort();

            Bitmap[] images = new Bitmap[lstfiles.Count];
            int irIndex = 0;
            foreach (var item in lstfiles)
            {
                Bitmap tempbm = new Bitmap(item);
                images[irIndex] = tempbm;
                irIndex++;
            }

            bow.Learn(images);

            double[][] features = bow.Transform(images);          

            int[] labels = new int[images.Length];
            irIndex = 0;
            foreach (var item in lstfiles)
            {
                labels[irIndex] = Convert.ToInt32( item.Split('_').Last().Replace(".png",""))-1;
                irIndex++;
            }

            var teacher = new MulticlassSupportVectorLearning<Linear>()
            {
                // using LIBLINEAR's L2-loss SVC dual for each SVM
                Learner = (p) => new LinearDualCoordinateDescent()
                {
                    Loss = Loss.L2
                }
            };

            teacher.ParallelOptions.MaxDegreeOfParallelism = 1; // (Remove, comment, or change this line to enable full parallelism)

            // Learn a machine
            var machine = teacher.Learn(features, labels);

            // Obtain class predictions for each sample
            int[] predicted = machine.Decide(features);

            // Compute classification error
            double error = new ZeroOneLoss(labels).Loss(predicted);

            //test unforseen data

            lstfiles = Directory.GetFiles("test").ToList();

            lstfiles.Sort();

            images = new Bitmap[lstfiles.Count];
            irIndex = 0;
            foreach (var item in lstfiles)
            {
                Bitmap tempbm = new Bitmap(item);
                images[irIndex] = tempbm;
                irIndex++;
            }

            var bow33 = BagOfVisualWords.Create(new BinarySplit(6));

            // Since we are using generics, we can setup properties 
            // of the binary split clustering algorithm directly:
            bow.Clustering.ComputeProportions = true;
            bow.Clustering.ComputeCovariances = true;

            // Compute the model
            bow33.Learn(images);

            double[][] features_test = bow33.Transform(images);

            labels = new int[images.Length];
            irIndex = 0;
            foreach (var item in lstfiles)
            {
                labels[irIndex] = Convert.ToInt32(item.Split('_').Last().Replace(".png", "")) - 1;
                irIndex++;
            }

            // Obtain class predictions for each sample
            predicted = machine.Decide(features);

            // Compute classification error
            error = new ZeroOneLoss(labels).Loss(predicted);
        }
    }
}
