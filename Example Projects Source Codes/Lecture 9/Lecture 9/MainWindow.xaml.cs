using Accord.MachineLearning;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
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

namespace Lecture_8
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
            string[] texts =
    {
    @"The concept of grouping students together in a centralized location for learning has existed since Classical antiquity. Formal schools have existed at least since ancient Greece (see Academy), ancient Rome (see Education in Ancient Rome) ancient India (see Gurukul), and ancient China (see History of education in China). The Byzantine Empire had an established schooling system beginning at the primary level. According to Traditions and Encounters, the founding of the primary education system began in 425 AD and ... military personnel usually had at least a primary education .... The sometimes efficient and often large government of the Empire meant that educated citizens were a must. Although Byzantium lost much of the grandeur of Roman culture and extravagance in the process of surviving, the Empire emphasized efficiency in its war manuals. The Byzantine education system continued until the empire's collapse in 1453 AD.[4]",
    @"In Western Europe a considerable number of cathedral schools were founded during the Early Middle Ages in order to teach future clergy and administrators, with the oldest still existing, and continuously operated, cathedral schools being The King's School, Canterbury (established 597 CE), King's School, Rochester (established 604 CE), St Peter's School, York (established 627 CE) and Thetford Grammar School (established 631 CE). Beginning in the 5th century CE monastic schools were also established throughout Western Europe, teaching both religious and secular subjects.",
      @"Islam was another culture that developed a school system in the modern sense of the word. Emphasis was put on knowledge, which required a systematic way of teaching and spreading knowledge, and purpose-built structures. At first, mosques combined both religious performance and learning activities, but by the 9th century, the madrassa was introduced, a school that was built independently from the mosque, such as al-Qarawiyyin, founded in 859 CE. They were also the first to make the Madrassa system a public domain under the control of the Caliph.",
      @"Under the Ottomans, the towns of Bursa and Edirne became the main centers of learning. The Ottoman system of Külliye, a building complex containing a mosque, a hospital, madrassa, and public kitchen and dining areas, revolutionized the education system, making learning accessible to a wider public through its free meals, health care and sometimes free accommodation.",
      @"In Europe, universities emerged during the 12th century; here, scholasticism was an important tool, and the academicians were called schoolmen. During the Middle Ages and much of the Early Modern period, the main purpose of schools (as opposed to universities) was to teach the Latin language. This led to the term grammar school, which in the United States informally refers to a primary school, but in the United Kingdom means a school that selects entrants based on ability or aptitude. Following this, the school curriculum has gradually broadened to include literacy in the vernacular language as well as technical, artistic, scientific and practical subjects.",
      @"Obligatory school attendance became common in parts of Europe during the 18th century. In Denmark-Norway, this was introduced as early as in 1739-1741, the primary end being to increase the literacy of the almue, i.e. the regular people.[5] Many of the earlier public schools in the United States and elsewhere were one-room schools where a single teacher taught seven grades of boys and girls in the same classroom. Beginning in the 1920s, one-room schools were consolidated into multiple classroom facilities with transportation increasingly provided by kid hacks and school buses."
        };

            string[][] words = texts.Tokenize();

            var Bow = new BagOfWords(words);

            // Create a new TF-IDF with options:
            var codebook = new TFIDF()
            {
                Tf = TermFrequency.Log,
                Idf = InverseDocumentFrequency.Default,
            };

            // Compute the codebook (note: this would have to be done only for the training set)
            codebook.Learn(words);

            // Now, we can use the learned codebook to extract fixed-length
            // representations of the different texts (paragraphs) above:

            // Extract a feature vector from the text 1:
            List<double[]> lstDocumentsScores = new List<double[]>();

            for (int i = 0; i < texts.Length; i++)
            {
                lstDocumentsScores.Add(codebook.Transform(words[i]));
            }

            // Extract a feature vector from the text 2:
            //example
            // double[] bow2 = codebook.Transform(words[1]);
       
            var indexSerachedTerm = Bow.StringToCode["Ottomans".ToLower()];

            double dblMaxScore = double.MinValue;
            int irWhichDocument = int.MinValue;

            for (int i = 0; i < texts.Length; i++)
            {        
                if(lstDocumentsScores[i][indexSerachedTerm]> dblMaxScore)
                {
                    irWhichDocument = i;
                    dblMaxScore = lstDocumentsScores[i][indexSerachedTerm];
                }
            }
        }
    }
}
