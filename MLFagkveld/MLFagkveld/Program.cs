using Microsoft.ML;
using MLFagkveld;
using Microsoft.ML.Transforms.Text;

var ctx = new MLContext();

/* TODO 1 
 * Last inn datasettet fra filen og del opp i trenings og testsett
 * Hint1: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/load-data-ml-net
 * Hint2: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-ml-net#split-data-for-training-and-testing
 */


// Skriv ut de første 10 radene i treningssettet for å undersøke dataene
// Utils.Print<DatasetFileInput>(ctx, trainingData, 10);


/* TODO 2
 * Lag en pipeline som transformerer dataene slik at vi kan bruke de i en maskinlæringsmodell.
 * 1. Normaliser teksten i Title kolonnen
 * 2. Tokenize teksten i Title kolonnen
 * 3. Bruk GloVe 50D pre-trained word embedding for å konvertere tekst til vektorer. Får du ulike resultater ved å bruke forskjellige pre-trained word embeddings?
 * Hint: https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.textcatalog.applywordembedding?view=ml-dotnet
 */



/* TODO 3
 * Lag en maskinlæringsmodell som predikerer TopicKey basert på Embedding
 * Hint (multiclass trainers): https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.multiclassclassificationcatalog.multiclassclassificationtrainers?view=ml-dotnet
 */


/* TODO 4
 * Tren modellen og evaluer resultatene
 * Hint: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-ml-net#train-the-model
 */


/* TODO 5
 * Bruk modellen til å predikere TopicKey for noen eksempler
 * Hint: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/machine-learning-model-predictions-ml-net
 * Ekstra: Lagre modellen og kjør prediksjoner i en annen applikasjon: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net
 */

