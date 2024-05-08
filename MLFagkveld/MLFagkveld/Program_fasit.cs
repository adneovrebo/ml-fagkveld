using Microsoft.ML;
using MLFagkveld;
using Microsoft.ML.Transforms.Text;

var ctx = new MLContext();

/* TODO 1 
 * Last inn datasettet fra filen og del opp i trenings og testsett
 * Hint1: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/load-data-ml-net
 * Hint2: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-ml-net#split-data-for-training-and-testing
 */
var dataPath = Path.Combine(Environment.CurrentDirectory, "dataset.csv");
var dataView = ctx.Data.LoadFromTextFile<DatasetFileInput>(dataPath, hasHeader: true, separatorChar: ',');
var trainTestSplit = ctx.Data.TrainTestSplit(dataView, testFraction: 0.2);
var trainingData = trainTestSplit.TrainSet;
var testData = trainTestSplit.TestSet;

// Skriv ut de første 10 radene i treningssettet for å undersøke dataene
Utils.Print<DatasetFileInput>(ctx, trainingData, 10);


/* TODO 2
 * Lag en pipeline som transformerer dataene slik at vi kan bruke de i en maskinlæringsmodell.
 * 1. Normaliser teksten i Title kolonnen
 * 2. Tokenize teksten i Title kolonnen
 * 3. Bruk GloVe 50D pre-trained word embedding for å konvertere tekst til vektorer. Får du ulike resultater ved å bruke forskjellige pre-trained word embeddings?
 * Hint: https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.textcatalog.applywordembedding?view=ml-dotnet
 */
var dataPipeline = ctx.Transforms.Text.NormalizeText(nameof(DatasetFileInput.Title))
    .Append(ctx.Transforms.Text.TokenizeIntoWords("Tokens", nameof(DatasetFileInput.Title)))
    .Append(ctx.Transforms.Text.ApplyWordEmbedding(
        "Embedding",
        "Tokens", 
        WordEmbeddingEstimator.PretrainedModelKind.GloVe50D))
    .Append(ctx.Transforms.Conversion.MapValueToKey(
        "TopicKey", nameof(DatasetFileInput.Topic))
    );



/* TODO 3
 * Lag en maskinlæringsmodell som predikerer TopicKey basert på Embedding
 * Hint (multiclass trainers): https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.multiclassclassificationcatalog.multiclassclassificationtrainers?view=ml-dotnet
 */
var trainer = ctx.MulticlassClassification.Trainers.NaiveBayes(
    nameof(TransformedModelInput.TopicKey), nameof(TransformedModelInput.Embedding)
);
var trainingPipeline = dataPipeline.Append(trainer);


/* TODO 4
 * Tren modellen og evaluer resultatene
 * Hint: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/train-machine-learning-model-ml-net#train-the-model
 */
var model = trainingPipeline.Fit(trainingData);
var predictions = model.Transform(testData);

Utils.Print<ModelOutput>(ctx, predictions, 10);

var metrics = ctx.MulticlassClassification.Evaluate(predictions, labelColumnName: nameof(TransformedModelInput.TopicKey));
Utils.PrintMulticlassClassificationMetrics(metrics);

/* TODO 5
 * Bruk modellen til å predikere TopicKey for noen eksempler
 * Hint: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/machine-learning-model-predictions-ml-net
 * Ekstra: Lagre modellen og kjør prediksjoner i en annen applikasjon: https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net
 */
var predictionEngine = ctx.Model.CreatePredictionEngine<TransformedModelInput, ModelOutput>(model);

var prediction = predictionEngine.Predict(new TransformedModelInput
{
    Title = "Case of Covid-19 confirmed at community creche in south Dublin"
});
Console.WriteLine($"Prediction: {prediction}");


var prediction2 = predictionEngine.Predict(new TransformedModelInput
{
    Title = "Astronauts on the International Space Station are growing radishes"
});

Console.WriteLine($"Prediction: {prediction2}");

var prediction3 = predictionEngine.Predict(new TransformedModelInput
{
    Title = "Ronaldo scores a hat-trick in the Champions League final"
});

Console.WriteLine($"Prediction: {prediction3}");