using Microsoft.ML.Data;

namespace MLFagkveld;


public class DatasetFileInput
{
    [LoadColumn(0)]
    public string Topic { get; set; }
    
    [LoadColumn(1)]
    public string Title { get; set; }
    
    
    public override string ToString()
    {
        return $"Title: {Title}, Topic: {Topic}";
    }
}

public class TransformedModelInput: DatasetFileInput
{
    public float[] Embedding { get; set; }
    public uint TopicKey { get; set; }
    
    public override string ToString()
    {
        return $"Title: {Title}, " +
               $"Embedding: {string.Join(", ", Embedding[0..5])} ..., " +
               $"Embedding length: {Embedding.Length}, " +
               $"Topic: {Topic}, " +
               $"TopicKey: {TopicKey}";
    }
}

class ModelOutput
{
    public string Title { get; set; }
    public uint PredictedLabel { get; set; }
    
    private string[] _topicMap = {
        "SCIENCE",
        "HEALTH",
        "SPORTS",
    };
    
    public override string ToString()
    {
        return $"Title: {Title}, Predicted key: {PredictedLabel}, Predicted topic: {_topicMap[PredictedLabel - 1]}";
    }
}
