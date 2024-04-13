using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLFagkveld;

public class Utils
{
    
    public static void Print<T>(MLContext ctx, IDataView data, int take) where T : class, new()
    {
        var firstData = ctx.Data.CreateEnumerable<T>(data, reuseRowObject: false).Take(take);
        foreach (var entitiy in firstData)
        {
            Console.WriteLine(entitiy);
        }
        Console.WriteLine();
    }
    
    public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics)
    {
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }
}