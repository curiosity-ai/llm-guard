using HNSW.Net;
using MessagePack;
using SentenceTransformers.ArcticXs;

namespace LLM.Guard;

public sealed class Predictor
{
    public static readonly Predictor Instance = new Predictor();
    private readonly EncodedPrompt[] _items;
    private readonly SmallWorld<EncodedPrompt, float> _graph;
    private readonly SentenceEncoder _encoder;

    private Predictor()
    {
        using (var hnswGraph = ResourceLoader.OpenResource(typeof(Predictor).Assembly, "hnsw-graph.bin"))
        using (var prompts   = ResourceLoader.OpenResource(typeof(Predictor).Assembly, "prompts.bin"))
        {
            var parameters = new SmallWorld<EncodedPrompt, float>.Parameters()
            {
                M = 20,
                ConstructionPruning = 200,
                LevelLambda = 1 / Math.Log(40),
                ExpandBestSelection = true,
                NeighbourHeuristic = NeighbourSelectionHeuristic.SelectSimple,
            };

            _items = MessagePackSerializer.Deserialize<EncodedPrompt[]>(prompts);
            _graph = SmallWorld<EncodedPrompt, float>.DeserializeGraph(_items, DistanceHelper.EncodedPromptDistance, DefaultRandomGenerator.Instance, hnswGraph);
            _encoder = new SentenceTransformers.ArcticXs.SentenceEncoder();
        }
    }

    public PromptType Predict(string text, int count = 50, float maxDistance = 0.2f)
    {
        var encoded = _encoder.Encode([text])[0];
        var closest = _graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), count);
        var best = closest.OrderBy(kv => kv.Distance).First();

        if (best.Distance < maxDistance) return best.Item.PromptType;

        return PromptType.Normal;
    }

    public Prediction PredictWithScore(string text, int count = 50, float maxDistance = 0.2f)
    {
        var encoded = _encoder.Encode([text])[0];
        var closest = _graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), count);
        var best = closest.OrderBy(kv => kv.Distance).First();

        if (best.Distance < maxDistance) return new (best.Item.PromptType, best.Distance);

        return new (PromptType.Normal, best.Distance);
    }
}
