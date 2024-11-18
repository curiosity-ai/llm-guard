using HNSW.Net;
using MessagePack;
using SentenceTransformers.ArcticXs;
using System.Text;

namespace LLM.Guard;

public sealed class Predictor
{
    public static readonly Predictor Instance = new Predictor();
    private readonly EncodedPrompt[] _items;
    private readonly SmallWorld<EncodedPrompt, float> _graph;
    private readonly SentenceEncoder _encoder;
    private readonly Dictionary<char, char> _homoglyphs;

    private Predictor()
    {
        using (var hnswGraph  = ResourceLoader.OpenResource(typeof(Predictor).Assembly, "hnsw-graph.bin"))
        using (var prompts    = ResourceLoader.OpenResource(typeof(Predictor).Assembly, "prompts.bin"))
        using (var homoglyphs = ResourceLoader.OpenResource(typeof(Predictor).Assembly, "homoglyphs.txt"))
        using (var homoglyphsReader = new StreamReader(homoglyphs))
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
            _homoglyphs = homoglyphsReader.ReadToEnd().Split(['\r', '\n'])
                            .Where(l => l.Length > 1 && l[0] != '#')
                            .SelectMany(l => l.Skip(1).Select(c => (final: l[0], variation: c)))
                            .DistinctBy(d => d.variation)
                            .ToDictionary(d => d.variation, d => d.final);
        }
    }

    public PromptType Predict(string text, int count = 50, float maxDistance = 0.1f, bool removeHomoglyphs = true)
    {
        var encoded = _encoder.Encode([removeHomoglyphs ? RemoveHomoglyphs(text) : text])[0];
        var closest = _graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), count);
        var best = closest.OrderBy(kv => kv.Distance).First();

        if (best.Distance < maxDistance) return best.Item.PromptType;

        return PromptType.Normal;
    }

    public Prediction PredictWithScore(string text, int count = 50, float maxDistance = 0.1f, bool removeHomoglyphs = true)
    {
        var encoded = _encoder.Encode([removeHomoglyphs ? RemoveHomoglyphs(text) : text])[0];
        var closest = _graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), count);
        var best = closest.OrderBy(kv => kv.Distance).First();

        if (best.Distance < maxDistance) return new (best.Item.PromptType, best.Distance);

        return new (PromptType.Normal, best.Distance);
    }

    public string RemoveHomoglyphs(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach(var c in text)
        {
            if(_homoglyphs.TryGetValue(c, out var r))
            {
                sb.Append(r);
            }
            else
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }
}
