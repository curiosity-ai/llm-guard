using CsvHelper;
using HNSW.Net;
using LLM.Guard;
using MessagePack;
using System.Globalization;
using System.Linq;
using System.Numerics;

Console.WriteLine(Directory.GetCurrentDirectory());

var simpleJailbreakPromptsPath = Path.Combine(Directory.GetCurrentDirectory(), "data", "simple", "jailbreak-variations.txt"); 
var forbiddenQuestionsPath     = Path.Combine(Directory.GetCurrentDirectory(), "data", "jailbreak_llms", "data", "forbidden_question"); 
var promptsPath                = Path.Combine(Directory.GetCurrentDirectory(), "data", "jailbreak_llms", "data", "prompts");
var outputPath                 = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory().Replace($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", "§").Split(new char[] { '§' })[0], "..", "LLM.Guard", "Resources"));

Console.WriteLine(outputPath);

var simpleJailbreakPrompts = File.ReadAllLines(simpleJailbreakPromptsPath)
    .Select(f => new Prompt(f.Trim('"', '\n', ' ', '\r'), true))
    .DistinctBy(q => q.prompt)
    .ToArray();

Console.WriteLine($"Read {simpleJailbreakPrompts.Length:n0} simple jailbreak prompts");

var forbiddenQuestions = Directory.GetFiles(forbiddenQuestionsPath, "*.csv")
    .SelectMany(f =>
    {
        using (var reader = new StreamReader(f))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            var records = csv.GetRecords<ForbiddenQuestion>();
            return records.ToArray();
        }
    })
    .DistinctBy(q => q.question)
    .ToArray();

Console.WriteLine($"Read {forbiddenQuestions.Length:n0} forbidden questions");

//var jailbreakPrompts = Directory.GetFiles(promptsPath, "jailbreak*.csv")
//    .SelectMany(f =>
//    {
//        using (var reader = new StreamReader(f))
//        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
//        {
//            var records = csv.GetRecords<Prompt>();
//            return records.ToArray();
//        }
//    })
//    .DistinctBy(q => q.prompt)
//    .ToArray();

//Console.WriteLine($"Read {jailbreakPrompts.Length:n0} jailbreak prompts");

var normalPrompts = Directory.GetFiles(promptsPath, "regular*.csv")
    .SelectMany(f =>
    {
        using (var reader = new StreamReader(f))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            var records = csv.GetRecords<Prompt>();
            return records.ToArray();
        }
    })
    .DistinctBy(q => q.prompt)
    .ToArray();

Console.WriteLine($"Read {normalPrompts.Length:n0} normal prompts");

//Console.WriteLine(string.Join("\n", forbiddenQuestions.Select(f => f.content_policy_name).Distinct()));


var encoder = new SentenceTransformers.ArcticXs.SentenceEncoder();

Console.WriteLine($"Encoding forbidden questions");
var forbiddenQuestionsType    = forbiddenQuestions.ToDictionary(f => f, f => Enum.Parse<PromptType>("ForbiddenQuestion_" + f.content_policy_name.Replace(" ", "_")));
var forbiddenQuestionsEncoded = forbiddenQuestions.ToDictionary(f => f, f => new EncodedPrompt(encoder.Encode([f.question])[0], forbiddenQuestionsType[f]));

//Console.WriteLine($"Encoding jailbreak prompts");
//var jailbreakPromptsEncoded = jailbreakPrompts.ToDictionary(f => f, f => new EncodedPrompt(encoder.Encode([f.prompt])[0], PromptType.Jailbreak));

Console.WriteLine($"Encoding simple jailbreak prompts");
var simpleJailbreakPromptsEncoded = simpleJailbreakPrompts.ToDictionary(f => f, f => new EncodedPrompt(encoder.Encode([f.prompt])[0], PromptType.Jailbreak));

//Console.WriteLine($"Encoding normal prompts "); 
//var normalPromptsEncoded = normalPrompts.ToDictionary(f => f, f => new EncodedPrompt(encoder.Encode([f.prompt])[0], f.prompt, PromptType.Jailbreak));

var allPrompts = forbiddenQuestionsEncoded.Values
                    //.Concat(jailbreakPromptsEncoded.Values)
                    .Concat(simpleJailbreakPromptsEncoded.Values)
                    //.Concat(normalPromptsEncoded.Values)
                    .ToArray();

var reverseMap = forbiddenQuestionsEncoded.Select(kv => (t: kv.Key.question, p:kv.Value))
                        //.Concat(jailbreakPromptsEncoded.Select(kv => (t: kv.Key.prompt, p: kv.Value)))
                        .Concat(simpleJailbreakPromptsEncoded.Select(kv => (t: kv.Key.prompt, p: kv.Value)))
                    .ToDictionary(d => d.p,  d => d.t);

var parameters = new SmallWorld<EncodedPrompt, float>.Parameters()
{
    EnableDistanceCacheForConstruction = true,
    InitialDistanceCacheSize = 1_000_000,
    InitialItemsSize = forbiddenQuestions.Length + /*jailbreakPrompts.Length +*/ + simpleJailbreakPrompts.Length + normalPrompts.Length,
    M = 20,
    ConstructionPruning = 200,
    LevelLambda = 1 / Math.Log(40),
    ExpandBestSelection = true,
    NeighbourHeuristic = NeighbourSelectionHeuristic.SelectSimple,
};

var graph = new SmallWorld<EncodedPrompt, float>(DistanceHelper.EncodedPromptDistance, DefaultRandomGenerator.Instance, parameters);
graph.AddItems(allPrompts);

using (var output = File.Open(Path.Combine(outputPath, "hnsw-graph.bin"), FileMode.Create))
{
    graph.SerializeGraph(output);
}

using (var output = File.Open(Path.Combine(outputPath, "prompts.bin"), FileMode.Create))
{
    await MessagePackSerializer.SerializeAsync(output, graph.Items.ToArray());
}

var scoresForbidden       = new Dictionary<ForbiddenQuestion, (float, PromptType)>();
var scoresJailbreak       = new Dictionary<Prompt, (float, PromptType)>();
var scoresSimpleJailbreak = new Dictionary<Prompt, (float, PromptType)>();
var scoresNormal          = new Dictionary<Prompt, (float, PromptType)>();

Console.WriteLine("Testing predictions, please wait...");
var finalStats = new List<string>();
for (float f = 0.01f; f <= 0.19f; f+= 0.01f)
{
    int correct = 0, wrong = 0;

    foreach (var p in allPrompts)
    {
        var closest = graph.KNNSearch(p, 50);
        var best    = closest.OrderBy(kv => kv.Distance).First();
        var guess   = PromptType.Normal;

        if (best.Distance <= f)
        {
            guess = best.Item.PromptType;
        }

        if (p.PromptType == guess)
        {
            correct++;
        }
        else
        {
            wrong++;
            Console.WriteLine($"{p.PromptType} != {guess} for '{reverseMap[p]}'");
        }
    }
    finalStats.Add($"{f:n2} -> C:{correct:n0} W: {wrong:n0} P: {100f * correct / allPrompts.Length:n1}");
}

finalStats.ForEach(l => Console.WriteLine(l));

//void PrintStats(IEnumerable<KeyValuePair<ForbiddenQuestion, (float score, PromptType type)>> scores, PromptType expected)
//{
//    int correct = 0, total = 0, wrong = 0;
//    float minScore = float.MaxValue;
//    float maxScore = float.MinValue;
//    foreach (var (source, pred) in scores)
//    {
//        total++;
//        if(pred.type == expected)
//        {
//            correct++;
//        }
//        else
//        {
//            minScore = Math.Min(minScore, pred.score);
//            maxScore = Math.Max(maxScore, pred.score);
//            wrong++;
//        }
//    }
//    Console.WriteLine($"{expected} -> minScore: {minScore:n2} maxScore: {maxScore:n2} correct: {correct:n0} ({100f*correct/total:n1}%) wrong {wrong:n0} ({100f * wrong/ total:n1}%) total {total:n0}");
//}


//void PrintStatsPrompts(Dictionary<Prompt, (float score, PromptType type)> scores, PromptType expected)
//{
//    int correct = 0, total = 0, wrong = 0;
//    float minScore = float.MaxValue;
//    float maxScore = float.MinValue;
//    foreach (var (source, pred) in scores)
//    {
//        total++;
//        if (pred.type == expected)
//        {
//            correct++;
//        }
//        else
//        {
//            minScore = Math.Min(minScore, pred.score);
//            maxScore = Math.Max(maxScore, pred.score);
//            wrong++;
//        }
//    }
//    Console.WriteLine($"{expected} -> minScore: {minScore:n2} maxScore: {maxScore:n2} correct: {correct:n0} ({100f * correct / total:n1}%) wrong {wrong:n0} ({100f * wrong / total:n1}%) total {total:n0}");
//}

//void PrintStatsNormals(Dictionary<Prompt, (float score, PromptType type)> scores)
//{
//    int total = 0;
//    float minScore = float.MaxValue;
//    float maxScore = float.MinValue;
//    float avgScore = 0;
//    foreach (var (source, pred) in scores)
//    {
//        total++;
//        minScore = Math.Min(minScore, pred.score);
//        maxScore = Math.Max(maxScore, pred.score);
//        avgScore += pred.score;
//    }
//    avgScore /= total;
//    Console.WriteLine($"Normal Prompts: minScore = {minScore:n2} avgScore = {avgScore:n2} maxScore = {maxScore:n2}");
//}



public record ForbiddenQuestion(int content_policy_id, string content_policy_name, int q_id, string question);
public record Prompt(string prompt, bool jailbreak);