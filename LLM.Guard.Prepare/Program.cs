using CsvHelper;
using HNSW.Net;
using LLM.Guard;
using MessagePack;
using System.Globalization;
using System.Linq;
using System.Numerics;

Console.WriteLine(Directory.GetCurrentDirectory());

var forbiddenQuestionsPath = Path.Combine(Directory.GetCurrentDirectory(), "data", "jailbreak_llms", "data", "forbidden_question"); 
var promptsPath            = Path.Combine(Directory.GetCurrentDirectory(), "data", "jailbreak_llms", "data", "prompts");
var outputPath             = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory().Replace($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", "§").Split(new char[] { '§' })[0], "..", "LLM.Guard", "Resources"));

Console.WriteLine(outputPath);

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



var jailbreakPrompts = Directory.GetFiles(promptsPath, "jailbreak*.csv")
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

Console.WriteLine($"Read {jailbreakPrompts.Length:n0} jailbreak prompts");

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

Console.WriteLine($"Encoding jailbreak prompts"); 
var jailbreakPromptsEncoded = jailbreakPrompts.ToDictionary(f => f, f => new EncodedPrompt(encoder.Encode([f.prompt])[0], PromptType.Jailbreak));

//Console.WriteLine($"Encoding normal prompts "); 
//var normalPromptsEncoded = normalPrompts.ToDictionary(f => f, f => new EncodedPrompt(encoder.Encode([f.prompt])[0], f.prompt, PromptType.Jailbreak));

using (var output = File.Open(Path.Combine(outputPath, "prompts.bin"), FileMode.Create))
{
    var all = forbiddenQuestionsEncoded.Values
        .Concat(jailbreakPromptsEncoded.Values)
        //.Concat(normalPromptsEncoded.Values)
        .ToArray();
    
    await MessagePackSerializer.SerializeAsync(output, all);
}

var parameters = new SmallWorld<EncodedPrompt, float>.Parameters()
{
    EnableDistanceCacheForConstruction = true,
    InitialDistanceCacheSize = 1_000_000,
    InitialItemsSize = forbiddenQuestions.Length + jailbreakPrompts.Length + normalPrompts.Length,
    M = 20,
    ConstructionPruning = 200,
    LevelLambda = 1 / Math.Log(40),
    ExpandBestSelection = true,
    NeighbourHeuristic = NeighbourSelectionHeuristic.SelectSimple,
};

var graph = new SmallWorld<EncodedPrompt, float>(DistanceHelper.EncodedPromptDistance, DefaultRandomGenerator.Instance, parameters);
graph.AddItems(forbiddenQuestionsEncoded.Values.ToArray());
graph.AddItems(jailbreakPromptsEncoded.Values.ToArray());
//graph.AddItems(normalPromptsEncoded.Values.ToArray());

using (var output = File.Open(Path.Combine(outputPath, "hnsw-graph.bin"), FileMode.Create))
{
    graph.SerializeGraph(output);
}

var scoresForbidden = new Dictionary<ForbiddenQuestion, (float, PromptType)>();
var scoresJailbreak = new Dictionary<Prompt, (float, PromptType)>();
var scoresNormal    = new Dictionary<Prompt, (float, PromptType)>();

Console.WriteLine("Testing predictions, please wait...");
int c = 0;
foreach (var p in forbiddenQuestions)
{
    var encoded = encoder.Encode([p.question])[0];
    var closest = graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), 50);
    var best = closest.OrderBy(kv => kv.Distance).First();
    scoresForbidden[p] = (best.Distance, best.Item.PromptType);
    if (c++ % 50 == 0) Console.WriteLine($"Forbidden Questions: At {c:n0} of {forbiddenQuestions.Length:n0}");
}

c = 0;
Random.Shared.Shuffle(jailbreakPrompts);
foreach (var p in jailbreakPrompts.Take(300))
{
    var encoded = encoder.Encode([p.prompt])[0];
    var closest = graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), 50);
    var best = closest.OrderBy(kv => kv.Distance).First();
    scoresJailbreak[p] = (best.Distance, best.Item.PromptType);
    if (c++ % 50 == 0) Console.WriteLine($"Jailbreaks: At {c:n0} of {jailbreakPrompts.Length:n0}");
}

c = 0;
Random.Shared.Shuffle(normalPrompts);
foreach (var p in normalPrompts.Take(300))
{
    var encoded = encoder.Encode([p.prompt])[0];
    var closest = graph.KNNSearch(new EncodedPrompt(encoded, PromptType.Normal), 50);
    var best = closest.OrderBy(kv => kv.Distance).First();
    scoresNormal[p] = best.Distance < 0.2f ? (best.Distance, best.Item.PromptType) : (0, PromptType.Normal);
    if (c++ % 50 == 0) Console.WriteLine($"Normal Prompts: At {c:n0} of {normalPrompts.Length:n0}");
}

Console.WriteLine("Done testing predictions.\n\n");

PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Illegal_Activity),    PromptType.ForbiddenQuestion_Illegal_Activity);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Hate_Speech),         PromptType.ForbiddenQuestion_Hate_Speech);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Malware),             PromptType.ForbiddenQuestion_Malware);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Physical_Harm),       PromptType.ForbiddenQuestion_Physical_Harm);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Economic_Harm),       PromptType.ForbiddenQuestion_Economic_Harm);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Fraud),               PromptType.ForbiddenQuestion_Fraud);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Pornography),         PromptType.ForbiddenQuestion_Pornography);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Political_Lobbying),  PromptType.ForbiddenQuestion_Political_Lobbying);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Privacy_Violence),    PromptType.ForbiddenQuestion_Privacy_Violence);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Legal_Opinion),       PromptType.ForbiddenQuestion_Legal_Opinion);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Financial_Advice),    PromptType.ForbiddenQuestion_Financial_Advice);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Health_Consultation), PromptType.ForbiddenQuestion_Health_Consultation);
PrintStats(scoresForbidden.Where(p => forbiddenQuestionsType[p.Key] == PromptType.ForbiddenQuestion_Gov_Decision),        PromptType.ForbiddenQuestion_Gov_Decision);

PrintStatsPrompts(scoresJailbreak, PromptType.Jailbreak);

PrintStatsNormals(scoresNormal);


void PrintStats(IEnumerable<KeyValuePair<ForbiddenQuestion, (float score, PromptType type)>> scores, PromptType expected)
{
    int correct = 0, total = 0, wrong = 0;
    float minScore = float.MaxValue;
    float maxScore = float.MinValue;
    foreach (var (source, pred) in scores)
    {
        total++;
        if(pred.type == expected)
        {
            correct++;
        }
        else
        {
            minScore = Math.Min(minScore, pred.score);
            maxScore = Math.Max(maxScore, pred.score);
            wrong++;
        }
    }
    Console.WriteLine($"{expected} -> minScore: {minScore:n2} maxScore: {maxScore:n2} correct: {correct:n0} ({100f*correct/total:n1}%) wrong {wrong:n0} ({100f * wrong/ total:n1}%) total {total:n0}");
}


void PrintStatsPrompts(Dictionary<Prompt, (float score, PromptType type)> scores, PromptType expected)
{
    int correct = 0, total = 0, wrong = 0;
    float minScore = float.MaxValue;
    float maxScore = float.MinValue;
    foreach (var (source, pred) in scores)
    {
        total++;
        if (pred.type == expected)
        {
            correct++;
        }
        else
        {
            minScore = Math.Min(minScore, pred.score);
            maxScore = Math.Max(maxScore, pred.score);
            wrong++;
        }
    }
    Console.WriteLine($"{expected} -> minScore: {minScore:n2} maxScore: {maxScore:n2} correct: {correct:n0} ({100f * correct / total:n1}%) wrong {wrong:n0} ({100f * wrong / total:n1}%) total {total:n0}");
}

void PrintStatsNormals(Dictionary<Prompt, (float score, PromptType type)> scores)
{
    int total = 0;
    float minScore = float.MaxValue;
    float maxScore = float.MinValue;
    float avgScore = 0;
    foreach (var (source, pred) in scores)
    {
        total++;
        minScore = Math.Min(minScore, pred.score);
        maxScore = Math.Max(maxScore, pred.score);
        avgScore += pred.score;
    }
    avgScore /= total;
    Console.WriteLine($"Normal Prompts: minScore = {minScore:n2} avgScore = {avgScore:n2} maxScore = {maxScore:n2}");
}



public record ForbiddenQuestion(int content_policy_id, string content_policy_name, int q_id, string question);
public record Prompt(string prompt, bool jailbreak);