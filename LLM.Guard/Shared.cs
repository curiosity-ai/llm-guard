using MessagePack;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace LLM.Guard;

[MessagePackObject(keyAsPropertyName: true)]
public record EncodedPrompt(float[] Vector, PromptType PromptType);

public enum PromptType
{
    Normal,
    Jailbreak,
    ForbiddenQuestion_Illegal_Activity,
    ForbiddenQuestion_Hate_Speech,
    ForbiddenQuestion_Malware,
    ForbiddenQuestion_Physical_Harm,
    ForbiddenQuestion_Economic_Harm,
    ForbiddenQuestion_Fraud,
    ForbiddenQuestion_Pornography,
    ForbiddenQuestion_Political_Lobbying,
    ForbiddenQuestion_Privacy_Violence,
    ForbiddenQuestion_Legal_Opinion,
    ForbiddenQuestion_Financial_Advice,
    ForbiddenQuestion_Health_Consultation,
    ForbiddenQuestion_Gov_Decision,
}

public record struct Prediction(PromptType Type, float Score);

public static class DistanceHelper
{
    public static float EncodedPromptDistance(EncodedPrompt a, EncodedPrompt b)
    {
        ReadOnlySpan<float> u = a.Vector.AsSpan();
        ReadOnlySpan<float> v = b.Vector.AsSpan();

        if (u.Length != v.Length)
        {
            throw new Exception("Invalid distance");
        }

        float dot = 0;
        var norm = default(Vector2);
        int step = Vector<float>.Count;

        int i, to = u.Length - step;

        for (i = 0; i <= to; i += step)
        {
            var ui = new Vector<float>(u.Slice(i));
            var vi = new Vector<float>(v.Slice(i));

            dot += System.Numerics.Vector.Dot(ui, vi);
            norm.X += System.Numerics.Vector.Dot(ui, ui);
            norm.Y += System.Numerics.Vector.Dot(vi, vi);
        }

        for (; i < u.Length; ++i)
        {
            dot += u[i] * v[i];
            norm.X += u[i] * u[i];
            norm.Y += v[i] * v[i];
        }

        norm = Vector2.SquareRoot(norm);

        float n = (norm.X * norm.Y);

        if (n == 0)
        {
            return 1f;
        }

        var similarity = dot / n;

        return 1f - similarity;
    }
}
