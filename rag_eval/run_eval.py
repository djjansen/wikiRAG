from ragas.metrics.collections import Faithfulness, AnswerRelevancy, AnswerCorrectness
from utils import llm
import pandas as pd
from tqdm import tqdm

# Create metric
faithfulness_scorer = Faithfulness(llm=llm.instructor_llm)
relevancy_scorer = AnswerRelevancy(llm=llm.instructor_llm, embeddings=llm.embeddings)
correctness_scorer = AnswerCorrectness(llm=llm.instructor_llm, embeddings=llm.embeddings)

eval_df = pd.read_json('rag_eval/files/test_set.jsonl', lines=True)

def run_metric(eval_df: pd.DataFrame, scorer, metric: str, user_input_col='user_input',
                      answer_col=None, context_col =None) -> pd.DataFrame:
    scores = []
    answers = []
    contexts = []

    for idx, row in tqdm(eval_df.iterrows()):
        query = row[user_input_col]
        response = llm.bedrock_rag(query)
        answer = response["output"]["text"]
        if metric == 'correctness':
            ref = row[answer_col]
            retrieved_contexts = row[context_col]
        else:
            retrieved_contexts = []
            seen = set()
            for citation in response.get("citations", []):
                for ref in citation.get("retrievedReferences", []):
                    text = ref.get("content", {}).get("text")
                    if text and text not in seen:
                        seen.add(text)
                        retrieved_contexts.append(text)

        try:
            if metric == 'faithfulness':
                result = scorer.score(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                )

            elif metric == 'correctness':
                result = scorer.score(
                    user_input=query,
                    response=ref,
                    reference=answer
                )                
            else:
                result = scorer.score(
                    user_input=query,
                    response=answer,
                )
            score = result.value
        except Exception as e:
            print(f"row {idx} failed: {type(e).__name__}: {e}")
            score = None


        scores.append(score)
        answers.append(answer)
        contexts.append(retrieved_contexts)

    eval_df[f'{metric}_score'] = scores
    eval_df[f'{metric}_answer'] = answers
    eval_df[f'{metric}_context'] = contexts

    return eval_df

# faithful_df = run_metric(eval_df, faithfulness_scorer, 'faithfulness')
# faithful_df.to_csv('rag_eval/files/faithfulness.csv')

# relevance_df = run_metric(eval_df, relevancy_scorer, 'relevancy')
# relevance_df.to_csv('rag_eval/files/relevance.csv')

correctness_df = run_metric(eval_df, correctness_scorer, 'correctness',
                            answer_col='reference', context_col='reference_contexts')
correctness_df.to_csv('rag_eval/files/correctness.csv')
