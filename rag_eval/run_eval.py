from ragas.metrics.collections import Faithfulness, AnswerRelevancy, AnswerCorrectness
from utils import llm
from utils.s3 import DATA_VERSION
from rag_inference import rag
import pandas as pd
from tqdm import tqdm


def results_path(metric: str) -> str:
    # tag results with the data version so runs against different chunkings sit side by side
    return f'rag_eval/files/{metric}_{DATA_VERSION}.csv'

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
        response = rag.bedrock_rag(query)
        answer = response["answer"]
        if metric == 'correctness':
            ref = row[answer_col]
            retrieved_contexts = row[context_col]
        else:
            # the parent sections/tables the model was given, which is what the answer should be faithful to
            retrieved_contexts = response["contexts"]

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
                    response=answer,  # the RAG answer being graded
                    reference=ref     # the eval set's ground-truth answer
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

faithful_df = run_metric(eval_df, faithfulness_scorer, 'faithfulness')
faithful_df.to_csv(results_path('faithfulness'))

# relevance_df = run_metric(eval_df, relevancy_scorer, 'relevancy')
# relevance_df.to_csv(results_path('relevance'))

# correctness_df = run_metric(eval_df, correctness_scorer, 'correctness',
#                             answer_col='reference', context_col='reference_contexts')
# correctness_df.to_csv(results_path('correctness'))
