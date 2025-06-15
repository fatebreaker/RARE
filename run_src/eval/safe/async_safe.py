import collections
import dataclasses
from typing import Any
import asyncio
from safe_common import open_modeling  # , modeling
from safe_common import utils
from eval.safe import classify_relevance
from eval.safe import get_atomic_facts
from eval.safe import colbert_rate_atomic_fact
from tqdm import tqdm

IRRELEVANT_LABEL = 'Irrelevant'
SUPPORTED_LABEL = colbert_rate_atomic_fact.SUPPORTED_LABEL
NOT_SUPPORTED_LABEL = colbert_rate_atomic_fact.NOT_SUPPORTED_LABEL

_MAX_PIPELINE_RETRIES = 3


class CheckedStatement:
    """Class for storing checked statements."""

    def __init__(
        self,
        sentence: str,
        atomic_fact: str,
        self_contained_atomic_fact: str,
        relevance_data: dict[str, Any] | None = None,
        rate_data: colbert_rate_atomic_fact.FinalAnswer | None = None,
        annotation: str = '',
    ):
        self.sentence = sentence
        self.atomic_fact = atomic_fact
        self.self_contained_atomic_fact = self_contained_atomic_fact
        self.relevance_data = relevance_data
        self.rate_data = rate_data
        self.annotation = annotation
        self.data = {
            'sentence': self.sentence,
            'atomic_fact': self.atomic_fact,
            'self_contained_atomic_fact': self.self_contained_atomic_fact,
            'relevance_data': self.relevance_data if self.relevance_data else None,
            'rate_data': (
                dataclasses.asdict(self.rate_data) if self.rate_data else None
            ),
            'annotation': self.annotation,
        }


def count_labels(checked_statements: list[CheckedStatement]) -> dict[str, int]:
    """Extract scores from the checked statements for a single response."""
    result_dict = collections.defaultdict(int)

    # Ensure that these labels are in the dictionary
    for label in [SUPPORTED_LABEL, IRRELEVANT_LABEL, NOT_SUPPORTED_LABEL]:
        result_dict[label] = 0

    for statement in checked_statements:
        if not isinstance(statement, CheckedStatement) or not statement.annotation:
            continue

        if statement.annotation.lower() == SUPPORTED_LABEL.lower():
            result_dict[SUPPORTED_LABEL] += 1
        elif statement.annotation.lower() == IRRELEVANT_LABEL.lower():
            result_dict[IRRELEVANT_LABEL] += 1
        elif statement.annotation.lower() == NOT_SUPPORTED_LABEL.lower():
            result_dict[NOT_SUPPORTED_LABEL] += 1
        else:
            result_dict[statement.annotation] += 1
            utils.maybe_print_error(
                f'Unknown statement factuality type: {statement.annotation}'
            )

    return dict(result_dict)


async def classify_relevance_and_rate_single(
    prompt: str,
    response: str,
    sentence: str,
    atomic_fact: str,
    rater: open_modeling.Model,
) -> tuple[CheckedStatement, dict[str, Any], dict[str, Any]]:
    """Classify relevance of and rate a single atomic fact."""
    context = prompt
    rate_data, past_steps_dict = colbert_rate_atomic_fact.check_atomic_fact(
        atomic_fact=atomic_fact, context=context, rater=rater
    )
    print(atomic_fact, rate_data.answer)

    if not isinstance(rate_data, colbert_rate_atomic_fact.FinalAnswer):
        raise ValueError('No rate data found for atomic fact.')

    checked_statement = CheckedStatement(
        sentence=sentence,
        atomic_fact=atomic_fact,
        self_contained_atomic_fact=atomic_fact,
        relevance_data=None,
        rate_data=rate_data,
        annotation=rate_data.answer,
    )

    return checked_statement, None, past_steps_dict


async def check_sentence(prompt, response, sentence, atomic_fact, rater):
    checked_statement, num_fails = None, 0
    revised_fact_dict, past_steps_dict = {}, {}

    while checked_statement is None and num_fails < _MAX_PIPELINE_RETRIES:
        try:
            checked_statement, revised_fact_dict, past_steps_dict = await classify_relevance_and_rate_single(
                prompt=prompt,
                response=response,
                sentence=sentence,
                atomic_fact=atomic_fact,
                rater=rater,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            utils.maybe_print_error(e)
            checked_statement, revised_fact_dict, past_steps_dict = None, {}, {}
            num_fails += 1
    return checked_statement, revised_fact_dict, past_steps_dict


async def classify_relevance_and_rate(
    prompt: str,
    response: str,
    sentences_and_atomic_facts: list[dict[str, Any]],
    rater: open_modeling.Model,
) -> dict[str, Any]:
    """Classify relevance of and rate all given atomic facts."""
    checked_statements, revised_fact_dicts, past_steps_dicts = [], [], []

    # Create a list of async tasks instead of ThreadPoolExecutor
    tasks = [
        check_sentence(prompt, response, sentence_data['sentence'], sentence_data['atomic_facts'][0], rater)
        for sentence_data in sentences_and_atomic_facts
    ]

    # Gather results asynchronously
    future_results = await asyncio.gather(*tasks)

    for result in future_results:
        checked_statement, revised_fact_dict, past_steps_dict = result
        if isinstance(checked_statement, CheckedStatement):
            checked_statements.append(checked_statement)
            revised_fact_dicts.append(revised_fact_dict)
            past_steps_dicts.append(past_steps_dict)

    return {
        'checked_statements': [item.data for item in checked_statements],
        'revised_fact_jsonified_all': revised_fact_dicts,
        'past_steps_jsonified_all': past_steps_dicts,
        **count_labels(checked_statements=checked_statements),
    }


async def main(prompt: str, response: str, rater: open_modeling.Model, split_fact: bool) -> dict[str, Any]:
    atomic_facts = get_atomic_facts.main(response=response, model=rater, split_fact=split_fact)
    rating_result = await classify_relevance_and_rate(
        prompt=prompt,
        response=response,
        sentences_and_atomic_facts=atomic_facts['all_atomic_facts'],
        rater=rater,
    )
    return {
        'prompt': prompt, 'response': response, **atomic_facts, **rating_result
    }
