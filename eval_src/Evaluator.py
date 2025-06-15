# Licensed under the MIT license.

from eval_src.toolkit_for_MATH.latex_answer_check import latex_answer_check as latex_equiv

import os, json, re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process
import numpy as np
from thefuzz import fuzz
from collections import Counter

class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text

    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_intermediate_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass
        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                    len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                [most_confident_answer],
                [answer2completions[most_confident_answer][0]],
                [answer2ids[most_confident_answer][0]],
                [confidence],
            )

    def check_valid_answer(self, answer: str, completion, options: dict, threshold):
        highest_score = -1
        for char, opt in options.items():
            score = fuzz.ratio(answer.lower(), opt.lower())
            if score > highest_score:
                highest_score = score
                highest_option = char
        valid = highest_score >= threshold
        return valid, highest_option
        # return True, highest_option

    def extract_reward_from_critic(self, critic):
        sc_idx = critic.rfind("score is")
        if sc_idx == -1:
            return -1
        score_str = critic[sc_idx+9:sc_idx+11]
        try:
            score = int(score_str)
            return score
        except:
            return -1


    def find_threshold(self, completions, options):
        threshold = 60
        while threshold > 0:
            num_valid_completions = 0
            for id, c in enumerate(completions):
                try:
                    model_answer = self.extract_answer_from_model_completion(c)
                    valid_answer, choice = self.check_valid_answer(model_answer, c, options, threshold)
                    if not valid_answer:
                        continue
                    num_valid_completions += 1
                except:
                    pass
            if num_valid_completions > 0:
                return threshold
            threshold = threshold - 10
        return threshold


    def find_most_confident_answer_with_options(self, completions: List[str], options: dict, prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None
        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        num_valid_completions = 0
        threshold = self.find_threshold(completions, options)
        import ipdb; ipdb.set_trace()
        # threshold = 0
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                import ipdb; ipdb.set_trace()
                valid_answer, choice = self.check_valid_answer(model_answer, c, options, threshold)
                if not valid_answer:
                    continue
                num_valid_completions += 1
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if existing_answer == choice:
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[choice].append(c)
                    answer2ids[choice].append(id)
            except:
                pass
        if len(answer2completions.keys()) == 0:
            sample_index = random.randint(0, len(completions)-1)
            confidence = 1.0 / len(completions)
            most_completion = completions[sample_index]
            return None, [most_completion], None, [confidence]
        # assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            valid_choices = []
            valid_confidences = []
            valid_completions = []
            for choice in answer2completions.keys():
                confidence  = len(answer2completions[choice]) / num_valid_completions
                completion = answer2completions[choice][0]
                valid_confidences.append(confidence)
                valid_completions.append(completion)
                valid_choices.append(choice)
            # Sorting by confidence in decreasing order
            sorted_data = sorted(
                zip(valid_confidences, valid_choices, valid_completions),
                key=lambda x: x[0],  # Sort by the confidence (first item in tuple)
                reverse=True  # Sort in decreasing order
            )
            valid_confidences, valid_choices, valid_completions = zip(*sorted_data)
            # import ipdb; ipdb.set_trace()
            # return (
            #     valid_choices,
            #     valid_completions,
            #     None,
            #     valid_confidences,
            # )
            return (
                [valid_choices[0]],
                [valid_completions[0]],
                None,
                [valid_confidences[0]],
            )

    def find_most_confident_answer_with_options_and_critiques(self, completions: List[str], options: dict, critiques: List[str]):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None
        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        answer2score = defaultdict(list)
        threshold = self.find_threshold(completions, options)
        num_valid_completions = 0
        for id, c in enumerate(completions):
            try:
                critic = critiques[id]
                model_answer = self.extract_answer_from_model_completion(c)
                valid_answer, choice = self.check_valid_answer(model_answer, c, options, threshold)
                score = self.extract_reward_from_critic(critic)
                if score == -1:
                    continue
                if not valid_answer:
                    continue
                num_valid_completions += 1
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if existing_answer == choice:
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                        answer2score[existing_answer].append(score)
                if not has_existed:
                    answer2completions[choice].append(c)
                    answer2ids[choice].append(id)
                    answer2score[choice].append(score)
            except:
                pass
        if len(answer2completions.keys()) == 0:
            sample_index = random.randint(0, len(completions)-1)
            confidence = 1.0 / len(completions)
            most_completion = completions[sample_index]
            return None, [most_completion], None, [confidence]

        valid_choices = []
        valid_confidences = []
        valid_completions = []
        for choice in answer2completions.keys():
            confidence  = sum(answer2score[choice]) / 100
            completion = ""
            for kdx in range(len(answer2completions[choice])):
                if answer2score[choice][kdx] == max(answer2score[choice]):
                    completion = answer2completions[choice][kdx]
                    break
            valid_confidences.append(confidence)
            valid_completions.append(completion)
            valid_choices.append(choice)
        # import ipdb; ipdb.set_trace()
        # Sorting by confidence in decreasing order
        sorted_data = sorted(
            zip(valid_confidences, valid_choices, valid_completions),
            key=lambda x: x[0],  # Sort by the confidence (first item in tuple)
            reverse=True  # Sort in decreasing order
        )
        valid_confidences, valid_choices, valid_completions = zip(*sorted_data)
        return (
            valid_choices,
            valid_completions,
            None,
            valid_confidences,
        )

    def find_most_confident_answer_with_critiques(self, completions: List[str], critiques: List[str]):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None
        valid_choices = []
        valid_confidences = []
        valid_completions = []
        for id, c in enumerate(completions):
            try:
                critic = critiques[id]
                model_answer = self.extract_answer_from_model_completion(c)
                score = self.extract_reward_from_critic(critic)
                if score == -1:
                    continue
                valid_confidences.append(score/100)
                valid_completions.append(c)
                valid_choices.append(model_answer)
            except:
                pass
        # Sorting by confidence in decreasing order
        sorted_data = sorted(
            zip(valid_confidences, valid_choices, valid_completions),
            key=lambda x: x[0],  # Sort by the confidence (first item in tuple)
            reverse=True  # Sort in decreasing order
        )
        valid_confidences, valid_choices, valid_completions = zip(*sorted_data)
        return (
            valid_choices,
            valid_completions,
            None,
            valid_confidences,
        )


    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
        prior_weights: List[float] = None,
    ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError


class GSM8KEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct

    def extract_answer_from_gold_solution(self, solution):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.split("#### ")[-1].strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


class MedQAEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv_with_options(self, answer_a: str, answer_b: str, options: dict):
        """Judge whether two answers are equivalent."""
        op_lst = ['A', 'B', 'C', 'D', 'E']
        pred_score = []
        gol_score = []
        for char, opt in options.items():
            pred_score.append(fuzz.ratio(answer_a.lower(), opt.lower()))
        for char, opt in options.items():
            gol_score.append(fuzz.ratio(answer_b.lower(), opt.lower()))
        pred_score = np.array(pred_score)
        gol_score = np.array(gol_score)
        option_pred = options[op_lst[np.argmax(pred_score)]]
        option_gol = options[op_lst[np.argmax(gol_score)]]
        correct = option_gol == option_pred
        return correct

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        score = fuzz.ratio(answer_a.lower(), answer_b.lower())
        correct = score >= 90
        return correct

    def extract_answer_from_gold_solution(self, solution: str):
        """Extract the answer from the gold solution."""
        return solution

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None
        start_idx = completion.lower().rfind("the answer is")
        if start_idx == -1:
            if len(completion.split(".")) > 1:
                return completion.split(".")[-2]
            else:
                return completion
        answer = completion[start_idx+17:]
        if len(answer) > 1 and answer[-1] == ".":
            answer = answer[:-1]
        return answer


    def extract_intermediate_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None
        start_idx = completion.lower().rfind("the answer is")
        if start_idx == -1:
            return completion
        answer = completion[start_idx+14:]
        if len(answer) > 1 and answer[-1] == ".":
            answer = answer[:-1]
        return answer


    def get_majority_answer(self, answers, options):
        op_lst = ['A', 'B', 'C', 'D', 'E']
        all_choices = []
        for answer in answers:
            pred_score = []
            for char, opt in options.items():
                pred_score.append(fuzz.ratio(answer, opt))
            pred_score = np.array(pred_score)
            option_pred = op_lst[np.argmax(pred_score)]
            all_choices.append(option_pred)
        most_common_choice, freq = Counter(all_choices).most_common(1)[0]
        return options[most_common_choice], freq, all_choices


class ChatDoctorEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        score = fuzz.ratio(answer_a.lower(), answer_b.lower())
        correct = score >= 90
        return correct

    def extract_answer_from_gold_solution(self, solution: str):
        """Extract the answer from the gold solution."""
        return solution

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None
        return completion


    def extract_intermediate_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None
        return completion


# class MedMCQAEvaluator(Evaluator):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def check_answers_equiv_with_options(self, answer_a: str, answer_b: str, options: dict):
#         """Judge whether two answers are equivalent."""
#         op_lst = ['A', 'B', 'C', 'D']
#         pred_score = []
#         gol_score = []
#         for char, opt in options.items():
#             pred_score.append(fuzz.ratio(answer_a.lower(), opt.lower()))
#         for char, opt in options.items():
#             gol_score.append(fuzz.ratio(answer_b.lower(), opt.lower()))
#         pred_score = np.array(pred_score)
#         gol_score = np.array(gol_score)
#         option_pred = options[op_lst[np.argmax(pred_score)]]
#         option_gol = options[op_lst[np.argmax(gol_score)]]
#         correct = option_gol == option_pred
#         return correct
#
#     def check_answers_equiv(self, answer_a: str, answer_b: str):
#         score = fuzz.ratio(answer_a.lower(), answer_b.lower())
#         correct = score >= 90
#         return correct
#
#     def extract_answer_from_gold_solution(self, solution: str):
#         """Extract the answer from the gold solution."""
#         return solution
#
#     def extract_answer_from_model_completion(self, completion: str):
#         """Extract the answer from the model completion."""
#         if completion is None:
#             return None
#         start_idx = completion.lower().rfind("the answer is")
#         if start_idx == -1:
#             return completion.split(".")[-2]
#         answer = completion[start_idx+17:]
#         if answer[-1] == ".":
#             answer = answer[:-1]
#         return answer
#
#
#     def get_majority_answer(self, answers, options):
#         op_lst = ['A', 'B', 'C', 'D']
#         all_choices = []
#         for answer in answers:
#             pred_score = []
#             for char, opt in options.items():
#                 pred_score.append(fuzz.ratio(answer, opt))
#             pred_score = np.array(pred_score)
#             option_pred = op_lst[np.argmax(pred_score)]
#             all_choices.append(option_pred)
#         most_common_choice, freq = Counter(all_choices).most_common(1)[0]
#         return options[most_common_choice], freq, all_choices


GSM8KHARDEvaluator = GSM8KEvaluator
MULTIARITHEvaluator = GSM8KEvaluator
Ari_MedQAEvaluator = MedQAEvaluator
MedMCQAEvaluator = MedQAEvaluator
MMLUEvaluator = MedQAEvaluator
STGEvaluator = MedQAEvaluator
CommonsenseQAEvaluator = MedQAEvaluator
SIQAEvaluator = MedQAEvaluator
PIQAEvaluator = MedQAEvaluator


class MATHEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True

        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False

        return res

    def extract_answer_from_gold_solution(self, solution: str):
        def remove_boxed(s):
            left = "\\boxed{"
            try:
                assert s[: len(left)] == left
                assert s[-1] == "}"
                return s[len(left) : -1]
            except:
                return None

        def last_boxed_only_string(string):
            idx = string.rfind("\\boxed")
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None

            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1

            if right_brace_idx == None:
                retval = None
            else:
                retval = string[idx : right_brace_idx + 1]

            return retval

        return remove_boxed(last_boxed_only_string(solution))

    def extract_answer_from_model_completion(self, completion):
        answer_split = self.isolate_answer(completion)
        return answer_split


class SVAMPEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct

    def extract_answer_from_gold_solution(self, solution):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


# class STGEvaluator(Evaluator):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def _format_answer(self, answer: str):
#         if answer.lower() in ["proved", "true", "yes", "correct", "positive", "affirmative", "right", "1", "t", "y"]:
#             return "true"
#         elif answer.lower() in ["disproved", "false", "no", "incorrect", "negative", "wrong", "0", "f", "n"]:
#             return "false"
#         else:
#             return answer.lower()
#
#     def check_answers_equiv(self, answer_a: str, answer_b: str):
#         if answer_a is None or answer_b is None:
#             return False
#
#         assert isinstance(answer_a, str) and isinstance(answer_b, str)
#
#         format_answer_a = self._format_answer(answer_a)
#         format_answer_b = self._format_answer(answer_b)
#         return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 90
#
#     def extract_answer_from_gold_solution(self, solution: str):
#         if solution is None:
#             return None
#
#         assert isinstance(solution, str)
#
#         return self._format_answer(solution)
#
#     def extract_answer_from_model_completion(self, completion: str):
#         if completion is None:
#             return None
#
#         assert isinstance(completion, str)
#
#         answer = self.isolate_answer(completion)
#         if answer is None:
#             return None
#
#         return self._format_answer(answer)
