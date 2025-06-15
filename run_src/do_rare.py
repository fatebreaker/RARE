# Licensed under the MIT license.

import sys
import os, json, time
from tqdm import tqdm

sys.path.append(".")
# sys.path.append("/home/htran/generation/med_preferences/rStar")
# sys.path.append("run_src")

from common.utils import fix_seeds, setup_model_parallel, read_json
# from common.utils import *
from common.arguments import get_parser, post_process_args, save_args
from run_src.rstar_utils import GeneratorError
from run_src.rstar_utils import Node_Type
from MCTS_for_reasoning_plus import Generator, search_for_answers
from eval_src.Evaluator import *
import evaluate


def get_pattern(solution_node):
    action_trace = []
    node = solution_node
    while node.node_type is not Node_Type.USER_QUESTION:
        if node.node_type is Node_Type.OST_STEP:
            action_trace.append("A1")
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            action_trace.append("A2")
        elif node.node_type is Node_Type.SUBQUESTION:
            action_trace.append("A3")
        elif node.node_type is Node_Type.RE_SUBANSWER:
            action_trace.append("A4")
        elif node.node_type is Node_Type.DIRECT_ANSWER_RAG:
            action_trace.append("A6")
        elif node.node_type is Node_Type.RE_SUBANSWER_RAG:
            action_trace.append("A7")
        node = node.parent
    action_trace.reverse()
    return action_trace


def extract_solution_patterns(all_solution_nodes, gold_answer):
    all_patterns = []
    for node in all_solution_nodes:
        if node.get_choice() == gold_answer:
            action_trace = get_pattern(node)
            pattern = "->".join(action_trace)
            all_patterns.append(pattern)
    return all_patterns

def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    if os.path.exists(args.save_path):
        with open(args.save_path, "r") as f:
            save_generations = json.load(f)
            num_correct = save_generations[-1]['num_correct']
            num_tested = save_generations[-1]['num_tested']
            num_correct_gen = save_generations[-1]['num_gen_correct']
    else:
        save_generations = []
        num_tested = 0
        num_correct = 0
        num_correct_gen = 0

    rouge = evaluate.load('rouge')
    bleu = evaluate.load("bleu")


    evaluator = eval(f"{args.dataset_name}Evaluator()")

    tokenizer, model = None, None
    if args.api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(args.model_ckpt)
    elif args.api == "vllm":
        from models.vLLM_API import load_vLLM_model
        tokenizer, model = load_vLLM_model(args.model_ckpt, args.seed, args.tensor_parallel_size, args.half_precision)
    elif args.api == "gpt3.5-turbo":
        from models.OpenAI_API import load_OpenAI_model
        tokenizer, model = load_OpenAI_model(args.model_ckpt)
    elif args.api == "llama":
        from models.Llama_API import load_OpenAI_model
        tokenizer, model = load_OpenAI_model(args.model_ckpt)

    generator = Generator(args, tokenizer, model, evaluator)

    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < len(save_generations):
            continue

        problem_id, problem, gt_solution, options, gold_answer = data_item["id"], data_item["problem"], data_item["solution"], data_item["options"], data_item['answer']
        num_tested += 1
        best_choice, freq_choice, choice_info, all_solution_nodes, all_solutions = search_for_answers(
            args=args, user_question=problem, question_id=i, gt_answer=gt_solution, generator=generator,
            options=options,
        )
        if options is not None:
            correct = best_choice == gold_answer
            correct_gen = freq_choice == gold_answer
            if correct:
                num_correct += 1
            # else:
            #     import ipdb; ipdb.set_trace()
            if correct_gen:
                num_correct_gen += 1
            accuracy = num_correct / num_tested
            gen_accuracy = num_correct_gen / num_tested
            call_counter = generator.io.call_counter
            token_counter = generator.io.token_counter
            save_obj = {}
            save_obj['question'] = problem
            save_obj['id'] = i
            save_obj['options'] = options
            save_obj['gold_solution'] = gt_solution
            save_obj['gold_answer'] = gold_answer
            save_obj["best_choice"] = best_choice
            save_obj["frequent_choice"] = freq_choice
            save_obj["choices_info"] = choice_info
            save_obj['all_solutions'] = all_solutions
            save_obj['num_correct'] = num_correct
            save_obj['num_tested'] = num_tested
            save_obj['num_gen_correct'] = num_correct_gen
            save_obj['accuracy'] = accuracy
            patterns = extract_solution_patterns(all_solution_nodes, gold_answer)
            save_obj["correct_patterns"] = patterns
            save_obj["call_counter"] = call_counter
            save_obj["token_counter"] = token_counter
            # import ipdb; ipdb.set_trace()
            save_generations.append(save_obj)
            print("Accuracy: ", accuracy, "num tested: ", num_tested, "avg call: ", call_counter/num_tested, "avg token: ", token_counter/num_tested)
            with open(args.save_path, "w") as f:
                json.dump(save_generations, f)
        else:
            solution_scores = [node.node_value for node in all_solution_nodes]
            save_obj = {}
            save_obj['question'] = problem
            save_obj['id'] = i
            save_obj['gold_answer'] = gold_answer
            save_obj["best_answer"] = best_choice
            save_obj['all_solutions'] = all_solutions
            save_obj['all_solutions_score'] = solution_scores
            save_obj['num_correct'] = num_correct
            save_obj['num_tested'] = num_tested
            save_obj['num_gen_correct'] = num_correct_gen
            save_generations.append(save_obj)
            predictions = [obj["best_answer"] for obj in save_generations]
            references = [obj["gold_answer"] for obj in save_generations]
            rouge_results = rouge.compute(predictions=predictions, references=references)
            bleu_results = bleu.compute(predictions=predictions, references=references)
            print("BLEU: ", bleu_results["precisions"][0], "ROUGE: ", rouge_results["rouge1"], "num tested: ", num_tested)
            with open(args.save_path, "w") as f:
                json.dump(save_generations, f)



if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument(
        "--num_subquestions", type=int, default=3, help="Number of trials for proposing the next subquestion"
    )
    parser.add_argument(
        "--num_queries", type=int, default=3, help="Number of trials for proposing the next query"
    )
    parser.add_argument(
        "--num_retrieval", type=int, default=3, help="Number of documents retrieval"
    )
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")
    parser.add_argument("--search_query_weight", type=float, default=0.5)
    parser.add_argument("--combine_distributions", type=str, default="add")
    parser.add_argument("--majority_threshold", type=float, default=0.5)
    parser.add_argument("--enable_majority", type=bool, default=False)
    parser.add_argument("--retrieval_threshold", type=float, default=0.0)
    parser.add_argument("--enable_chat_template", type=bool, default=False)
    parser.add_argument("--enable_self_reward", type=bool, default=False)


    # Action1: Propose an one-step thought.
    parser.add_argument("--num_a1_steps", type=int, default=None)
    parser.add_argument("--disable_a1", action="store_true")
    parser.add_argument("--disable_a3", action="store_true")
    parser.add_argument("--disable_a4", action="store_true")
    parser.add_argument("--disable_a6", action="store_true")
    parser.add_argument("--disable_a7", action="store_true")
    parser.add_argument("--disable_a8", action="store_true")

    parser.add_argument("--enable_answer_checking", type=bool, default=False)

    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--disable_a5", action="store_true")


    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")
    parser.add_argument("--disable_answer_selection", action="store_true")
    parser.add_argument("--save_generation", action="store_true")
    parser.add_argument("--save_path", type=str, default="save/results.json")

    parser.add_argument("--retrieval_corpus", type=str, default="medcorp")
    parser.add_argument("--use_triples", action="store_true")


    #! -------------------------------------------------------------------------------

    args = parser.parse_args()

    if args.mcts_num_last_votes is None:
        if args.enable_self_reward:
            args.mcts_num_last_votes = 10
        else:
            args.mcts_num_last_votes = 32

    if not args.disable_a1:
        if args.num_a1_steps is None:
            args.num_a1_steps = 3

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)

    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")

    args.fewshot_cot_rag_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_rag_prompt.txt")
    args.fewshot_cot_rag_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_rag_config.json")

    args.fewshot_self_reward_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_self_reward_prompt.txt")
    args.fewshot_self_reward_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_self_reward_config.json")

    args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
    args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config.json")

    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args.decompose_query_template_path = os.path.join(prompts_dir, "decompose", "decompose_query_template.json")
    args.decompose_query_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_query_prompt.txt")

    if not args.disable_a5:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        if args.modify_prompts_for_rephrasing:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_ost_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args = post_process_args(args)
    print(args)
    save_args(args)
    suffix = "ro" + str(args.num_rollouts)
    if not args.disable_a6 or not args.disable_a7:
        suffix = suffix + "_s" +str(args.num_queries)
        suffix = suffix + "_r" + str(args.num_retrieval)
        suffix = suffix + "_" + args.retrieval_corpus
        if args.retrieval_threshold != 0:
            suffix = suffix + "_t" + str(int(args.retrieval_threshold * 10))
    if args.enable_answer_checking:
        suffix = suffix + "_w" + str(int(100*args.search_query_weight))
        if args.combine_distributions != "add":
            suffix = suffix + "_" + args.combine_distributions
        if args.enable_majority:
            suffix = suffix + "_" + "maj_" + str(int(args.majority_threshold * 10))
    if args.enable_chat_template:
        suffix = suffix + "_chat"
    if args.enable_self_reward:
        suffix = suffix + "_self_reward"
    if args.mcts_num_last_votes != 32:
        suffix = suffix + "_v"+str(args.mcts_num_last_votes)
    if not args.disable_a1:
        suffix = suffix + "_a1"
    if not args.disable_a3:
        suffix = suffix + "_a3"
    if not args.disable_a4:
        suffix = suffix + "_a4"
    if not args.disable_a5:
        suffix = suffix + "_a5"
    if not args.disable_a6:
        suffix = suffix + "_a6"
    if not args.disable_a7:
        suffix = suffix + "_a7"
    if not args.disable_a8:
        suffix = suffix + "_a8"

    args.save_path = "save/" + args.model_ckpt.split("/")[-1] + "_" + args.dataset_name + "_"  + suffix + "_f.json"
    print("save generation to ", args.save_path)
    main(args)
