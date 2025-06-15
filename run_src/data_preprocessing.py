import jsonlines
import json

prompt_cot_medqa = "{question} A: {opa}, B: {opb}, C:{opc}, D: {opd}, E: {ope}"
prompt_cot = "{question} A: {A}, B: {B}, C:{C}, D: {D}"
prompt_stg = "{question} A: {A}, B: {B}"
prompt_sqa = "{question} A: {A}, B: {B}, C: {C}"

def load_json(filename):
    """
    Load a JSON file given a filename
    If the file doesn't exist, then return an empty dictionary instead
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def load_jsonl(filename):
    file_content = []
    try:
        with jsonlines.open(filename) as reader:
            for obj in reader:
                file_content.append(obj)
            return file_content
    except FileNotFoundError:
        return []

def write_jsonl(data, filepath):
    with open(filepath, 'w') as jsonl_file:
        for line in data:
            jsonl_file.write(json.dumps(line))
            jsonl_file.write('\n')


def write_json(data, filepath):
    with open(filepath, "w") as writer:
        json.dump(data, writer)

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def medqa():
    data = load_jsonl("/home/htran/generation/med_preferences/rStar/data/MedQA/test.jsonl")
    pro_data = []
    for idx in range(len(data)):
        question = data[idx]['question']
        options = data[idx]['options']
        answer = data[idx]['answer_idx']
        prompt = prompt_cot_medqa.format(
            **{"question": question, "opa": options['A'], "opb": options['B'], "opc": options['C'], "opd": options['D'],
               "ope": options['E']})
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer
        }
        pro_data.append(item)
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/MedQA/test_all.json")


def medqa_train():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/MedQA/train.jsonl")
    pro_data = []
    for idx in range(len(data)):
        question = data[idx]['question']
        options = data[idx]['options']
        answer = data[idx]['answer_idx']
        prompt = prompt_cot_medqa.format(
            **{"question": question, "opa": options['A'], "opb": options['B'], "opc": options['C'], "opd": options['D'],
               "ope": options['E']})
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/MedQA/train_all.json")

def medmcqa():
    data = load_json("/home/htran/generation/med_preferences/raise/data/MedMCQA/benchmark.json")["medmcqa"]
    pro_data = []
    for key in data.keys():
        question = data[key]['question']
        options = data[key]['options']
        answer = data[key]['answer']
        prompt = prompt_cot.format(
            **{"question": question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
        item = {
            "id": key,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
        import ipdb; ipdb.set_trace()
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/MedMCQA/test_all.json")

def usmle():
    data = load_json("/home/htran/generation/med_preferences/raise/data/MedMCQA/benchmark.json")["medqa"]
    pro_data = []
    for key in data.keys():
        question = data[key]['question']
        options = data[key]['options']
        answer = data[key]['answer']
        prompt = prompt_cot.format(
            **{"question": question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
        item = {
            "id": key,
            "problem": prompt,
            "question": question,
            "solution": options[answer],
            "options": options,
            "answer": answer
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/MedQA/test_all.json")

def mmlu():
    data = load_json("/home/htran/generation/med_preferences/raise/data/MedMCQA/benchmark.json")["mmlu"]
    pro_data = []
    for key in data.keys():
        question = data[key]['question']
        options = data[key]['options']
        answer = data[key]['answer']
        prompt = prompt_cot.format(
            **{"question": question, "A": options['A'], "B": options['B'], "C": options['C'], "D": options['D']})
        item = {
            "id": key,
            "problem": prompt,
            "question": question,
            "solution": options[answer],
            "options": options,
            "answer": answer
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/MMLU/test_all.json")

def icliniq():
    data = load_json("/home/htran/generation/med_preferences/raise/data/ChatDoctor/iCliniq.json")
    pro_data = []
    for idx in range(len(data)):
        input = data[idx]["input"]
        answer = data[idx]["answer_icliniq"]
        answer_chatgpt = data[idx]["answer_chatgpt"]
        answer_chatdoctor = data[idx]["answer_chatdoctor"]
        if len(input.split(" ")) <= 100 or len(answer.split(" ")) <= 100:
            continue
        item = {
            "id": idx,
            "problem": input,
            "solution": answer,
            "options": None,
            "answer": answer,
            "answer_chatgpt": answer_chatgpt,
            "answer_chatdoctor": answer_chatdoctor
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/ChatDoctor/icliniq_filtered.json")

def strategyqa():
    data = load_json("/home/htran/generation/med_preferences/raise/data/STG/test_all.json")
    pro_data = []
    options = {"A": "true", "B": "false"}
    for idx in range(len(data)):
        if data[idx]['solution'] == 'true':
            answer = "A"
        else:
            answer = "B"
        question = data[idx]['problem']
        sol = data[idx]['solution']
        prompt = prompt_stg.format(
            **{"question": question, "A": options['A'], "B": options['B']})

        item = {
            "id": idx,
            "problem": prompt,
            "solution": sol,
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/STG/test.json")


def commonsenseqa():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/CommonsenseQA/dev_rand_split.jsonl")
    pro_data = []
    for idx in range(len(data)):
        choices = data[idx]['question']['choices']
        options = {choice['label']: choice['text'] for choice in choices}
        question = data[idx]['question']['stem']
        prompt = prompt_cot_medqa.format(
            **{"question": question, "opa": options['A'], "opb": options['B'], "opc": options['C'], "opd": options['D'],
               "ope": options['E']})

        answer = data[idx]['answerKey']
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/CommonsenseQA/dev.json")


def commonsenseqa_train():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/CommonsenseQA/train_rand_split.jsonl")
    pro_data = []
    for idx in range(len(data)):
        choices = data[idx]['question']['choices']
        options = {choice['label']: choice['text'] for choice in choices}
        question = data[idx]['question']['stem']
        prompt = prompt_cot_medqa.format(
            **{"question": question, "opa": options['A'], "opb": options['B'], "opc": options['C'], "opd": options['D'],
               "ope": options['E']})

        answer = data[idx]['answerKey']
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/CommonsenseQA/train.json")

def siqa():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/SIQA/dev.jsonl")
    pro_data = []
    labels = read_txt("/home/htran/generation/med_preferences/raise/data/SIQA/dev-labels.lst")
    filtered_labels = []
    answer_dct = {"1": "A", "2": "B", "3": "C"}
    for label in labels:
        if label != "\n":
            filtered_labels.append(label)
    for idx in range(len(data)):
        options = {"A": data[idx]["answerA"], "B": data[idx]["answerB"], "C": data[idx]["answerC"]}
        question = data[idx]['context'] + " " + data[idx]["question"]
        prompt = prompt_sqa.format(
            **{"question": question, "A": options['A'], "B": options['B'], "C": options['C']})
        answer = answer_dct[filtered_labels[idx]]
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/SIQA/dev.json")

def siqa_test():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/SIQA/socialiqa.jsonl")
    pro_data = []
    filtered_labels = ["1"] * len(data)
    answer_dct = {"1": "A", "2": "B", "3": "C"}
    for idx in range(len(data)):
        options = {"A": data[idx]["answerA"], "B": data[idx]["answerB"], "C": data[idx]["answerC"]}
        question = data[idx]['context'] + " " + data[idx]["question"]
        prompt = prompt_sqa.format(
            **{"question": question, "A": options['A'], "B": options['B'], "C": options['C']})
        answer = answer_dct[filtered_labels[idx]]
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/SIQA/test.json")

def siqa_train():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/SIQA/train.jsonl")
    pro_data = []
    labels = read_txt("/home/htran/generation/med_preferences/raise/data/SIQA/socialiqa-train-dev/train-labels.lst")
    filtered_labels = []
    answer_dct = {"1": "A", "2": "B", "3": "C"}
    for label in labels:
        if label != "\n":
            filtered_labels.append(label)
    for idx in range(len(data)):
        options = {"A": data[idx]["answerA"], "B": data[idx]["answerB"], "C": data[idx]["answerC"]}
        question = data[idx]['context'] + " " + data[idx]["question"]
        prompt = prompt_sqa.format(
            **{"question": question, "A": options['A'], "B": options['B'], "C": options['C']})
        answer = answer_dct[filtered_labels[idx]]
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/SIQA/train.json")


def piqa():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/PIQA/valid.jsonl")
    pro_data = []
    labels = read_txt("/home/htran/generation/med_preferences/raise/data/PIQA/valid-labels.lst")
    filtered_labels = []
    answer_dct = {"0": "A", "1": "B"}
    for label in labels:
        if label != "\n":
            filtered_labels.append(label)
    for idx in range(len(data)):
        options = {"A": data[idx]["sol1"], "B": data[idx]["sol2"]}
        question = data[idx]['goal']
        prompt = prompt_stg.format(
            **{"question": question, "A": options['A'], "B": options['B']})
        answer = answer_dct[filtered_labels[idx]]
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/PIQA/valid.json")

def piqa_train():
    data = load_jsonl("/home/htran/generation/med_preferences/raise/data/PIQA/train.jsonl")
    pro_data = []
    labels = read_txt("/home/htran/generation/med_preferences/raise/data/PIQA/train-labels.lst")
    filtered_labels = []
    answer_dct = {"0": "A", "1": "B"}
    for label in labels:
        if label != "\n":
            filtered_labels.append(label)
    for idx in range(len(data)):
        options = {"A": data[idx]["sol1"], "B": data[idx]["sol2"]}
        question = data[idx]['goal']
        prompt = prompt_stg.format(
            **{"question": question, "A": options['A'], "B": options['B']})
        answer = answer_dct[filtered_labels[idx]]
        item = {
            "id": idx,
            "problem": prompt,
            "solution": options[answer],
            "options": options,
            "answer": answer,
            "question": question
        }
        pro_data.append(item)
    print(len(pro_data))
    write_json(pro_data, "/home/htran/generation/med_preferences/raise/data/PIQA/train.json")


if __name__ == "__main__":
    # usmle()
    # mmlu()
    # medqa()
    medmcqa()
    # medqa_train()
    # icliniq()
    # strategyqa()
    # commonsenseqa()
    # siqa()
    # piqa()
    # siqa_test()
    # commonsenseqa_train()
    # piqa_train()
    # siqa_train()
    # usmle()
    # mmlu()
