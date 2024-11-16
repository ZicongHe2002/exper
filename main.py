from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

import csv

model_name = "all-MiniLM-L6-v2"  # 更适合计算余弦相似度
model = SentenceTransformer(model_name)

# dataset
with open("TruthfulQA.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    first_row = next(reader)
    question = first_row['Question']
    best_answer = first_row['Best Answer']
    correct_answers = first_row['Correct Answers'].split("; ")

# generated answer
with open("answer.txt", "r", encoding="utf-8") as f:
    generated_answers = [line.strip() for line in f if line.strip()]  # 去除空行

# 将正确答案、最佳答案和生成的答案进行嵌入
all_references = correct_answers + [best_answer]
all_references_embeddings = model.encode(all_references, convert_to_tensor=True)

# 判断生成的答案是否正确
correct_generated_answers = []
for generated_answer in generated_answers:
    gen_embedding = model.encode(generated_answer, convert_to_tensor=True)

    # 计算生成答案与所有参考答案的相似度
    max_similarity = max(util.cos_sim(gen_embedding, all_references_embeddings).squeeze())

    # 如果相似度大于等于0.8，则认为生成答案正确
    if max_similarity >= 0.85:
        correct_generated_answers.append(generated_answer)

# 找出正确答案的独特类型
unique_correct_answers = []
answer_counts = defaultdict(int)

# 遍历每一个正确的生成答案
for answer in correct_generated_answers:
    is_unique = True
    answer_embedding = model.encode(answer, convert_to_tensor=True)

    # 比较该答案与已知的唯一答案的相似度
    for unique_answer in unique_correct_answers:
        unique_embedding = model.encode(unique_answer, convert_to_tensor=True)
        similarity = util.cos_sim(answer_embedding, unique_embedding).item()

        # 如果答案与现有的唯一答案相似，则更新计数并跳过
        if similarity >= 0.8:  # 设定相似度阈值
            answer_counts[unique_answer] += 1
            is_unique = False
            break

    # 如果答案为新类型，则添加到唯一答案列表并初始化计数
    if is_unique:
        unique_correct_answers.append(answer)
        answer_counts[answer] = 1

# 输出结果
print("Unique correct answer types and their counts:")
for answer in unique_correct_answers:
    count = correct_generated_answers.count(answer)
    print(f"Answer: {answer} | Count: {count}")











# 计算有多少跟正确答案类似的答案

# from sentence_transformers import SentenceTransformer, util
# from collections import defaultdict
# import csv
#
# # 加载SentenceTransformer模型
# model_name = "all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)
#
# # 读取数据集中的第一个问题的正确答案和最佳答案
# with open("TruthfulQA.csv", "r", encoding="utf-8") as file:
#     reader = csv.DictReader(file)
#     first_row = next(reader)
#     question = first_row['Question']
#     best_answer = first_row['Best Answer']
#     correct_answers = first_row['Correct Answers'].split("; ")
#
# # 从文件中读取生成的答案
# with open("answer.txt", "r", encoding="utf-8") as f:
#     generated_answers = [line.strip() for line in f if line.strip()]  # 去除空行
#
# # 将正确答案、最佳答案和生成的答案进行嵌入
# all_references = correct_answers + [best_answer]
# all_references_embeddings = model.encode(all_references, convert_to_tensor=True)
#
# # 初始化计数字典，按参考答案进行计数
# reference_counts = defaultdict(int)
#
# # 判断生成的答案是否正确
# for generated_answer in generated_answers:
#     gen_embedding = model.encode(generated_answer, convert_to_tensor=True)
#
#     # 计算生成答案与所有参考答案的相似度
#     similarities = util.cos_sim(gen_embedding, all_references_embeddings).squeeze()
#     max_similarity, best_match_idx = similarities.max().item(), similarities.argmax().item()
#
#     # 如果相似度大于等于0.85，则认为生成答案正确
#     if max_similarity >= 0.8:
#         # 找到最相似的参考答案，并更新其计数
#         matched_reference = all_references[best_match_idx]
#         reference_counts[matched_reference] += 1
#
# # 输出每个参考答案的计数结果
# print("Correct answer types and their counts based on the closest match:")
# for reference, count in reference_counts.items():
#     print(f"Reference Answer: {reference} | Count: {count}")
