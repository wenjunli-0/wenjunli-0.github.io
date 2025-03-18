import numpy as np


# keyword pool
# v0 is the one always use, from the beginning
keyword_pool_v0 = {
        "recheck": 0, 
        "rethink": 0, 

        "reevaluate": 0, 
        "re-evaluate": 0, 
        "reevaluation": 0, 
        
        "re-examine": 0, 
        "reexamine": 0, 

        "try again": 0, 
        "check again": 0, 
        "think again": 0, 
        "go over the steps": 0,
    }

# v1 is expanded upon v0
keyword_pool_v1 = {
        "recheck": 0, 
        "rethink": 0, 
        "reassess": 0,

        "reevaluate": 0, 
        "re-evaluate": 0, 
        "reevaluation": 0, 
        
        "re-examine": 0, 
        "reexamine": 0, 

        "reconsider": 0,
        "reanalyze": 0,
        "double-check": 0,

        "check again": 0, 
        "think again": 0, 
        "verify again": 0,
        "go over the steps": 0,
    }


_keywords = {
    0.1: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.6: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
}


finemath_llama3b_r1 = {
    'num_correct': [19, 22, 26, 28, 23, 33, 27, 17, 20, 19],
    'correct_format_n_eos': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'completion_vs_answer': [352, 148],   # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [0, 17, 50, 7, 10, 8, 44, 167, 50, 54],
    'keyword_count_per_response': [0, 2, 3, 3, 4, 4, 8, 15, 13, 33],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [42, 42, 31, 55, 68, 81, 87, 103, 118, 164],
}
finemath_llama3b_r1_v2 = {
    'num_correct': [147, 172, 188, 197, 204, 197, 186, 186, 160, 143],
    'correct_format_n_eos': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'completion_vs_answer_wo_template': [352, 148],     # 0: completion, 1: answer
    'completion_vs_answer_w_template': [205, 295],        # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'sr_per_response_keyword': [16, 10, 10, 9, 5, 4, 9, 14, 16, 36],
    'sr_per_response_llm': [31, 26, 26, 34, 40, 45, 53, 51, 58, 45],
    'sr_per_response_combined': [0, 0, 0, 0, 0, 0, 1, 0, 1, 2],
}
finemath_llama3b_r1_v2_keywords = {
    0.1: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 1, 'reanalyze': 0, 'double-check': 14, 'check again': 0, 'think again': 3, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 25, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 7, 'check again': 0, 'think again': 4, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 10, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 0, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 8, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 3, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 6, 'reanalyze': 0, 'double-check': 2, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.6: {'recheck': 0, 'rethink': 2, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 4, 'reanalyze': 0, 'double-check': 2, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 0, 'rethink': 13, 'reassess': 0, 'reevaluate': 4, 're-evaluate': 2, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 83, 'reanalyze': 0, 'double-check': 1, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 0, 'rethink': 2, 'reassess': 0, 'reevaluate': 4, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 3, 'reanalyze': 0, 'double-check': 6, 'check again': 0, 'think again': 2, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 7, 'rethink': 6, 'reassess': 0, 'reevaluate': 1, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 2, 'reconsider': 3, 'reanalyze': 0, 'double-check': 9, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 1, 'rethink': 14, 'reassess': 6, 'reevaluate': 1, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 2, 'reexamine': 1, 'reconsider': 4, 'reanalyze': 0, 'double-check': 4, 'check again': 1, 'think again': 3, 'verify again': 0, 'go over the steps': 1},
}



deepseek_v3_base_r1 = {
    'num_response': [400, 326, 410, 319, 424, 370, 290, 423, 442, 482],
    'num_correct': [370, 385, 377, 403, 394, 401, 392, 394, 387, 369],
    'correct_format_n_eos': [0.951, 0.951, 0.959, 0.96, 0.966, 0.969, 0.975, 0.978, 0.985, 0.986],
    'completion_vs_answer_wo_template': [450, 50],      # 0: completion, 1: answer
    'completion_vs_answer_w_template': [3, 497],          # 0: completion, 1: answer
    
    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [47, 144, 6, 9, 76, 2, 2, 22, 6, 7],
    'keyword_count_per_response': [3, 4, 6, 5, 5, 2, 2, 3, 6, 6],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [40, 30, 43, 38, 41, 33, 51, 61, 71, 80],

    # final stats
    'sr_per_response_keyword': [3, 0, 7, 2, 2, 2, 5, 5, 10, 5],
    'sr_per_response_llm': [40, 30, 43, 38, 41, 33, 53, 61, 72, 80],
    'sr_per_response_combined': [3, 0, 6, 0, 2, 0, 3, 4, 4, 2],
}
deepseek_v3_base_r1_keywords = {
    0.1: {'recheck': 1, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 1, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 1, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 3, 'reexamine': 0, 'reconsider': 17, 'reanalyze': 0, 'double-check': 1, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 1, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 5, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 11, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.6: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 2, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 19, 'reanalyze': 0, 'double-check': 1, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 0, 'rethink': 0, 'reassess': 1, 'reevaluate': 0, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 5, 'reanalyze': 0, 'double-check': 40, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 1, 'rethink': 0, 'reassess': 2, 'reevaluate': 1, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 4, 'reanalyze': 0, 'double-check': 1, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 1, 'rethink': 0, 'reassess': 0, 'reevaluate': 1, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 2, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
}




# ========== QWen Models ========== #
qwen25_math_15b = {
    'num_correct': [257, 285, 292, 319, 336, 348, 382, 379, 375, 376],
    'correct_format_n_eos': [0.55775, 0.55675, 0.56575, 0.5615, 0.5705, 0.56975, 0.5755, 0.57575, 0.5815, 0.5965],
    'completion_vs_answer_wo_template': [0, 500],       # 0: completion, 1: answer
    'completion_vs_answer_w_template': [181, 319],      # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [88, 77, 102, 177, 120, 106, 110, 88, 131, 141],
    'keyword_count_per_response': [49, 47, 61, 55, 67, 59, 67, 61, 83, 92],
    'keyword_count_per_question': [15, 17, 19, 18, 18, 21, 22, 30, 35, 39],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [236, 246, 248, 243, 291, 272, 279, 262, 310, 267],

    # final stats
    'sr_per_response_keyword': [51, 48, 64, 62, 68, 70, 71, 61, 86, 97],
    'sr_per_response_llm': [234, 244, 247, 240, 289, 269, 277, 260, 307, 265],
    'sr_per_response_combined': [18, 20, 31, 30, 39, 40, 40, 24, 47, 45],

    # reflections in correct/incorrect responses, by llm-detection
    # 'count_reflection_correct_response': [5, 2, 8, 7, 7, 11, 12, 15, 11, 19],         # v1
    # 'count_reflection_incorrect_response': [34, 34, 45, 48, 46, 48, 60, 45, 76, 87],  # v1
    'count_reflection_correct_response': [0, 0, 0, 2, 1, 1, 3, 0, 0, 1],                # v2
    'count_reflection_incorrect_response': [2, 1, 1, 1, 1, 0, 0, 1, 0, 0],              # v2
}
qwen25_math_15b_keywords = {
    0.1: {'recheck': 34, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 49, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 10, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 36, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 36, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 12, 'reanalyze': 0, 'double-check': 1, 'check again': 0, 'think again': 0, 'verify again': 3, 'go over the steps': 0},
    0.3: {'recheck': 28, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 67, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 5, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 86, 'rethink': 2, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 81, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 19, 'reanalyze': 0, 'double-check': 1, 'check again': 1, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
    0.5: {'recheck': 26, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 77, 'reevaluation': 1, 're-examine': 3, 'reexamine': 0, 'reconsider': 9, 'reanalyze': 2, 'double-check': 1, 'check again': 0, 'think again': 0, 'verify again': 3, 'go over the steps': 0} ,
    0.6: {'recheck': 44, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 50, 'reevaluation': 0, 're-examine': 4, 'reexamine': 0, 'reconsider': 3, 'reanalyze': 0, 'double-check': 2, 'check again': 2, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 34, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 68, 'reevaluation': 0, 're-examine': 4, 'reexamine': 0, 'reconsider': 10, 'reanalyze': 0, 'double-check': 1, 'check again': 2, 'think again': 0, 'verify again': 4, 'go over the steps': 0},
    0.8: {'recheck': 35, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 40, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 8, 'reanalyze': 0, 'double-check': 1, 'check again': 3, 'think again': 0, 'verify again': 3, 'go over the steps': 0},
    0.9: {'recheck': 50, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 65, 'reevaluation': 0, 're-examine': 3, 'reexamine': 0, 'reconsider': 17, 'reanalyze': 0, 'double-check': 3, 'check again': 0, 'think again': 0, 'verify again': 1, 'go over the steps': 0},
    1.0: {'recheck': 51, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 74, 'reevaluation': 0, 're-examine': 4, 'reexamine': 0, 'reconsider': 10, 'reanalyze': 0, 'double-check': 2, 'check again': 3, 'think again': 0, 'verify again': 3, 'go over the steps': 0},
}


'''
qwen25_math_15b_stats['num_correct_reflection'] = np.ceil(np.array(qwen25_math_15b_stats['num_reflections']) * np.array(qwen25_math_15b_stats['precision']))
qwen25_math_15b_stats['num_wrong_reflection'] = np.array(qwen25_math_15b_stats['num_reflections']) - np.array(qwen25_math_15b_stats['num_correct_reflection'])
qwen25_math_15b_stats['correct_w_reflection'] = np.array(qwen25_math_15b_stats['precision'])
qwen25_math_15b_stats['correct_wo_reflection'] = (np.array(qwen25_math_15b_stats['num_correct'])-qwen25_math_15b_stats['num_correct_reflection']) / (np.array([500]*10)-np.array(qwen25_math_15b_stats['num_reflections']))
print(f'qwen2.5-math-1.5b, correctness_w_reflection: {np.mean(qwen25_math_15b_stats["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(qwen25_math_15b_stats["correct_wo_reflection"])}')

num_keywords_per_response_math_15b = {
    0.1: {"correct": {0: 1558, 1: 1, 2: 0, 3: 0}, "wrong": {0: 2419, 1: 16, 2: 4, 3: 2},},
    0.2: {"correct": {0: 1545, 1: 1, 2: 0, 3: 0}, "wrong": {0: 2433, 1: 12, 2: 4, 3: 5},},
    0.3: {"correct": {0: 1513, 1: 1, 2: 1, 3: 0}, "wrong": {0: 2461, 1: 18, 2: 4, 3: 2},},
    0.4: {"correct": {0: 1506, 1: 3, 2: 1, 3: 0}, "wrong": {0: 2471, 1: 8, 2: 4, 3: 7},},
    0.5: {"correct": {0: 1536, 1: 4, 2: 0, 3: 0}, "wrong": {0: 2444, 1: 9, 2: 2, 3: 5},},
    0.6: {"correct": {0: 1519, 1: 5, 2: 1, 3: 0}, "wrong": {0: 2455, 1: 14, 2: 4, 3: 2},},
    0.7: {"correct": {0: 1468, 1: 7, 2: 0, 3: 0}, "wrong": {0: 2507, 1: 14, 2: 0, 3: 4},},
    0.8: {"correct": {0: 1434, 1: 7, 2: 0, 3: 0}, "wrong": {0: 2531, 1: 20, 2: 6, 3: 2},},
    0.9: {"correct": {0: 1339, 1: 4, 2: 0, 3: 0}, "wrong": {0: 2623, 1: 21, 2: 8, 3: 5},},
    1.0: {"correct": {0: 1297, 1: 5, 2: 1, 3: 0}, "wrong": {0: 2662, 1: 26, 2: 5, 3: 4},},
}
'''


# qwen2.5-math-7b
qwen25_math_7b = {
    'num_correct': [356, 375, 398, 407, 404, 418, 422, 426, 431, 426],
    'precision': [0.667, 0.5, 0.833, 0.75, 0.778, 0.786, 0.643, 0.8, 1.0, 0.781],
    'recall': [0.011, 0.011, 0.013, 0.022, 0.017, 0.026, 0.021, 0.038, 0.032, 0.06],
    'correct_format_n_eos': [0.8125, 0.81075, 0.826, 0.8245, 0.828, 0.83075, 0.82225, 0.8075, 0.775, 0.74375],
    'completion_vs_answer_wo_template': [0, 500],   # 0: completion, 1: answer
    'completion_vs_answer_w_template': [84, 416],   # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [23, 34, 20, 36, 30, 33, 30, 36, 29, 43],
    'keyword_count_per_response': [59, 66, 47, 64, 53, 68, 52, 52, 59, 73],
    'keyword_count_per_question': [8, 11, 9, 14, 13, 18, 16, 21, 19, 27],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v1': [56, 52, 41, 58, 54, 65, 65, 67, 83, 89],
    'keyword_count_per_response_llm_v2': [954, 972, 985, 997, 981, 982, 919, 905, 859, 741],
    'keyword_count_per_response_llm_v3': [326, 330, 349, 380, 360, 382, 338, 347, 333, 329],
    
    # final stats
    'sr_per_response_keyword': [61, 68, 50, 66, 54, 75, 58, 54, 62, 80],
    'sr_per_response_llm': [325, 329, 349, 380, 359, 381, 335, 342, 331, 328],
    'sr_per_response_combined': [37, 44, 27, 38, 33, 42, 31, 31, 30, 36],

    # reflections in correct/incorrect responses, by llm-detection
    'count_reflection_correct_response': [12, 16, 14, 22, 14, 18, 15, 17, 20, 28],
    'count_reflection_incorrect_response': [44, 36, 27, 36, 40, 47, 50, 50, 63, 61],
}
qwen25_math_7b_keywords = {
    0.1: {'recheck': 16, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 65, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 46, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
    0.2: {'recheck': 31, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 55, 'reevaluation': 0, 're-examine': 3, 'reexamine': 0, 'reconsider': 7, 'reanalyze': 0, 'double-check': 0, 'check again': 2, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
    0.3: {'recheck': 16, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 39, 'reevaluation': 0, 're-examine': 7, 'reexamine': 0, 'reconsider': 11, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
    0.4: {'recheck': 33, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 62, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 5, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
    0.5: {'recheck': 26, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 44, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 4, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 0, 'verify again': 1, 'go over the steps': 0},
    0.6: {'recheck': 28, 'rethink': 1, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 58, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 17, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
    0.7: {'recheck': 24, 'rethink': 3, 'reassess': 1, 'reevaluate': 0, 're-evaluate': 39, 'reevaluation': 1, 're-examine': 5, 'reexamine': 0, 'reconsider': 6, 'reanalyze': 0, 'double-check': 1, 'check again': 0, 'think again': 0, 'verify again': 3, 'go over the steps': 0},
    0.8: {'recheck': 33, 'rethink': 2, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 48, 'reevaluation': 0, 're-examine': 2, 'reexamine': 0, 'reconsider': 6, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 23, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 44, 'reevaluation': 0, 're-examine': 4, 'reexamine': 0, 'reconsider': 6, 'reanalyze': 0, 'double-check': 2, 'check again': 3, 'think again': 0, 'verify again': 1, 'go over the steps': 0},
    1.0: {'recheck': 36, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 59, 'reevaluation': 0, 're-examine': 3, 'reexamine': 0, 'reconsider': 10, 'reanalyze': 0, 'double-check': 6, 'check again': 3, 'think again': 0, 'verify again': 2, 'go over the steps': 0},
}
'''
qwen25_math_7b_stats['num_correct_reflection'] = np.ceil(np.array(qwen25_math_7b_stats['num_reflections']) * np.array(qwen25_math_7b_stats['precision']))
qwen25_math_7b_stats['num_wrong_reflection'] = np.array(qwen25_math_7b_stats['num_reflections']) - np.array(qwen25_math_7b_stats['num_correct_reflection'])
qwen25_math_7b_stats['correct_w_reflection'] = np.array(qwen25_math_7b_stats['precision'])
qwen25_math_7b_stats['correct_wo_reflection'] = (np.array(qwen25_math_7b_stats['num_correct'])-qwen25_math_7b_stats['num_correct_reflection']) / (np.array([500]*10)-np.array(qwen25_math_7b_stats['num_reflections']))
print(f'qwen2.5-math-15b, correctness_w_reflection: {np.mean(qwen25_math_7b_stats["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(qwen25_math_7b_stats["correct_wo_reflection"])}')
'''

num_keywords_per_response_math_7b = {
    0.1: {"correct": {0: 2465, 1: 7, 2: 5, 3: 0}, "wrong": {0: 1517, 1: 6, 2: 0, 3: 0},},
    0.2: {"correct": {0: 2412, 1: 7, 2: 3, 3: 1}, "wrong": {0: 1565, 1: 8, 2: 3, 3: 1},},
    0.3: {"correct": {0: 2430, 1: 6, 2: 3, 3: 0}, "wrong": {0: 1554, 1: 6, 2: 1, 3: 0},},
    0.4: {"correct": {0: 2448, 1: 9, 2: 2, 3: 0}, "wrong": {0: 1529, 1: 7, 2: 1, 3: 4},},
    0.5: {"correct": {0: 2420, 1: 5, 2: 0, 3: 1}, "wrong": {0: 1562, 1: 9, 2: 3, 3: 0},},
    0.6: {"correct": {0: 2404, 1: 11, 2: 1, 3: 0}, "wrong": {0: 1570, 1: 11, 2: 2, 3: 1},},
    0.7: {"correct": {0: 2373, 1: 3, 2: 1, 3: 0}, "wrong": {0: 1610, 1: 7, 2: 4, 3: 2},},
    0.8: {"correct": {0: 2248, 1: 8, 2: 0, 3: 0}, "wrong": {0: 1729, 1: 10, 2: 2, 3: 3},},
    0.9: {"correct": {0: 2127, 1: 7, 2: 1, 3: 0}, "wrong": {0: 1852, 1: 9, 2: 3, 3: 1},},
    1.0: {"correct": {0: 1996, 1: 9, 2: 0, 3: 2}, "wrong": {0: 1976, 1: 11, 2: 3, 3: 3},},
}
keywords_count_math_7b = {
    0.1: {'recheck': 16, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 65, 'reevaluation': 0, 're-examine': 0, 'try again': 6, 'check again': 0, 'think again': 0, 'wait': 0},
    0.2: {'recheck': 31, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 55, 'reevaluation': 0, 're-examine': 3, 'try again': 3, 'check again': 2, 'think again': 0, 'wait': 0},
    0.3: {'recheck': 16, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 39, 'reevaluation': 0, 're-examine': 7, 'try again': 4, 'check again': 0, 'think again': 0, 'wait': 0},
    0.4: {'recheck': 33, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 62, 'reevaluation': 0, 're-examine': 0, 'try again': 3, 'check again': 0, 'think again': 0, 'wait': 0},
    0.5: {'recheck': 26, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 44, 'reevaluation': 0, 're-examine': 1, 'try again': 4, 'check again': 1, 'think again': 0, 'wait': 1},
    0.6: {'recheck': 28, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 58, 'reevaluation': 0, 're-examine': 0, 'try again': 4, 'check again': 1, 'think again': 0, 'wait': 0},
    0.7: {'recheck': 24, 'rethink': 3, 'reevaluate': 0, 're-evaluate': 39, 'reevaluation': 1, 're-examine': 5, 'try again': 3, 'check again': 0, 'think again': 0, 'wait': 6},
    0.8: {'recheck': 33, 'rethink': 2, 'reevaluate': 0, 're-evaluate': 48, 'reevaluation': 0, 're-examine': 2, 'try again': 1, 'check again': 0, 'think again': 0, 'wait': 0},
    0.9: {'recheck': 23, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 44, 'reevaluation': 0, 're-examine': 4, 'try again': 6, 'check again': 3, 'think again': 0, 'wait': 3},
    1.0: {'recheck': 36, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 59, 'reevaluation': 0, 're-examine': 3, 'try again': 7, 'check again': 3, 'think again': 0, 'wait': 2}
}
dicts = [
    {1: 0, 2: 1, 3: 2, 4: 1, 5: 8},
    {1: 0, 2: 1, 3: 6, 4: 2, 5: 11},
    {1: 0, 2: 0, 3: 4, 4: 2, 5: 5},
    {1: 0, 2: 1, 3: 5, 4: 6, 5: 8},
    {1: 2, 2: 1, 3: 2, 4: 3, 5: 6},
    {1: 3, 2: 0, 3: 5, 4: 3, 5: 5},
    {1: 0, 2: 0, 3: 2, 4: 4, 5: 9},
    {1: 2, 2: 5, 3: 4, 4: 4, 5: 7},
    {1: 4, 2: 2, 3: 4, 4: 2, 5: 4},
    {1: 6, 2: 6, 3: 8, 4: 5, 5: 9},
]
reflections_per_level_math_7b = {k: sum(d[k] for d in dicts) for k in dicts[0].keys()}


# qwen2.5-7b
qwen25_7b = {
    'num_correct': [379, 405, 395, 404, 412, 421, 412, 411, 410, 402],
    'precision': [0.5, 0.8, 0.8, 0.714, 0.875, 0.875, 1.0, 0.857, 0.643, 0.857],
    'recall': [0.003, 0.01, 0.01, 0.012, 0.017, 0.033, 0.012, 0.015, 0.022, 0.029],
    'correct_format_n_eos': [0.821, 0.8255, 0.82875, 0.80925, 0.815, 0.797, 0.783, 0.7815, 0.758, 0.741],
    'completion_vs_answer_wo_template': [0, 500],       # 0: completion, 1: answer
    'completion_vs_answer_w_template': [4, 496],        # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [103, 99, 64, 145, 156, 124, 127, 63, 126, 64],
    'keyword_count_per_response': [54, 41, 52, 48, 58, 44, 51, 37, 56, 43],
    'keyword_count_per_question': [2, 5, 4, 8, 9, 5, 6, 7, 14, 14],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [44, 44, 44, 50, 49, 50, 75, 72, 77, 96],
    
    # final stats
    'sr_per_response_keyword': [56, 45, 54, 50, 58, 46, 51, 44, 63, 46],
    'sr_per_response_llm': [44, 44, 43, 50, 49, 49, 75, 71, 76, 96],
    'sr_per_response_combined': [20, 19, 18, 20, 19, 16, 24, 18, 22, 23],

    # reflections in correct/incorrect responses, by llm-detection
    'count_reflection_correct_response_v1': [3, 3, 2, 3, 6, 1, 1, 6, 4, 15],
    'count_reflection_incorrect_response_v1': [28, 15, 16, 22, 24, 21, 36, 29, 33, 30],
}
qwen25_7b_keywords = {
    0.1: {'recheck': 5, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 72, 'reevaluation': 0, 're-examine': 7, 'reexamine': 0, 'reconsider': 2, 'reanalyze': 0, 'double-check': 1, 'check again': 19, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 8, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 79, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 3, 'reanalyze': 0, 'double-check': 1, 'check again': 12, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 9, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 40, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 1, 'reanalyze': 0, 'double-check': 1, 'check again': 14, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 15, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 115, 'reevaluation': 0, 're-examine': 4, 'reexamine': 0, 'reconsider': 7, 'reanalyze': 0, 'double-check': 0, 'check again': 9, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 39, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 108, 'reevaluation': 0, 're-examine': 2, 'reexamine': 0, 'reconsider': 9, 'reanalyze': 0, 'double-check': 1, 'check again': 6, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.6: {'recheck': 5, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 115, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 2, 'reanalyze': 0, 'double-check': 1, 'check again': 2, 'think again': 0, 'verify again': 1, 'go over the steps': 0},
    0.7: {'recheck': 32, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 90, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 3, 'reanalyze': 0, 'double-check': 0, 'check again': 4, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 13, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 48, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 6, 'reanalyze': 0, 'double-check': 3, 'check again': 2, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 14, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 100, 'reevaluation': 0, 're-examine': 7, 'reexamine': 0, 'reconsider': 11, 'reanalyze': 0, 'double-check': 2, 'check again': 4, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 16, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 41, 'reevaluation': 1, 're-examine': 1, 'reexamine': 0, 'reconsider': 6, 'reanalyze': 0, 'double-check': 2, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
}
'''
qwen25_7b_stats['num_correct_reflection'] = np.ceil(np.array(qwen25_7b_stats['num_reflections']) * np.array(qwen25_7b_stats['precision']))
qwen25_7b_stats['num_wrong_reflection'] = np.array(qwen25_7b_stats['num_reflections']) - np.array(qwen25_7b_stats['num_correct_reflection'])
qwen25_7b_stats['correct_w_reflection'] = np.array(qwen25_7b_stats['precision'])
qwen25_7b_stats['correct_wo_reflection'] = (np.array(qwen25_7b_stats['num_correct'])-qwen25_7b_stats['num_correct_reflection']) / (np.array([500]*10)-np.array(qwen25_7b_stats['num_reflections']))
print(f'qwen2.5-7b, correctness_w_reflection: {np.mean(qwen25_7b_stats["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(qwen25_7b_stats["correct_wo_reflection"])}')
'''

num_keywords_per_response_7b = {
    0.1: {"correct": {0: 2261, 1: 0, 2: 1, 3: 0}, "wrong": {0: 1735, 1: 3, 2: 0, 3: 0},},
    0.2: {"correct": {0: 2231, 1: 2, 2: 0, 3: 0}, "wrong": {0: 1764, 1: 2, 2: 0, 3: 1},},
    0.3: {"correct": {0: 2194, 1: 0, 2: 0, 3: 0}, "wrong": {0: 1801, 1: 3, 2: 0, 3: 2},},
    0.4: {"correct": {0: 2140, 1: 1, 2: 0, 3: 0}, "wrong": {0: 1851, 1: 5, 2: 1, 3: 2},},
    0.5: {"correct": {0: 2127, 1: 2, 2: 0, 3: 0}, "wrong": {0: 1862, 1: 5, 2: 1, 3: 3},},
    0.6: {"correct": {0: 2082, 1: 0, 2: 0, 3: 0}, "wrong": {0: 1913, 1: 4, 2: 1, 3: 0},},
    0.7: {"correct": {0: 1927, 1: 1, 2: 0, 3: 0}, "wrong": {0: 2066, 1: 3, 2: 0, 3: 3},},
    0.8: {"correct": {0: 1931, 1: 0, 2: 0, 3: 0}, "wrong": {0: 2062, 1: 5, 2: 1, 3: 1},},
    0.9: {"correct": {0: 1847, 1: 0, 2: 0, 3: 0}, "wrong": {0: 2138, 1: 15, 2: 0, 3: 0},},
    1.0: {"correct": {0: 1701, 1: 6, 2: 0, 3: 1}, "wrong": {0: 2285, 1: 5, 2: 1, 3: 1},},
}





# ========== DeepSeek Models ========== #
deepseek_math_7b_base_qwen = {
    'num_correct': [67, 66, 75, 98, 100, 103, 104, 119, 117, 105],
    'num_reflections': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'keyword_count_per_word': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'correct_format_n_eos': [0.14275, 0.1395, 0.138, 0.14275, 0.1415, 0.13775, 0.14, 0.167, 0.173, 0.177],
}
deepseek_math_7b_base_r1_v2 = {
    'num_correct': [48, 54, 70, 85, 110, 116, 142, 143, 145, 120],
    'correct_format_n_eos': [0.454, 0.457, 0.465, 0.463, 0.514, 0.517, 0.576, 0.622, 0.696, 0.75],
    'completion_vs_answer_wo_template': [151, 349],     # 0: completion, 1: answer
    'completion_vs_answer_w_template': [71, 429],       # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [0, 0, 1, 1, 5, 9, 0, 22, 9, 32],
    'keyword_count_per_response': [0, 0, 1, 1, 2, 2, 0, 7, 4, 7],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [12, 14, 14, 23, 30, 41, 56, 68, 80, 131],

    # final stats
    'sr_per_response_keyword': [0, 0, 1, 1, 2, 0, 0, 2, 1, 3],
    'sr_per_response_llm': [12, 14, 14, 24, 30, 41, 56, 68, 80, 131],
    'sr_per_response_combined': [0, 0, 0, 1, 0, 0, 0, 1, 0, 2],
}
deepseek_math_7b_base_r1_v2_keywords = {
    0.1: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 2},
    0.6: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 6, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 2, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 1, 'verify again': 0, 'go over the steps': 0},
}

'''
deepseek_math_7b_base_r1['num_correct_reflection'] = np.ceil(np.array(deepseek_math_7b_base_r1['num_reflections']) * np.array(deepseek_math_7b_base_r1['precision']))
deepseek_math_7b_base_r1['num_wrong_reflection'] = np.array(deepseek_math_7b_base_r1['num_reflections']) - np.array(deepseek_math_7b_base_r1['num_correct_reflection'])
deepseek_math_7b_base_r1['correct_w_reflection'] = np.array(deepseek_math_7b_base_r1['precision'])
deepseek_math_7b_base_r1['correct_wo_reflection'] = (np.array(deepseek_math_7b_base_r1['num_correct'])-deepseek_math_7b_base_r1['num_correct_reflection']) / (np.array([500]*10)-np.array(deepseek_math_7b_base_r1['num_reflections']))
print(f'deepseek-math-7b-base-r1, correctness_w_reflection: {np.mean(deepseek_math_7b_base_r1["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(deepseek_math_7b_base_r1["correct_wo_reflection"])}')
'''

num_keywords_per_response_7b_base_r1 = {
    0.1: {"correct": {0: 295, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3705, 1: 0, 2: 0, 3: 0},},
    0.2: {"correct": {0: 295, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3705, 1: 0, 2: 0, 3: 0},},
    0.3: {"correct": {0: 323, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3677, 1: 0, 2: 0, 3: 0},},
    0.4: {"correct": {0: 317, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3682, 1: 0, 2: 0, 3: 1},},
    0.5: {"correct": {0: 283, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3717, 1: 0, 2: 0, 3: 0},},
    0.6: {"correct": {0: 215, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3782, 1: 1, 2: 0, 3: 2},},
    0.7: {"correct": {0: 214, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3786, 1: 0, 2: 0, 3: 0},},
    0.8: {"correct": {0: 210, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3783, 1: 5, 2: 1, 3: 1},},
    0.9: {"correct": {0: 167, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3829, 1: 3, 2: 1, 3: 0},},
    1.0: {"correct": {0: 137, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3846, 1: 12, 2: 1, 3: 4},},
}



# ========== Rho Models ========== #
rho_math_7b_v01_qwen = {
    'num_correct': [180, 202, 216, 220, 228, 230, 210, 209, 214, 209],
    'precision': [0.333, 0, 0.4, 0.143, 0, 0.5, 0.5, 0.143, 0.353, 0.556],
    'recall': [0.006, 0, 0.009, 0.005, 0, 0.009, 0.019, 0.005, 0.028, 0.024],
    'correct_format_n_eos': [0.1555, 0.15325, 0.17425, 0.16975, 0.19075, 0.223, 0.23225, 0.263, 0.3095, 0.3385],
    'completion_vs_answer_wo_template': [50, 450],                  # 0: completion, 1: answer
    'completion_vs_answer_w_template': [131, 369],      # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [19, 27, 20, 17, 6, 53, 86, 76, 60, 32],
    'keyword_count_per_response': [6, 5, 5, 8, 5, 4, 8, 7, 17, 12],
    'keyword_count_per_question': [3, 3, 5, 7, 5, 4, 8, 7, 17, 9],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [92, 91, 114, 92, 115, 124, 151, 191, 275, 283],

    # final stats
    'sr_per_response_keyword': [2, 3, 5, 5, 4, 7, 8, 9, 8, 5],
    'sr_per_response_llm': [89, 90, 105, 88, 105, 116, 133, 172, 252, 267],
    'sr_per_response_combined': [0, 0, 1, 0, 0, 2, 3, 3, 4, 3],

    # reflections in correct/incorrect responses, by llm-detection
    'count_reflection_correct_response_v1': [8, 4, 3, 7, 8, 4, 9, 5, 8, 10],
    'count_reflection_incorrect_response_v1': [40, 58, 50, 53, 69, 62, 88, 106, 120, 137],
}
rho_math_7b_v01_qwen_keywords = {
    0.1: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 2, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 4, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 6, 'check again': 2, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 1, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 3, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 3, 'check again': 3, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.6: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 7, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 11, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 13, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 0, 'rethink': 10, 'reassess': 0, 'reevaluate': 7, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 3, 'reanalyze': 0, 'double-check': 10, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 1, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 5, 'check again': 1, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 3, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 1, 'reanalyze': 0, 'double-check': 1, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
}
rho_math_7b_v01_r1_v2 = {
    'num_correct': [81, 98, 121, 137, 142, 149, 165, 164, 154, 140],
    'keyword_count_per_word': [4, 0, 6, 6, 9, 3, 8, 9, 15, 21],
    'correct_format_n_eos': [0.464, 0.448, 0.441, 0.425, 0.43, 0.461, 0.507, 0.578, 0.645, 0.687],
}
rho_math_7b_v01_gist = {
    'num_correct': [15, 24, 27, 34, 45, 55, 47, 58, 55, 47],
    'num_reflections': [10, 8, 4, 4, 4, 6, 10, 10, 10, 21],
    'precision': [0, 0, 0.25, 0, 0, 0, 0.1, 0, 0, 0.048],
    'recall': [0, 0, 0.037, 0, 0, 0, 0.021, 0, 0, 0.021],
    'correct_format_n_eos': [0.02, 0.02475, 0.03375, 0.04925, 0.06475, 0.09125, 0.1055, 0.12475, 0.13575, 0.135],
}

'''
rho_math_7b_v01_stats_qwen['num_correct_reflection'] = np.ceil(np.array(rho_math_7b_v01_stats_qwen['num_reflections']) * np.array(rho_math_7b_v01_stats_qwen['precision']))
rho_math_7b_v01_stats_qwen['num_wrong_reflection'] = np.array(rho_math_7b_v01_stats_qwen['num_reflections']) - np.array(rho_math_7b_v01_stats_qwen['num_correct_reflection'])
rho_math_7b_v01_stats_qwen['correct_w_reflection'] = np.array(rho_math_7b_v01_stats_qwen['precision'])
rho_math_7b_v01_stats_qwen['correct_wo_reflection'] = (np.array(rho_math_7b_v01_stats_qwen['num_correct'])-rho_math_7b_v01_stats_qwen['num_correct_reflection']) / (np.array([500]*10)-np.array(rho_math_7b_v01_stats_qwen['num_reflections']))
print(f'rho-math-7b-v0.1, correctness_w_reflection: {np.mean(rho_math_7b_v01_stats_qwen["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(rho_math_7b_v01_stats_qwen["correct_wo_reflection"])}')
'''


num_keywords_per_response_rho_7b = {
    0.1: {"correct": {0: 620, 1: 1, 2: 0, 3: 2}, "wrong": {0: 3374, 1: 1, 2: 0, 3: 2},},
    0.2: {"correct": {0: 614, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3381, 1: 1, 2: 0, 3: 4},},
    0.3: {"correct": {0: 583, 1: 1, 2: 0, 3: 0}, "wrong": {0: 3412, 1: 1, 2: 2, 3: 1},},
    0.4: {"correct": {0: 586, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3406, 1: 6, 2: 1, 3: 1},},
    0.5: {"correct": {0: 581, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3414, 1: 4, 2: 1, 3: 0},},
    0.6: {"correct": {0: 534, 1: 0, 2: 0, 3: 1}, "wrong": {0: 3462, 1: 2, 2: 0, 3: 1},},
    0.7: {"correct": {0: 465, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3527, 1: 5, 2: 0, 3: 3},},
    0.8: {"correct": {0: 423, 1: 0, 2: 0, 3: 0}, "wrong": {0: 3570, 1: 2, 2: 1, 3: 4},},
    0.9: {"correct": {0: 404, 1: 0, 2: 0, 3: 1}, "wrong": {0: 3579, 1: 12, 2: 0, 3: 4},},
    1.0: {"correct": {0: 419, 1: 1, 2: 0, 3: 1}, "wrong": {0: 3572, 1: 4, 2: 0, 3: 3},},
}




# ========== Llama Models ========== #
# llama-3.2-1b
llama_32_1b_stats_qwen = {
    'num_correct': [0, 0, 0, 0, 0, 2, 0, 2, 3, 3],
    'num_reflections': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'precision': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'recall': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'end_with_eos_rate': [0.01125, 0.01675, 0.0245, 0.02625, 0.04575, 0.08025, 0.143, 0.2425, 0.376, 0.54975],
    'correct_format_n_eos': [0, 0, 0.001, 0, 0, 0.00425, 0, 0.01575, 0.02225, 0.041],
}
llama_32_1b_stats_inst = {
    'num_correct': [279, 296, 301, 293, 333, 330, 333, 326, 324, 312],
    'num_reflections': [17, 31, 28, 36, 21, 20, 26, 24, 28, 26],
    'precision': [0.471, 0.419, 0.429, 0.5, 0.429, 0.4, 0.462, 0.458, 0.464, 0.423],
    'recall': [0.029, 0.044, 0.04, 0.061, 0.027, 0.024, 0.036, 0.034, 0.04, 0.035],
    'end_with_eos_rate': [0.8645, 0.864, 0.8715, 0.87175, 0.87875, 0.89075, 0.90125, 0.9045, 0.91, 0.9135],
    'correct_format_n_eos': [0.567, 0.56225, 0.5685, 0.574, 0.582, 0.5795, 0.59975, 0.6065, 0.61075, 0.625],
}


# llama-3.1-8b, qwen
llama_31_8b_stats_qwen = {
    'num_correct': [5, 6, 11, 19, 17, 25, 26, 32, 26, 18],
    'num_reflections': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'keyword_count_per_word': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'correct_format_n_eos': [0.00525, 0.0095, 0.00925, 0.02375, 0.0275, 0.05125, 0.07625, 0.082, 0.092, 0.0985],
}
llama_31_8b_r1_v2 = {
    'num_correct': [116, 135, 172, 179, 184, 174, 171, 158, 137, 127],
    'correct_format_n_eos': [0.01, 0.015, 0.032, 0.054, 0.133, 0.224, 0.302, 0.345, 0.416, 0.588],
    'completion_vs_answer_wo_template': [183, 317],   # 0: completion, 1: answer
    'completion_vs_answer_w_template': [76, 424],   # 0: completion, 1: answer

    # keyword-detection, word-, response-, question-level
    'keyword_count_per_word': [0, 0, 0, 0, 0, 0, 8, 21, 12, 11],
    'keyword_count_per_response': [0, 0, 0, 0, 0, 0, 2, 4, 5, 7],
    # llm-detection, response-level
    'keyword_count_per_response_llm_v3': [31, 41, 46, 60, 61, 77, 97, 163, 151, 186],

    # final stats
    'sr_per_response_keyword': [0, 0, 0, 0, 0, 0, 0, 0, 3, 4],
    'sr_per_response_llm': [30, 41, 46, 60, 62, 77, 97, 162, 151, 186],
    'sr_per_response_combined': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
llama_31_8b_r1_v2_keywords = {
    0.1: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.4: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.6: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.7: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.8: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 0, 'think again': 0, 'verify again': 0, 'go over the steps': 0},
    0.9: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 0, 'reanalyze': 0, 'double-check': 0, 'check again': 1, 'think again': 6, 'verify again': 0, 'go over the steps': 0},
    1.0: {'recheck': 0, 'rethink': 0, 'reassess': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'reexamine': 0, 'reconsider': 2, 'reanalyze': 0, 'double-check': 1, 'check again': 0, 'think again': 2, 'verify again': 0, 'go over the steps': 0},
}



# llama-3.2-3b
llama_32_3b_stats_qwen = {
    'num_correct': [2, 8, 7, 12, 12, 7, 10, 5, 11, 6],
    'num_reflections': [0, 0, 0, 0, 0, 1, 6, 8, 17, 19],
    'precision': ['nan', 'nan', 'nan', 'nan', 'nan', 0.0, 0.167, 0.0, 0.059, 0.053],
    'recall': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.091, 0.167],
    'correct_format_n_eos': [0.00025, 0.00125, 0.00225, 0.00425, 0.01125, 0.022, 0.03475, 0.034, 0.05375, 0.06625],
}
llama_32_3b_stats_r1 = {
    'num_correct': [47, 53, 69, 67, 64, 61, 62, 55, 36, 39],
    'num_reflections': [0, 0, 0, 0, 0, 0, 1, 1, 6, 7],
    'precision': ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 0.0, 0.0, 0.0, 0.143],
    'recall': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.026],
    'correct_format_n_eos': [0.41325, 0.39375, 0.386, 0.36225, 0.3625, 0.3515, 0.34, 0.334, 0.316, 0.308],
}
llama_32_3b_stats_gist = {
    'num_correct': [37, 50, 59, 69, 66, 68, 61, 63, 47, 36],
    'num_reflections': [0, 0, 0, 3, 2, 3, 6, 2, 14, 15],
    'precision': ['nan', 'nan', 'nan', 0.333, 0.0, 0.0, 0.0, 0.5, 0.0, 0.2],
    'recall': [0.0, 0.0, 0.0, 0.014, 0.0, 0.0, 0.0, 0.016, 0.0, 0.083],
    'correct_format_n_eos': [0.18725, 0.1975, 0.20925, 0.21375, 0.22675, 0.2275, 0.23, 0.22575, 0.228, 0.21],
}





# ==================== plots ==================== #
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import matplotlib.ticker as ticker

# print(sns.color_palette("colorblind"))
hue_order = [
    "Qwen2.5-Math-7B",
    "Qwen2.5-7B",
    "Qwen2.5-Math-1.5B",
    "Llama-3.1-8B",
    "Llama-3.2-3B",
    "FineMath-Llama-3B",
    "Rho-Math-7B",
    "DeepSeek-Math-7B",
    "DeepSeek-V3-Base-685B",
]
methods_to_color = {
    "Qwen2.5-Math-7B": (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    "Qwen2.5-7B": (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    "Qwen2.5-Math-1.5B": (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    "Llama-3.1-8B": (0.8352941176470589, 0.3686274509803922, 0.0),
    "FineMath-Llama-3B": (0.9254901960784314, 0.8823529411764706, 0.2),
    "Rho-Math-7B": (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
    "DeepSeek-Math-7B": (0.8, 0.47058823529411764, 0.7372549019607844),
    "DeepSeek-V3-Base-685B": (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
}
temp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# sys.exit(0)


'''
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Keyword Detection
count_keyword = 'keyword_count_per_response'
sns.lineplot(ax=axes[0], x=temp, y=qwen25_math_15b_stats[count_keyword], label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[0], x=temp, y=qwen25_math_7b_stats[count_keyword], label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[0], x=temp, y=qwen25_7b_stats[count_keyword], label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[0], x=temp, y=rho_math_7b_v01_stats_qwen[count_keyword], label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[0], x=temp, y=llama_31_8b_stats_r1[count_keyword], label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[0], x=temp, y=deepseek_math_7b_base_r1[count_keyword], label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[0], x=temp, y=deepseek_v3_base_r1[count_keyword], label='DeepSeek-V3', marker='s', color=methods_to_color['DeepSeek-V3'])
axes[0].set_title('Keyword Detection', fontsize=14)
axes[0].set_xlabel('Temperature', fontsize=14)
axes[0].set_ylabel('Self-Reflection Count', fontsize=13)
axes[0].set_xticks(temp)
axes[0].tick_params(axis='both', labelsize=14)
axes[0].get_legend().remove()


# LLM Detection
# count_llm = 'keyword_count_per_response_llm_v1'
# count_llm = 'keyword_count_per_response_llm_v2'
count_llm = 'keyword_count_per_response_llm_v3'
sns.lineplot(ax=axes[1], x=temp, y=qwen25_math_15b_stats[count_llm], label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[1], x=temp, y=qwen25_math_7b_stats[count_llm], label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=qwen25_7b_stats[count_llm], label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[1], x=temp, y=rho_math_7b_v01_stats_qwen[count_llm], label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=llama_31_8b_stats_r1[count_llm], label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[1], x=temp, y=deepseek_math_7b_base_r1[count_llm], label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=deepseek_v3_base_r1[count_llm], label='DeepSeek-V3', marker='s', color=methods_to_color['DeepSeek-V3'])
axes[1].set_title('LLM Detection', fontsize=14)
axes[1].set_xlabel('Temperature', fontsize=14)
axes[1].set_ylabel('Self-Reflection Count', fontsize=13)
axes[1].set_xticks(temp)
axes[1].tick_params(axis='both', labelsize=14)
sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(-1.2, -0.15), ncol=4, frameon=True, fontsize=14)

plt.savefig(f'figures/keyword_vs_llm_detection_v3.png', dpi=100, bbox_inches='tight')
'''


# =========== Figure 1. model attributes =========== #
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# plt.subplots_adjust(wspace=0.25)

# 1. coqmpletion
model_names = [
    'Qwen2.5-Math-1.5B', 'Qwen2.5-Math-7B', 'Qwen2.5-7B',
    'Rho-Math-7B', 'Llama-3.1-8B', 'FineMath-Llama-3B',
    'DeepSeek-Math-7B', 'DeepSeek-V3-Base-685B'
]
attribute = 'completion_vs_answer'

# Completion rates (without and with prompt) — normalize by dividing by 500
answering_wo_prompt = [
    qwen25_math_15b['completion_vs_answer_wo_template'][1] / 500,
    qwen25_math_7b['completion_vs_answer_wo_template'][1] / 500,
    qwen25_7b['completion_vs_answer_wo_template'][1] / 500,
    rho_math_7b_v01_qwen['completion_vs_answer_wo_template'][1] / 500,
    llama_31_8b_r1_v2['completion_vs_answer_wo_template'][1] / 500,
    finemath_llama3b_r1_v2['completion_vs_answer_wo_template'][1] / 500,
    deepseek_math_7b_base_r1_v2['completion_vs_answer_wo_template'][1] / 500,
    deepseek_v3_base_r1['completion_vs_answer_wo_template'][1] / 500
]
answering_w_prompt = [
    qwen25_math_15b['completion_vs_answer_w_template'][1] / 500,
    qwen25_math_7b['completion_vs_answer_w_template'][1] / 500,
    qwen25_7b['completion_vs_answer_w_template'][1] / 500,
    rho_math_7b_v01_qwen['completion_vs_answer_w_template'][1] / 500,
    llama_31_8b_r1_v2['completion_vs_answer_w_template'][1] / 500,
    finemath_llama3b_r1_v2['completion_vs_answer_w_template'][1] / 500,
    deepseek_math_7b_base_r1_v2['completion_vs_answer_w_template'][1] / 500,
    deepseek_v3_base_r1['completion_vs_answer_w_template'][1] / 500
]

x = np.arange(len(model_names))
bar_width = 0.35
axes[0].bar(x - bar_width/2, answering_wo_prompt, width=bar_width, label='w/o template', color='steelblue')
axes[0].bar(x + bar_width/2, answering_w_prompt, width=bar_width, label='w/ template', color='orange')

# axes[0].bar(x, answering_wo_prompt, width=bar_width, label='w/o Prompt', color='steelblue', edgecolor='black')
# axes[0].bar(x, answering_w_prompt - answering_wo_prompt, bottom=answering_wo_prompt, width=bar_width, label='w/ Prompt', color='orange', edgecolor='black', hatch='//')

axes[0].set_xticks(x)
axes[0].set_ylabel('Answering Rate', fontsize=14)
axes[0].set_title('Answering Tendency', fontsize=14)
axes[0].legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.675, 1.0), ncol=1)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].set_ylim([0, 1.05])
axes[0].tick_params(axis='y', labelsize=14)
axes[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=14)


# 1. accuracy
attribute = 'num_correct'
sns.lineplot(ax=axes[1], x=temp, y=np.array(qwen25_math_15b[attribute])/500, label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(qwen25_math_7b[attribute])/500, label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(qwen25_7b[attribute])/500, label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(rho_math_7b_v01_qwen[attribute])/500, label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
# sns.lineplot(ax=axes[1], x=temp, y=np.array(llama_31_8b_r1[attribute])/500, label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(llama_31_8b_r1_v2[attribute])/500, label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
# sns.lineplot(ax=axes[1], x=temp, y=np.array(finemath_llama3b_r1[attribute])/500, label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(finemath_llama3b_r1_v2[attribute])/500, label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
# sns.lineplot(ax=axes[1], x=temp, y=np.array(deepseek_math_7b_base_r1[attribute])/500, label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(deepseek_math_7b_base_r1_v2[attribute])/500, label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=np.array(deepseek_v3_base_r1[attribute])/500, label='DeepSeek-V3-Base-685B', marker='s', color=methods_to_color['DeepSeek-V3-Base-685B'])
axes[1].set_title('Pass@8', fontsize=14)
axes[1].set_xlabel('Temperature', fontsize=14)
axes[1].set_ylabel('Pass Rate', fontsize=14)
axes[1].set_xticks(temp)
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='both', labelsize=14)
axes[1].get_legend().remove()

# 3. self-reflection

qwen25_math_15b_sr_temps = []
qwen25_math_7b_sr_temps = []
qwen25_7b_sr_temps = []
for tmp, keywords in qwen25_math_15b_keywords.items():
    qwen25_math_15b_sr_temps.append(sum(keywords.values()))
for tmp, keywords in qwen25_math_7b_keywords.items():
    qwen25_math_7b_sr_temps.append(sum(keywords.values()))
for tmp, keywords in qwen25_7b_keywords.items():
    qwen25_7b_sr_temps.append(sum(keywords.values()))

llama_31_8b_r1_v2_sr_temps = []
finemath_llama3b_r1_v2_sr_temps = []
for tmp, keywords in llama_31_8b_r1_v2_keywords.items():
    llama_31_8b_r1_v2_sr_temps.append(sum(keywords.values()))
for tmp, keywords in finemath_llama3b_r1_v2_keywords.items():
    finemath_llama3b_r1_v2_sr_temps.append(sum(keywords.values()))

rho_math_7b_v01_qwen_temps = []
deepseek_math_7b_base_r1_v2_sr_temps = []
deepseek_v3_base_r1_sr_temps = []
for tmp, keywords in rho_math_7b_v01_qwen_keywords.items():
    rho_math_7b_v01_qwen_temps.append(sum(keywords.values()))
for tmp, keywords in deepseek_math_7b_base_r1_v2_keywords.items():
    deepseek_math_7b_base_r1_v2_sr_temps.append(sum(keywords.values()))
for tmp, keywords in deepseek_v3_base_r1_keywords.items():
    deepseek_v3_base_r1_sr_temps.append(sum(keywords.values()))

''' # keyword-level count
sns.lineplot(ax=axes[2], x=temp, y=qwen25_math_15b_sr_temps, label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[2], x=temp, y=qwen25_math_7b_sr_temps, label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=qwen25_7b_sr_temps, label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[2], x=temp, y=rho_math_7b_v01_qwen_temps, label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=llama_31_8b_r1_v2_sr_temps, label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[2], x=temp, y=finemath_llama3b_r1_v2_sr_temps, label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
sns.lineplot(ax=axes[2], x=temp, y=deepseek_math_7b_base_r1_v2_sr_temps, label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=deepseek_v3_base_r1_sr_temps, label='DeepSeek-V3-Base-685B', marker='s', color=methods_to_color['DeepSeek-V3-Base-685B'])
# axes[2].set_yscale('log')
# axes[2].set_yticks([1, 10, 100, 1000])
# axes[2].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}" if x == 1 else f"$10^{{{int(round(np.log10(x)))}}}$"))
'''

attribute = 'sr_per_response_combined'
sns.lineplot(ax=axes[2], x=temp, y=qwen25_math_15b[attribute], label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[2], x=temp, y=qwen25_math_7b[attribute], label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=qwen25_7b[attribute], label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[2], x=temp, y=rho_math_7b_v01_qwen[attribute], label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=llama_31_8b_r1_v2[attribute], label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[2], x=temp, y=finemath_llama3b_r1_v2[attribute], label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
sns.lineplot(ax=axes[2], x=temp, y=deepseek_math_7b_base_r1_v2[attribute], label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=deepseek_v3_base_r1[attribute], label='DeepSeek-V3-Base-685B', marker='s', color=methods_to_color['DeepSeek-V3-Base-685B'])

axes[2].set_title('Self-Reflection', fontsize=14)
axes[2].set_xlabel('Temperature', fontsize=14)
axes[2].set_ylabel('Count', fontsize=14)
axes[2].set_xticks(temp)
axes[2].tick_params(axis='both', labelsize=14)
sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(-1.35, -0.15), ncol=3, frameon=True, fontsize=13.5)

plt.savefig(f'figures/model_attributes.png', dpi=300, bbox_inches='tight')



# =========== Figure 2. keyword pool and distribution =========== #
models = {
    "Qwen2.5-Math-1.5B": qwen25_math_15b_keywords,
    "Qwen2.5-Math-7B": qwen25_math_7b_keywords,
    "Qwen2.5-7B": qwen25_7b_keywords,
    "Rho-Math-7B": rho_math_7b_v01_qwen_keywords,
    "Llama-3.1-8B": llama_31_8b_r1_v2_keywords,
    "FineMath-Llama-3B": finemath_llama3b_r1_v2_keywords,
    "Deepseek-Math-7B": deepseek_math_7b_base_r1_v2_keywords,
    "DeepSeek-V3-Base-685B": deepseek_v3_base_r1_keywords,
}
# Get all unique keywords
keywords = set()
for model_data in models.values():
    for temp_data in model_data.values():
        keywords.update(temp_data.keys())
keywords = sorted(keywords)

# Sum occurrences over temperatures for each model
model_sums = {model: {kw: 0 for kw in keywords} for model in models}
for model, model_data in models.items():
    for temp_data in model_data.values():
        for kw, count in temp_data.items():
            model_sums[model][kw] += count

# Plot the histogram
x = np.arange(len(keywords))
width = 0.8 / len(models)  # Adjust the bar width based on number of models
fig, ax = plt.subplots(figsize=(13, 8))

# Add dashed vertical lines between keyword groups
for i in range(len(keywords) - 1):
    ax.axvline(x=i + 0.5, color="gray", linestyle="dashed", linewidth=0.7)

# Plot bars for each model with adjusted width
for i, (model, kw_counts) in enumerate(model_sums.items()):
    ax.bar(x + i * width - (width * (len(models) // 2)),  # Center-align bars
           [kw_counts[kw] for kw in keywords], 
           width, 
           label=model)

# Set log scale for the y-axis
ax.set_yscale("log")  
ax.set_ylim(0, 1000)  # Set the limit to avoid issues with log(0)
ax.set_xticks(x)
ax.tick_params(axis='y', labelsize=14)
ax.set_xticklabels(keywords, rotation=45, ha="right", fontsize=14)
ax.set_xlabel("Keywords", fontsize=14)
ax.set_xlim(-0.5, len(keywords) - 0.5)
ax.set_ylabel("Count of self-reflection keywords", fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.82, 1.0), ncol=1, fontsize=14)
plt.savefig("figures/keyword_pool.png", dpi=300, bbox_inches='tight')




# =========== Figure 3. keyword detection vs llm detection =========== #
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Plot the keyword detection counts for each model
# attribute = 'keyword_count_per_response'
attribute = 'sr_per_response_keyword'
sns.lineplot(ax=axes[0], x=temp, y=qwen25_math_15b_sr_temps, label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[0], x=temp, y=qwen25_math_7b_sr_temps, label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[0], x=temp, y=qwen25_7b_sr_temps, label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[0], x=temp, y=rho_math_7b_v01_qwen_temps, label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[0], x=temp, y=llama_31_8b_r1_v2_sr_temps, label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[0], x=temp, y=finemath_llama3b_r1_v2_sr_temps, label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
sns.lineplot(ax=axes[0], x=temp, y=deepseek_math_7b_base_r1_v2_sr_temps, label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[0], x=temp, y=deepseek_v3_base_r1_sr_temps, label='DeepSeek-V3-Base-685B', marker='s', color=methods_to_color['DeepSeek-V3-Base-685B'])
axes[0].legend().set_visible(False)

# Plot the LLM detection counts for each model
# attribute = 'keyword_count_per_response_llm_v3'
attribute = 'sr_per_response_llm'
sns.lineplot(ax=axes[1], x=temp, y=qwen25_math_15b[attribute], label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[1], x=temp, y=qwen25_math_7b[attribute], label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=qwen25_7b[attribute], label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[1], x=temp, y=rho_math_7b_v01_qwen[attribute], label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=llama_31_8b_r1_v2[attribute], label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[1], x=temp, y=finemath_llama3b_r1_v2[attribute], label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
sns.lineplot(ax=axes[1], x=temp, y=deepseek_math_7b_base_r1_v2[attribute], label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[1], x=temp, y=deepseek_v3_base_r1[attribute], label='DeepSeek-V3-Base-685B', marker='s', color=methods_to_color['DeepSeek-V3-Base-685B'])
axes[1].legend().set_visible(False)

# Plot the Cross detection counts for each model
attribute = 'sr_per_response_combined'
sns.lineplot(ax=axes[2], x=temp, y=qwen25_math_15b[attribute], label='Qwen2.5-Math-1.5B', marker='o', color=methods_to_color['Qwen2.5-Math-1.5B'])
sns.lineplot(ax=axes[2], x=temp, y=qwen25_math_7b[attribute], label='Qwen2.5-Math-7B', marker='o', color=methods_to_color['Qwen2.5-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=qwen25_7b[attribute], label='Qwen2.5-7B', marker='o', color=methods_to_color['Qwen2.5-7B'])
sns.lineplot(ax=axes[2], x=temp, y=rho_math_7b_v01_qwen[attribute], label='Rho-Math-7B', marker='d', color=methods_to_color['Rho-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=llama_31_8b_r1_v2[attribute], label='Llama-3.1-8B', marker='^', color=methods_to_color['Llama-3.1-8B'])
sns.lineplot(ax=axes[2], x=temp, y=finemath_llama3b_r1_v2[attribute], label='FineMath-Llama-3B', marker='^', color=methods_to_color['FineMath-Llama-3B'])
sns.lineplot(ax=axes[2], x=temp, y=deepseek_math_7b_base_r1_v2[attribute], label='DeepSeek-Math-7B', marker='s', color=methods_to_color['DeepSeek-Math-7B'])
sns.lineplot(ax=axes[2], x=temp, y=deepseek_v3_base_r1[attribute], label='DeepSeek-V3-Base-685B', marker='s', color=methods_to_color['DeepSeek-V3-Base-685B'])

# Set axis labels and titles
axes[0].set_title("Keyword-based Detection", fontsize=16)
axes[1].set_title("LLM-based Detection", fontsize=16)
axes[2].set_title("Cross Detection", fontsize=16)

# set x-axis ticks
axes[0].set_xticks(temp)
axes[1].set_xticks(temp)
axes[2].set_xticks(temp)
axes[0].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='x', labelsize=14)
axes[2].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[2].tick_params(axis='y', labelsize=14)

# Set x and y axis labels
for ax in axes:
    ax.set_xlabel("Temperature", fontsize=14)
    ax.set_ylabel("Self-Reflection Count (per Response)", fontsize=14)

# Adjust the legend and set its size
sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(-2, -0.13), ncol=3, frameon=True, fontsize=14)
plt.savefig('figures/keyword_llm_cross.png', dpi=300, bbox_inches='tight')





