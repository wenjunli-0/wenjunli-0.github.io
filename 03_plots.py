import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd







# get color rgbs
print(sns.color_palette("colorblind"))

hue_order = [
    "Qwen2.5-Math-7B",
    "Qwen2.5-7B",
    "Qwen2.5-Math-1.5B",
    "Llama-3.1-8B",
    "Llama-3.2-3B",
    "Deepseek-Math-7B",
    "Rho-Math-7B",
]
methods_to_color = {
    "Qwen2.5-Math-7B": (0.00392156862745098, 0.45098039215686275, 0.6980392156862745), # change this #xxxxx to rbg tuple printed from print(sns.color_palette("colorblind"))
    "Qwen2.5-7B": (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    "Qwen2.5-Math-1.5B": (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    "Llama-3.1-8B": (0.8352941176470589, 0.3686274509803922, 0.0),
    "Deepseek-Math-7B": (0.8, 0.47058823529411764, 0.7372549019607844),
    "Rho-Math-7B": (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
}


temp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]




# qwen2.5-math-1.5b
qwen25_math_15b_stats = {
    # 'num_correct': [260, 283, 302, 326, 353, 367, 382, 386, 392, 374],      # round-1
    # 'num_reflections': [9, 12, 16, 16, 16, 23, 21, 26, 23, 48],            # round-1
    # 'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # round-1
    'num_correct': [257, 285, 292, 319, 336, 348, 382, 379, 375, 376],      # round-2
    'num_reflections': [15, 17, 19, 18, 18, 21, 22, 30, 35, 39],            # round-2
    'num_total_keywords': [38, 41, 35, 94, 39, 50, 36, 45, 63, 60],         # round-2
    'precision': [0.778, 1.0, 0.75, 0.625, 0.75, 0.783, 0.81, 0.808, 0.739, 0.667],
    'recall': [0.027, 0.042, 0.04, 0.031, 0.034, 0.049, 0.045, 0.054, 0.043, 0.086],
    'correct_format_n_eos': [0.55775, 0.55675, 0.56575, 0.5615, 0.5705, 0.56975, 0.5755, 0.57575, 0.5815, 0.5965],
}
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
keywords_count_math_15b = {
    0.1: {'recheck': 34, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 49, 'reevaluation': 0, 're-examine': 1, 'try again': 4, 'check again': 0, 'think again': 0, 'wait': 0},
    0.2: {'recheck': 36, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 36, 'reevaluation': 0, 're-examine': 0, 'try again': 5, 'check again': 0, 'think again': 0, 'wait': 0},
    0.3: {'recheck': 28, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 67, 'reevaluation': 0, 're-examine': 0, 'try again': 6, 'check again': 0, 'think again': 0, 'wait': 0},
    0.4: {'recheck': 86, 'rethink': 2, 'reevaluate': 0, 're-evaluate': 81, 'reevaluation': 0, 're-examine': 1, 'try again': 6, 'check again': 1, 'think again': 0, 'wait': 0},
    0.5: {'recheck': 26, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 77, 'reevaluation': 1, 're-examine': 3, 'try again': 13, 'check again': 0, 'think again': 0, 'wait': 0},
    0.6: {'recheck': 44, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 50, 'reevaluation': 0, 're-examine': 4, 'try again': 5, 'check again': 2, 'think again': 0, 'wait': 0},
    0.7: {'recheck': 34, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 68, 'reevaluation': 0, 're-examine': 4, 'try again': 2, 'check again': 2, 'think again': 0, 'wait': 2},
    0.8: {'recheck': 35, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 40, 'reevaluation': 0, 're-examine': 0, 'try again': 9, 'check again': 3, 'think again': 0, 'wait': 33},
    0.9: {'recheck': 50, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 65, 'reevaluation': 0, 're-examine': 3, 'try again': 12, 'check again': 0, 'think again': 0, 'wait': 8},
    1.0: {'recheck': 51, 'rethink': 1, 'reevaluate': 0, 're-evaluate': 74, 'reevaluation': 0, 're-examine': 4, 'try again': 8, 'check again': 3, 'think again': 0, 'wait': 6}
}



# qwen2.5-math-7b
qwen25_math_7b_stats = {
    # 'num_correct': [356, 375, 398, 415, 411, 416, 430, 426, 440, 420],    # round-1
    # 'num_reflections': [6, 8, 6, 12, 9, 14, 14, 20, 14, 32],              # round-1
    # 'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                 # round-1
    'num_correct': [356, 375, 398, 407, 404, 418, 422, 426, 431, 426],      # round-2
    'num_reflections': [8, 11, 9, 14, 13, 18, 16, 21, 19, 27],              # round-2
    'num_total_keywords': [23, 34, 20, 36, 30, 33, 30, 36, 29, 43],         # round-2
    'precision': [0.667, 0.5, 0.833, 0.75, 0.778, 0.786, 0.643, 0.8, 1.0, 0.781],
    'recall': [0.011, 0.011, 0.013, 0.022, 0.017, 0.026, 0.021, 0.038, 0.032, 0.06],
    'correct_format_n_eos': [0.8125, 0.81075, 0.826, 0.8245, 0.828, 0.83075, 0.82225, 0.8075, 0.775, 0.74375],
}
qwen25_math_7b_stats['num_correct_reflection'] = np.ceil(np.array(qwen25_math_7b_stats['num_reflections']) * np.array(qwen25_math_7b_stats['precision']))
qwen25_math_7b_stats['num_wrong_reflection'] = np.array(qwen25_math_7b_stats['num_reflections']) - np.array(qwen25_math_7b_stats['num_correct_reflection'])
qwen25_math_7b_stats['correct_w_reflection'] = np.array(qwen25_math_7b_stats['precision'])
qwen25_math_7b_stats['correct_wo_reflection'] = (np.array(qwen25_math_7b_stats['num_correct'])-qwen25_math_7b_stats['num_correct_reflection']) / (np.array([500]*10)-np.array(qwen25_math_7b_stats['num_reflections']))
print(f'qwen2.5-math-15b, correctness_w_reflection: {np.mean(qwen25_math_7b_stats["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(qwen25_math_7b_stats["correct_wo_reflection"])}')

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
qwen25_7b_stats = {
    # 'num_correct': [379, 411, 419, 410, 421, 419, 418, 413, 410, 408],    # round-1
    # 'num_reflections': [2, 5, 5, 7, 8, 16, 5, 7, 14, 14],                 # round-1
    # 'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                 # round-1
    'num_correct': [379, 405, 395, 404, 412, 421, 412, 411, 410, 402],      # round-2
    'num_reflections': [2, 5, 4, 8, 9, 5, 6, 7, 14, 14],                    # round-2
    'num_total_keywords': [5, 8, 9, 17, 40, 6, 33, 13, 15, 21],             # round-2
    'precision': [0.5, 0.8, 0.8, 0.714, 0.875, 0.875, 1.0, 0.857, 0.643, 0.857],
    'recall': [0.003, 0.01, 0.01, 0.012, 0.017, 0.033, 0.012, 0.015, 0.022, 0.029],
    'correct_format_n_eos': [0.821, 0.8255, 0.82875, 0.80925, 0.815, 0.797, 0.783, 0.7815, 0.758, 0.741],

}
qwen25_7b_stats['num_correct_reflection'] = np.ceil(np.array(qwen25_7b_stats['num_reflections']) * np.array(qwen25_7b_stats['precision']))
qwen25_7b_stats['num_wrong_reflection'] = np.array(qwen25_7b_stats['num_reflections']) - np.array(qwen25_7b_stats['num_correct_reflection'])
qwen25_7b_stats['correct_w_reflection'] = np.array(qwen25_7b_stats['precision'])
qwen25_7b_stats['correct_wo_reflection'] = (np.array(qwen25_7b_stats['num_correct'])-qwen25_7b_stats['num_correct_reflection']) / (np.array([500]*10)-np.array(qwen25_7b_stats['num_reflections']))
print(f'qwen2.5-7b, correctness_w_reflection: {np.mean(qwen25_7b_stats["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(qwen25_7b_stats["correct_wo_reflection"])}')


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
keywords_count_7b = {
    0.1: {'recheck': 5, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 72, 'reevaluation': 0, 're-examine': 7, 'try again': 0, 'check again': 19, 'think again': 0, 'wait': 0},
    0.2: {'recheck': 8, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 79, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 12, 'think again': 0, 'wait': 0},
    0.3: {'recheck': 9, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 40, 'reevaluation': 0, 're-examine': 1, 'try again': 0, 'check again': 14, 'think again': 0, 'wait': 0},
    0.4: {'recheck': 15, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 115, 'reevaluation': 0, 're-examine': 4, 'try again': 2, 'check again': 9, 'think again': 0, 'wait': 0},
    0.5: {'recheck': 39, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 108, 'reevaluation': 0, 're-examine': 2, 'try again': 1, 'check again': 6, 'think again': 0, 'wait': 0},
    0.6: {'recheck': 5, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 115, 'reevaluation': 0, 're-examine': 1, 'try again': 1, 'check again': 2, 'think again': 0, 'wait': 0},
    0.7: {'recheck': 32, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 90, 'reevaluation': 0, 're-examine': 0, 'try again': 1, 'check again': 4, 'think again': 0, 'wait': 0},
    0.8: {'recheck': 13, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 48, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 2, 'think again': 0, 'wait': 10},
    0.9: {'recheck': 14, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 100, 'reevaluation': 0, 're-examine': 7, 'try again': 1, 'check again': 4, 'think again': 0, 'wait': 12},
    1.0: {'recheck': 16, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 41, 'reevaluation': 1, 're-examine': 1, 'try again': 5, 'check again': 0, 'think again': 0, 'wait': 21}
}


# qwen2.5-3b, qwen instruction
qwen25_3b_stats = {
    'num_correct': [363, 369, 386, 382, 380, 386, 385, 382, 371, 358],
    'num_reflections': [21, 36, 32, 36, 28, 27, 36, 42, 30, 41],
    'num_total_keywords': [40, 51, 47, 49, 35, 41, 45, 48, 34, 50],
    'correct_format_n_eos': [0.524, 0.639, 0.594, 0.528, 0.536, 0.593, 0.75, 0.548, 0.5, 0.634],
}



# deepseek-math-7b-base, qwen
deepseek_math_7b_base_qwen = {
    'num_correct': [67, 66, 75, 98, 100, 103, 104, 119, 117, 105],
    'num_reflections': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'precision': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'recall': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'correct_format_n_eos': [0.14275, 0.1395, 0.138, 0.14275, 0.1415, 0.13775, 0.14, 0.167, 0.173, 0.177],
}

# deepseek-math-7b-base, r1
deepseek_math_7b_base_r1 = {
    'num_correct': [96, 93, 125, 131, 132, 126, 125, 120, 108, 101],
    'num_reflections': [0, 0, 0, 1, 0, 3, 0, 7, 4, 16],
    'num_total_keywords': [0, 0, 0, 4, 0, 12, 0, 10, 5, 29],
    'precision': [0, 0, 0, 0, 0, 0.667, 0, 0.286, 0.25, 0.25],
    'recall': [0, 0, 0, 0, 0, 0.016, 0, 0.017, 0.009, 0.04],
    'correct_format_n_eos': [0.2675, 0.28225, 0.302, 0.31125, 0.32975, 0.32725, 0.35575, 0.36625, 0.36025, 0.34325],
}
deepseek_math_7b_base_r1['num_correct_reflection'] = np.ceil(np.array(deepseek_math_7b_base_r1['num_reflections']) * np.array(deepseek_math_7b_base_r1['precision']))
deepseek_math_7b_base_r1['num_wrong_reflection'] = np.array(deepseek_math_7b_base_r1['num_reflections']) - np.array(deepseek_math_7b_base_r1['num_correct_reflection'])
deepseek_math_7b_base_r1['correct_w_reflection'] = np.array(deepseek_math_7b_base_r1['precision'])
deepseek_math_7b_base_r1['correct_wo_reflection'] = (np.array(deepseek_math_7b_base_r1['num_correct'])-deepseek_math_7b_base_r1['num_correct_reflection']) / (np.array([500]*10)-np.array(deepseek_math_7b_base_r1['num_reflections']))
print(f'deepseek-math-7b-base-r1, correctness_w_reflection: {np.mean(deepseek_math_7b_base_r1["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(deepseek_math_7b_base_r1["correct_wo_reflection"])}')


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
keywords_count_7b_base_r1 = {
    0.1: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 1},
    0.3: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 84},
    0.4: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 34},
    0.5: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 47},
    0.6: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 3},
    0.7: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 3},
    0.8: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 10, 'check again': 0, 'think again': 0, 'wait': 24},
    0.9: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 2, 'reevaluation': 0, 're-examine': 0, 'try again': 3, 'check again': 0, 'think again': 0, 'wait': 13},
    1.0: {'recheck': 0, 'rethink': 5, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 22, 'check again': 0, 'think again': 2, 'wait': 76}
}




# rho-math-7b-v0.1, qwen
rho_math_7b_v01_stats_qwen = {
    'num_correct': [180, 202, 216, 220, 228, 230, 210, 209, 214, 209],
    'num_reflections': [3, 3, 5, 7, 5, 4, 8, 7, 17, 9],
    # 'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'num_total_keywords': [19, 27, 20, 17, 6, 53, 86, 69, 60, 29],
    'precision': [0.333, 0, 0.4, 0.143, 0, 0.5, 0.5, 0.143, 0.353, 0.556],
    'recall': [0.006, 0, 0.009, 0.005, 0, 0.009, 0.019, 0.005, 0.028, 0.024],
    # 'correct_format_n_eos': [0.192, 0.1895, 0.21925, 0.22625, 0.25825, 0.30775, 0.333, 0.39675, 0.4655, 0.54825],
    'correct_format_n_eos': [0.1555, 0.15325, 0.17425, 0.16975, 0.19075, 0.223, 0.23225, 0.263, 0.3095, 0.3385],
}
# rho-math-7b-v0.1, r1
rho_math_7b_v01_stats_gist = {
    'num_correct': [15, 24, 27, 34, 45, 55, 47, 58, 55, 47],
    'num_reflections': [10, 8, 4, 4, 4, 6, 10, 10, 10, 21],
    'precision': [0, 0, 0.25, 0, 0, 0, 0.1, 0, 0, 0.048],
    'recall': [0, 0, 0.037, 0, 0, 0, 0.021, 0, 0, 0.021],
    'correct_format_n_eos': [0.02, 0.02475, 0.03375, 0.04925, 0.06475, 0.09125, 0.1055, 0.12475, 0.13575, 0.135],
}

rho_math_7b_v01_stats_qwen['num_correct_reflection'] = np.ceil(np.array(rho_math_7b_v01_stats_qwen['num_reflections']) * np.array(rho_math_7b_v01_stats_qwen['precision']))
rho_math_7b_v01_stats_qwen['num_wrong_reflection'] = np.array(rho_math_7b_v01_stats_qwen['num_reflections']) - np.array(rho_math_7b_v01_stats_qwen['num_correct_reflection'])
rho_math_7b_v01_stats_qwen['correct_w_reflection'] = np.array(rho_math_7b_v01_stats_qwen['precision'])
rho_math_7b_v01_stats_qwen['correct_wo_reflection'] = (np.array(rho_math_7b_v01_stats_qwen['num_correct'])-rho_math_7b_v01_stats_qwen['num_correct_reflection']) / (np.array([500]*10)-np.array(rho_math_7b_v01_stats_qwen['num_reflections']))
print(f'rho-math-7b-v0.1, correctness_w_reflection: {np.mean(rho_math_7b_v01_stats_qwen["correct_w_reflection"])}, correctness_wo_reflection: {np.mean(rho_math_7b_v01_stats_qwen["correct_wo_reflection"])}')


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
keywords_count_rho_7b = {
    0.1: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 19, 'check again': 0, 'think again': 0, 'wait': 1},
    0.2: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 27, 'check again': 0, 'think again': 0, 'wait': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'try again': 17, 'check again': 2, 'think again': 0, 'wait': 0},
    0.4: {'recheck': 1, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'try again': 12, 'check again': 3, 'think again': 0, 'wait': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 3, 'check again': 3, 'think again': 0, 'wait': 0},
    0.6: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 52, 'check again': 1, 'think again': 0, 'wait': 1},
    0.7: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 11, 'reevaluation': 0, 're-examine': 0, 'try again': 74, 'check again': 1, 'think again': 0, 'wait': 0},
    0.8: {'recheck': 0, 'rethink': 10, 'reevaluate': 7, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'try again': 58, 'check again': 0, 'think again': 0, 'wait': 4},
    0.9: {'recheck': 1, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 1, 'try again': 57, 'check again': 1, 'think again': 0, 'wait': 20},
    1.0: {'recheck': 0, 'rethink': 0, 'reevaluate': 3, 're-evaluate': 1, 'reevaluation': 0, 're-examine': 0, 'try again': 28, 'check again': 0, 'think again': 0, 'wait': 8}
}



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
    'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'precision': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'recall': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'correct_format_n_eos': [0.00525, 0.0095, 0.00925, 0.02375, 0.0275, 0.05125, 0.07625, 0.082, 0.092, 0.0985],
}
llama_31_8b_stats_r1 = {
    'num_correct': [109, 135, 149, 162, 166, 146, 148, 128, 120, 97],
    # 'num_reflections': [0, 0, 0, 0, 0, 0, 0, 1, 0, 15],
    'num_reflections': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'num_total_keywords': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 'precision': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.133],
    'precision': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 'recall': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.021],
    'recall': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'correct_format_n_eos': [0.1037, 0.075, 0.1135, 0.15825, 0.22275, 0.284, 0.33025, 0.36125, 0.3865, 0.37575],
}
keywords_count_8b_r1 = {
    0.1: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.4: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 7},
    0.6: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 1},
    0.7: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 24},
    0.8: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 47},
    0.9: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 62},
    1.0: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 138}
    # 0.8: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 3, 'wait': 0},
    # 0.9: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 31, 'check again': 1, 'think again': 11, 'wait': 0},
    # 1.0: {'recheck': 0, 'rethink': 2, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 45, 'check again': 0, 'think again': 1, 'wait': 0}
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
llama_32_3b_keywords = {
    0.1: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.2: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.3: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.4: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 0},
    0.5: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 7},
    0.6: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 1},
    0.7: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 24},
    0.8: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 47},
    0.9: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 62},
    1.0: {'recheck': 0, 'rethink': 0, 'reevaluate': 0, 're-evaluate': 0, 'reevaluation': 0, 're-examine': 0, 'try again': 0, 'check again': 0, 'think again': 0, 'wait': 138}
}



# llama-3.2-3b-instruct, qwen instruction
llama_32_3b_inst_qwen_stats = {
    'num_correct': [414, 425, 431, 417, 423, 424, 414, 420, 419, 407],
    'num_reflections': [31, 41, 44, 42, 30, 37, 32, 38, 45, 39],
    'num_total_keywords': [55, 55, 60, 48, 42, 46, 38, 47, 51, 45],
    'correct_format_n_eos': [0.677, 0.805, 0.841, 0.738, 0.833, 0.757, 0.719, 0.789, 0.756, 0.744],
}

# llama-3.2-3b-instruct, qwen instruction
llama_32_3b_inst_r1_stats = {
    'num_correct': [305, 329, 342, 337, 349, 333, 349, 334, 314, 302],
    'num_reflections': [46, 50, 68, 73, 93, 86, 109, 130, 136, 129],
    'num_total_keywords': [69, 68, 78, 84, 124, 106, 133, 155, 147, 150],
    'correct_format_n_eos': [0.478, 0.68, 0.632, 0.589, 0.602, 0.628, 0.679, 0.646, 0.61, 0.62],
}


# qwen2.5-math-1.5b, qwen instruction
qwen25_math_15b_stats = {
    'num_correct': [257, 285, 292, 319, 336, 348, 382, 379, 375, 376],      
    'num_reflections': [15, 17, 19, 18, 18, 21, 22, 30, 35, 39],            
    'num_total_keywords': [38, 41, 35, 94, 39, 50, 36, 45, 63, 60],         
    'correct_format_n_eos': [0.55775, 0.55675, 0.56575, 0.5615, 0.5705, 0.56975, 0.5755, 0.57575, 0.5815, 0.5965],
}

# r1-distill-qwen-1.5b, qwen instruction
distill_qwen_15b_stats = {
    'num_correct': [298, 296, 310, 310, 295, 303, 301, 293, 287, 288],
    'num_reflections': [290, 303, 303, 312, 307, 313, 306, 300, 304, 295],
    'num_total_keywords': [725, 725, 744, 735, 688, 683, 662, 561, 552, 524],
    'correct_format_n_eos': [0.555, 0.531, 0.581, 0.58, 0.599, 0.575, 0.578, 0.58, 0.562, 0.566],
}

# r1-distill-qwen-1.5b, r1 instruction
distill_qwen_15b_r1_stats = {
    'num_correct': [90, 99, 119, 126, 137, 150, 154, 148, 163, 152],
    'num_reflections': [159, 169, 182, 167, 177, 171, 181, 176, 179, 176],
    'num_total_keywords': [338, 334, 363, 303, 304, 283, 302, 299, 268, 272],
    'correct_format_n_eos': [0.157, 0.124, 0.214, 0.21, 0.203, 0.24, 0.276, 0.233, 0.274, 0.239],
}




# Define the datasets
models = {
    "Qwen2.5-Math-7B": keywords_count_math_7b,
    "Qwen2.5-7B": keywords_count_7b,
    "Qwen2.5-Math-1.5B": keywords_count_math_15b,
    # "Llama-3.1-8B": keywords_count_8b_r1,
    "Deepseek-Math-7B": keywords_count_7b_base_r1,
    "Rho-Math-7B": keywords_count_rho_7b,
}
# remove 'wait' from keywords
for model_data in models.values():
    for temp_data in model_data.values():
        temp_data.pop('wait', None)

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
width = 0.8 / 5  # 5 models, so each bar gets 16% of the available space
fig, ax = plt.subplots(figsize=(8, 6))

# Add dashed vertical lines between keyword groups
for i in range(len(keywords) - 1):
    ax.axvline(x=i + 0.5, color="gray", linestyle="dashed", linewidth=0.7)

# Plot bars with adjusted width
for i, (model, kw_counts) in enumerate(model_sums.items()):
    ax.bar(x + i * width - (2 * width),  # Center-align bars around each tick
           [kw_counts[kw] for kw in keywords], 
           width, 
           label=model)

ax.set_yscale("symlog", linthresh=40)  
ax.set_ylim(0, 1000)
ax.set_xticks(x)
ax.set_xlim(-0.5, len(keywords) - 0.5)
ax.set_xticklabels(keywords, rotation=45, ha="right", fontsize=16)
ax.set_ylabel("Number of self-reflection keywords", fontsize=16)
ax.legend(loc='upper center', bbox_to_anchor=(0.7, 1.0), ncol=1, fontsize=16)
plt.savefig("responses/plot_keyword_occurrences.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_keyword_occurrences.svg", dpi=300, bbox_inches='tight')


def plot_keyword_distribution(data_dict, model_name, save_path=None):
    # Prepare data for DataFrame
    data = []
    for temp, responses in data_dict.items():
        for response_type, keyword_counts in responses.items():
            for num_keywords, count in keyword_counts.items():
                if num_keywords >= 1:  # Only consider num_keywords >= 1
                    data.extend([[num_keywords, response_type]] * count)
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Num Keywords", "Response Type"])
    palette = {"correct": "blue", "wrong": "red"}
    plt.figure(figsize=(10, 6))
    sns.histplot(df, x="Num Keywords", hue="Response Type", multiple="stack", discrete=True, shrink=0.8, palette=palette)

    # Improve legend with color patches
    correct_patch = mpatches.Patch(color="blue", label="Correct Response")
    wrong_patch = mpatches.Patch(color="red", label="Wrong Response")
    plt.legend(handles=[correct_patch, wrong_patch], title="Response Type", loc="upper right")

    # Labels & title
    plt.xlabel("Number of Keywords in Response")
    plt.ylabel("Frequency")
    plt.title(f"Model = {model_name}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# plot_keyword_distribution(num_keywords_per_response_math_15b, "Qwen2.5-Math-1.5B", "figures/plot_num_keywords_per_response_math_15b.png")
# plot_keyword_distribution(num_keywords_per_response_math_7b, "Qwen2.5-Math-7B", "plots/plot_num_keywords_per_response_math_7b.png")
# plot_keyword_distribution(num_keywords_per_response_math_7b, "Qwen2.5-7B", "plots/plot_num_keywords_per_response_7b.png")


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

def plot_keyword_distribution_temp(data_dict, model_name, save_path=None):
    # Prepare data for DataFrame
    data = []
    for temp, responses in data_dict.items():
        for response_type, keyword_counts in responses.items():
            for num_keywords, count in keyword_counts.items():
                if num_keywords >= 1:  # Only consider num_keywords >= 1
                    data.append([temp, num_keywords, response_type, count])
    df = pd.DataFrame(data, columns=["Temperature", "Num Keywords", "Response Type", "Count"])

    # Plot with grouped bars
    plt.figure(figsize=(10, 6))
    palette = {"correct": "blue", "wrong": "red"}

    # Use only one barplot
    sns.barplot(data=df, x="Temperature", y="Count", hue="Response Type", estimator=sum, ci=None, palette=palette, alpha=0.85)

    # Improve legend with color patches
    correct_patch = mpatches.Patch(color="blue", alpha=0.7, label="Correct Response")
    wrong_patch = mpatches.Patch(color="red", alpha=0.7, label="Incorrect Response")
    plt.legend(handles=[correct_patch, wrong_patch], loc="upper left", fontsize=14)

    # Labels & title
    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("Number of self-reflections", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.savefig(save_path.replace(".png", ".svg"), format='svg', dpi=300, bbox_inches='tight')

# Example usage
plot_keyword_distribution_temp(num_keywords_per_response_math_15b, "Qwen2.5-Math-1.5B", "figures/plot_num_keywords_per_response_math_15b_temp.png")
# plot_keyword_distribution_temp(num_keywords_per_response_math_7b, "Qwen2.5-Math-7B", "responses/plot_num_keywords_per_response_math_7b_temp.png")
# plot_keyword_distribution_temp(num_keywords_per_response_7b, "Qwen2.5-Math-7B", "responses/plot_num_keywords_per_response_7b_temp.png")

print(f'plot done.')
sys.exit(0)


temp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# plot the correct_format_n_eos, with sns
plt.figure(figsize=(10, 5))
sns.lineplot(x=temp, y=qwen25_math_7b_stats['correct_format_n_eos'], label="Qwen2.5-Math-7B", marker="o", color=methods_to_color["Qwen2.5-Math-7B"])
sns.lineplot(x=temp, y=qwen25_7b_stats['correct_format_n_eos'], label="Qwen2.5-7B", marker="o", color=methods_to_color["Qwen2.5-7B"])
sns.lineplot(x=temp, y=qwen25_math_15b_stats['correct_format_n_eos'], label="Qwen2.5-Math-1.5B", marker="o", color=methods_to_color["Qwen2.5-Math-1.5B"])
sns.lineplot(x=temp, y=llama_31_8b_stats_r1['correct_format_n_eos'], label="Llama-3.1-8B", marker="^", color=methods_to_color["Llama-3.1-8B"])
sns.lineplot(x=temp, y=deepseek_math_7b_base_r1['correct_format_n_eos'], label="Deepseek-Math-7B", marker="s", color=methods_to_color["Deepseek-Math-7B"])
sns.lineplot(x=temp, y=rho_math_7b_v01_stats_qwen['correct_format_n_eos'], label="Rho-Math-7B", marker="d", color=methods_to_color["Rho-Math-7B"])
plt.ylabel("Percentage of correct format and <eos>", fontsize=14)
plt.xlabel("Temperature", fontsize=14)
plt.xticks(temp, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.0, 1.01)
plt.xlim(0.08, 1.02)
sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(0.05, 1.2), ncol=3, frameon=True, fontsize=12)
plt.savefig(f"responses/plot_correct_format_n_eos.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_correct_format_n_eos.svg", format='svg', dpi=300, bbox_inches='tight')


# number of questions with self-reflection
plt.figure(figsize=(8, 6))
sns.lineplot(x=temp, y=qwen25_math_15b_stats['num_reflections'], label="Qwen2.5-Math-1.5B", marker='o', color=methods_to_color["Qwen2.5-Math-1.5B"])
sns.lineplot(x=temp, y=qwen25_math_7b_stats['num_reflections'], label="Qwen2.5-Math-7B", marker='o', color=methods_to_color["Qwen2.5-Math-7B"])
sns.lineplot(x=temp, y=qwen25_7b_stats['num_reflections'], label="Qwen2.5-7B", marker='o', color=methods_to_color["Qwen2.5-7B"])
sns.lineplot(x=temp, y=rho_math_7b_v01_stats_qwen['num_reflections'], label="Rho-Math-7B", marker='d', color=methods_to_color["Rho-Math-7B"])
sns.lineplot(x=temp, y=deepseek_math_7b_base_r1['num_reflections'], label="Deepseek-Math-7B", marker='s', color=methods_to_color["Deepseek-Math-7B"])
sns.lineplot(x=temp, y=llama_31_8b_stats_r1['num_reflections'], label="Llama-3.1-8B", marker='^', color=methods_to_color["Llama-3.1-8B"])
plt.ylabel("Number of questions eliciting self-reflection", fontsize=13)
plt.xlabel("Temperature", fontsize=14)
plt.xticks(temp, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-1, 50)
plt.xlim(0.08, 1.02)
sns.move_legend(plt.gca(), "upper left", ncol=1, frameon=True, fontsize=14)
plt.savefig("responses/plot_num_q_with_reflections.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_num_q_with_reflections.svg", format='svg', dpi=300, bbox_inches='tight')



# plot num_correct and num_reflections in two horizontal subplots, using sns plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plt.subplots_adjust(wspace=0.15) 

# First subplot: Number of correct answers
sns.lineplot(x=temp, y=qwen25_math_7b_stats['num_correct'], label="Qwen2.5-Math-7B", ax=ax1, marker='o', color=methods_to_color["Qwen2.5-Math-7B"])
sns.lineplot(x=temp, y=qwen25_7b_stats['num_correct'], label="Qwen2.5-7B", ax=ax1, marker='o', color=methods_to_color["Qwen2.5-7B"])
sns.lineplot(x=temp, y=qwen25_math_15b_stats['num_correct'], label="Qwen2.5-Math-1.5B", ax=ax1, marker='o', color=methods_to_color["Qwen2.5-Math-1.5B"])
sns.lineplot(x=temp, y=llama_31_8b_stats_r1['num_correct'], label="Llama-3.1-8B", ax=ax1, marker='^', color=methods_to_color["Llama-3.1-8B"])
sns.lineplot(x=temp, y=deepseek_math_7b_base_r1['num_correct'], label="Deepseek-Math-7B", ax=ax1, marker='s', color=methods_to_color["Deepseek-Math-7B"])
sns.lineplot(x=temp, y=rho_math_7b_v01_stats_qwen['num_correct'], label="Rho-Math-7B", ax=ax1, marker='d', color=methods_to_color["Rho-Math-7B"])
ax1.set_ylabel("Number of correct answers", fontsize=134)
ax1.set_xlabel("Temperature", fontsize=14)
ax1.set_ylim(-5, 500)
ax1.set_xlim(0.08, 1.02)
ax1.set_xticks(temp)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend().set_visible(False)
# Second subplot: Number of self-reflections
sns.lineplot(x=temp, y=qwen25_math_15b_stats['num_reflections'], label="Qwen2.5-Math-1.5B", ax=ax2, marker='o', color=methods_to_color["Qwen2.5-Math-1.5B"])
sns.lineplot(x=temp, y=qwen25_math_7b_stats['num_reflections'], label="Qwen2.5-Math-7B", ax=ax2, marker='o', color=methods_to_color["Qwen2.5-Math-7B"])
sns.lineplot(x=temp, y=qwen25_7b_stats['num_reflections'], label="Qwen2.5-7B", ax=ax2, marker='o', color=methods_to_color["Qwen2.5-7B"])
sns.lineplot(x=temp, y=rho_math_7b_v01_stats_qwen['num_reflections'], label="Rho-Math-7B", ax=ax2, marker='d', color=methods_to_color["Rho-Math-7B"])
sns.lineplot(x=temp, y=deepseek_math_7b_base_r1['num_reflections'], label="Deepseek-Math-7B", ax=ax2, marker='s', color=methods_to_color["Deepseek-Math-7B"])
sns.lineplot(x=temp, y=llama_31_8b_stats_r1['num_reflections'], label="Llama-3.1-8B", ax=ax2, marker='^', color=methods_to_color["Llama-3.1-8B"])
ax2.set_ylabel("Number of questions eliciting self-reflection", fontsize=13)
ax2.set_xlabel("Temperature", fontsize=14)
ax2.set_xticks(temp)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_ylim(-1, 50)
ax2.set_xlim(0.08, 1.02)
sns.move_legend(plt.gca(), "upper left", ncol=1, frameon=True, fontsize=14)
# sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(-1.15, 1.12), ncol=6, frameon=True, fontsize=10.5)
plt.savefig("responses/plot_num_correct_and_reflections.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_num_correct_and_reflections.svg", format='svg', dpi=300, bbox_inches='tight')


# plot the total number of keywords
plt.figure(figsize=(10, 5))
plt.plot(temp, qwen25_math_15b_stats['num_total_keywords'], label="Qwen2.5-Math-1.5B", marker='o', color=methods_to_color["Qwen2.5-Math-1.5B"])
plt.plot(temp, qwen25_math_7b_stats['num_total_keywords'], label="Qwen2.5-Math-7B", marker='o', color=methods_to_color["Qwen2.5-Math-7B"])
plt.plot(temp, qwen25_7b_stats['num_total_keywords'], label="Qwen2.5-7B", marker='o', color=methods_to_color["Qwen2.5-7B"])
plt.plot(temp, rho_math_7b_v01_stats_qwen['num_total_keywords'], label="Rho-Math-7B", marker='d', color=methods_to_color["Rho-Math-7B"])
plt.plot(temp, deepseek_math_7b_base_r1['num_total_keywords'], label="Deepseek-Math-7B", marker='s', color=methods_to_color["Deepseek-Math-7B"])
plt.plot(temp, llama_31_8b_stats_r1['num_total_keywords'], label="Llama-3.1-8B", marker='^', color=methods_to_color["Llama-3.1-8B"])
plt.ylabel("Total number of keywords", fontsize=14)
plt.xlabel("Temperature", fontsize=14)
plt.xticks(temp, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(-5, 100)
plt.xlim(0.08, 1.02)
plt.legend()
plt.savefig("responses/plot_total_keywords.png", dpi=100, bbox_inches='tight')
plt.savefig("responses/plot_total_keywords.svg", format='svg', dpi=300, bbox_inches='tight')


# plot bar plot, show num_correct_reflection vs num_wrong_reflection
models = [
    ("Qwen2.5-Math-7B", qwen25_math_7b_stats['num_correct_reflection'], qwen25_math_7b_stats['num_wrong_reflection'], methods_to_color["Qwen2.5-Math-7B"]),
    ("Qwen2.5-7B", qwen25_math_7b_stats['num_correct_reflection'], qwen25_math_7b_stats['num_wrong_reflection'], methods_to_color["Qwen2.5-7B"]),
    ("Qwen2.5-Math-1.5B", qwen25_math_15b_stats['num_correct_reflection'], qwen25_math_15b_stats['num_wrong_reflection'], methods_to_color["Qwen2.5-Math-1.5B"]),
    ("Deepseek-Math-7B-base", deepseek_math_7b_base_r1['num_correct_reflection'], deepseek_math_7b_base_r1['num_wrong_reflection'], methods_to_color["Deepseek-Math-7B"]),
    ("Rho-Math-7B", rho_math_7b_v01_stats_qwen['num_correct_reflection'], rho_math_7b_v01_stats_qwen['num_wrong_reflection'], methods_to_color["Rho-Math-7B"]),
]
fig, ax = plt.subplots(figsize=(12, 6))
barWidth = 0.15  # Adjusted to fit all models
r = np.arange(len(temp))  # Base x-axis positions
for i, (model_name, correct, wrong, color) in enumerate(models):
    x_pos = r + i * barWidth  # Offset each model's bars
    ax.bar(x_pos, correct, color=color, alpha=1.0, edgecolor=color, width=barWidth, label=f'Correct - {model_name}')
    ax.bar(x_pos, wrong, color='white', edgecolor=color, alpha=1.0, width=barWidth, bottom=correct, hatch='//', label=f'Wrong - {model_name}')
ax.set_ylabel("Number of self-reflections", fontsize=14)
ax.set_xlabel("Temperature", fontsize=14)
ax.set_xticks(r + (len(models) - 1) * barWidth / 2)  # Center xticks
ax.set_xticklabels(temp)
ax.set_xlim(-0.2, len(temp) - 0.2)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# ax.legend(loc='upper left', fontsize=14)

# First Legend: Model colors
# '''
handles_color = [mpatches.Patch(color=color, label=model_name) for model_name, _, _, color in models]
legend1 = ax.legend(handles=handles_color, loc='upper left', frameon=False, fontsize=14)
ax.add_artist(legend1)  # Ensure the first legend stays on the plot
# Second Legend: Bar fill and hatch explanation
correct_patch = mpatches.Patch(color="gray", alpha=1.0, label="Correct (filled)")
wrong_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Incorrect (hatched)")
legend2 = ax.legend(handles=[correct_patch, wrong_patch], loc='upper left', bbox_to_anchor=(0.35, 1), frameon=False, fontsize=14)
# '''
plt.savefig(f"responses/plot_correct_wrong_reflections.png", dpi=100, bbox_inches='tight')
plt.savefig(f"responses/plot_correct_wrong_reflections.svg", format='svg', dpi=300, bbox_inches='tight')


# plot the number of reflections per level
plt.figure(figsize=(5, 5))
plt.bar(reflections_per_level_math_7b.keys(), reflections_per_level_math_7b.values(), color="red", alpha=0.7)
plt.ylabel("Number of self-reflections")
plt.xlabel("Difficulty level")
# plt.title(f"Reflections per level (Qwen2.5-Math-7B across 10 temps)")
plt.savefig(f"responses/plot_reflections_per_level.png", dpi=300, bbox_inches='tight')


'''
# precision
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=temp, y=precision_math_15b, label="Qwen2.5-Math-1.5B", ax=ax)
sns.lineplot(x=temp, y=precision_math_7b, label="Qwen2.5-Math-7B", ax=ax)
sns.lineplot(x=temp, y=precision_7b, label="Qwen2.5-7B", ax=ax)
sns.lineplot(x=temp, y=precision_7b_base, label="Deepseek-Math-7B", ax=ax)
sns.lineplot(x=temp, y=precision_rho_7b, label="Rho-Math-7B", ax=ax)
sns.lineplot(x=temp, y=precision_8b, label="Llama-3.1-8B", ax=ax)
ax.set_ylabel("Precision", fontsize=14)
ax.set_xlabel("Temperature", fontsize=14)
ax.set_xticks(temp)
ax.set_ylim(-0.02, 1.02)
ax.legend()
plt.savefig(f"responses/plot_precision.png", dpi=100, bbox_inches='tight')
plt.savefig(f"responses/plot_precision.svg", format='svg', dpi=300, bbox_inches='tight')


# recall
plt.figure(figsize=(10, 5))
plt.plot(temp, recall_math_15b, label="Qwen2.5-Math-1.5B", color="blue", marker="o")
plt.plot(temp, recall_math_7b, label="Qwen2.5-Math-7B", color="red", marker="o")
plt.plot(temp, recall_7b, label="Qwen2.5-7B", color="green", marker="o")
plt.plot(temp, recall_1b, label="LLAMA-3.2-1B", color="purple", marker="o")
plt.ylabel("Recall")
plt.title(f"Recall")
plt.legend()
plt.ylim(0, 0.16)
# plt.savefig(f"responses/plot_recall.png", dpi=300, bbox_inches='tight')
'''





