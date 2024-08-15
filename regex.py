import itertools

import cv2
import numpy as np
import pytesseract
import re
from Levenshtein import distance

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def split_into_sections(text):
    sections = []
    section = []
    for line in text.split("\n"):
        if line.strip() == "":
            if section:
                sections.append("\n".join(section))
                section = []
        else:
            section.append(line)
    if section:
        sections.append("\n".join(section))
    return sections


def remove_short_lined_sections(sections):
    relevant_sections = []
    for section in sections:
        split_section = section.split("\n")
        for line in split_section:
            if len(line) > 20:
                relevant_sections.append(section)
                break
            else:
                continue

    return relevant_sections


def find_table_pattern_levenshtein(first_column, last_column):
    min_rows = 3
    if len(first_column) < min_rows:
        return False

    # Check similarity in each column
    columns = list(zip(first_column, last_column))
    for col in columns:
        avg_distance = sum(distance(col[0], item) for item in col[1:]) / (len(col) - 1)
        if avg_distance > 8:  # (5) Adjust this threshold as needed
            return False

    return True


def find_table_pattern_statistical(first_column, last_column):
    min_rows = 3
    if len(first_column) < min_rows:
        return False

    columns = list(zip(first_column, last_column))
    for col in columns:
        lengths = [len(item) for item in col]
        length_std = np.std(lengths)

        char_types = [''.join(set(item.lower())) for item in col]
        type_std = np.std([len(types) for types in char_types])

        if length_std > 4 or type_std > 3:  # (2, 1) Adjust these thresholds as needed
            return False

    return True


def gather_column_patterns(column_patterns):
    same_length = False  # all strings have the exact same length
    same_length_1 = False  # all strings have the same length +/- 1
    same_length_2 = False  # all strings have the same length +/- 2

    has_digits = False  # all strings have at least one digit
    same_digits = False  # all strings have the same number of digits
    same_digits_1 = False  # all strings have the same number of digits +/- 1
    same_digits_2 = False  # all strings have the same number of digits +/- 2

    has_alpha_upper = False  # all strings have at least one uppercase letter
    same_alpha_upper = False  # all strings have the same number of uppercase letters
    same_alpha_upper_1 = False  # all strings have the same number of uppercase letters +/- 1
    same_alpha_upper_2 = False  # all strings have the same number of uppercase letters +/- 2

    has_alpha_lower = False  # all strings have at least one lowercase letter
    same_alpha_lower = False  # all strings have the same number of lowercase letters
    same_alpha_lower_1 = False  # all strings have the same number of lowercase letters +/- 1
    same_alpha_lower_2 = False  # all strings have the same number of lowercase letters +/- 2

    has_special = False  # all strings have at least one special character
    same_special = False  # all strings have the same number of special characters
    same_special_1 = False  # all strings have the same number of special characters +/- 1
    same_special_2 = False  # all strings have the same number of special characters +/- 2

    patterns = {
        "same_length": False, "same_length_1": False, "same_length_2": False,
        "has_digits": False, "same_digits": False, "same_digits_1": False, "same_digits_2": False,
        "has_alpha_upper": False, "same_alpha_upper": False, "same_alpha_upper_1": False, "same_alpha_upper_2": False,
        "has_alpha_lower": False, "same_alpha_lower": False, "same_alpha_lower_1": False, "same_alpha_lower_2": False,
        "has_special": False, "same_special": False, "same_special_1": False, "same_special_2": False,
    }

    checks = [
        ("length", "same_length", "same_length_1", "same_length_2"),
        ("digits", "same_digits", "same_digits_1", "same_digits_2"),
        ("alpha_upper", "same_alpha_upper", "same_alpha_upper_1", "same_alpha_upper_2"),
        ("alpha_lower", "same_alpha_lower", "same_alpha_lower_1", "same_alpha_lower_2"),
        ("special", "same_special", "same_special_1", "same_special_2")
    ]

    length = [item["length"] for item in column_patterns]
    digits = [item["digit"] for item in column_patterns]
    alpha_upper = [item["alpha_upper"] for item in column_patterns]
    alpha_lower = [item["alpha_lower"] for item in column_patterns]
    special = [item["special"] for item in column_patterns]

    # Perform the checks in a loop
    for feature, same, same_1, same_2 in checks:
        values = locals()[feature]  # Get the list of values for the current feature

        if all(v > 0 for v in values):  # Check for presence
            patterns["has_" + feature] = True

        if len(set(values)) == 1:  # All values are the same
            patterns[same] = True
        if max(values) - min(values) == 1:  # Values differ by at most 1
            patterns[same_1] = True
        if max(values) - min(values) == 2:  # Values differ by at most 2
            patterns[same_2] = True

    return patterns


def generate_regex_patterns(pattern, length=[], digits=[], alpha_upper=[], alpha_lower=[], special=[]):
    patterns = []

    # Define character classes
    digit_class = r'\d'
    upper_class = r'[A-Z]'
    lower_class = r'[a-z]'
    special_class = r'[^A-Za-z0-9]'

    # Generate length constraints
    length_constraints = []
    if pattern["same_length"]:
        length_constraints.append(f'{{{min(length)}}}')
    elif pattern["same_length_1"]:
        length_constraints.append(f'{{{min(length) - 1},{max(length) + 1}}}')
    elif pattern["same_length_2"]:
        length_constraints.append(f'{{{min(length) - 2},{max(length) + 2}}}')
    else:
        length_constraints.append(f'{{{min(length)},{max(length)}}}')

    # Generate character class constraints
    char_constraints = []
    for has_flag, same_flag, same_1_flag, same_2_flag, char_class, counts in [
        (pattern["has_digits"], pattern["same_digits"], pattern["same_digits_1"], pattern["same_digits_2"], digit_class,
         digits),
        (pattern["has_alpha_upper"], pattern["same_alpha_upper"], pattern["same_alpha_upper_1"],
         pattern["same_alpha_upper_2"], upper_class, alpha_upper),
        (pattern["has_alpha_lower"], pattern["same_alpha_lower"], pattern["same_alpha_lower_1"],
         pattern["same_alpha_lower_2"], lower_class, alpha_lower),
        (pattern["has_special"], pattern["same_special"], pattern["same_special_1"], pattern["same_special_2"],
         special_class, special)
    ]:
        if has_flag:
            if same_flag:
                char_constraints.append(f'(?:{char_class}{{{min(counts)}}})')
            elif same_1_flag:
                char_constraints.append(f'(?:{char_class}{{{max(0, min(counts) - 1)},{max(counts) + 1}}})')
            elif same_2_flag:
                char_constraints.append(f'(?:{char_class}{{{max(0, min(counts) - 2)},{max(counts) + 2}}})')
            else:
                char_constraints.append(f'(?:{char_class}{{{min(counts)},{max(counts)}}})')
        else:
            char_constraints.append(f'(?:{char_class}*)')

    # Generate all combinations of character class constraints
    for combo in itertools.product(*[['', constraint] for constraint in char_constraints]):
        for length_constraint in length_constraints:
            pattern = f'^{"".join(combo)}{length_constraint}$'
            patterns.append(pattern)

    # Return the pattern with the most constraints
    index_longest_constraint = np.argmax([len(pattern) for pattern in patterns])

    return patterns[index_longest_constraint]


def match_pattern(pattern, item):
    try:
        re.match(pattern, item)
    except re.error:
        return False
    return True


# def find_most_restraining_pattern(regex_patterns, column):
#     matching_counts = [sum(bool(match_pattern(pattern, item)) for item in column) for pattern in regex_patterns]
#     print(matching_counts)
#     matching_patterns = regex_patterns[np.argmax(matching_counts)]
# 
#     if len(matching_patterns) == 1:
#         return matching_patterns[0]
#     else:


def find_column_pattern(column_patterns, column):
    pattern = gather_column_patterns(column_patterns)

    length = [item['length'] for item in column_patterns]
    digits = [item['digit'] for item in column_patterns]
    alpha_upper = [item['alpha_upper'] for item in column_patterns]
    alpha_lower = [item['alpha_lower'] for item in column_patterns]
    special = [item['special'] for item in column_patterns]

    regex_pattern = generate_regex_patterns(pattern, length=length, digits=digits, alpha_upper=alpha_upper,
                                   alpha_lower=alpha_lower, special=special)

    # best_regex = find_most_restraining_pattern(regex_patterns, column)
    print("Len pattern: " + str(len(regex_pattern)))
    if len(regex_pattern) > 55:  # 10
        return True
    return False


def find_table_pattern_heuristics(first_column, last_column):
    first_column_patterns = []
    last_column_patterns = []

    # Check if the first column is a header
    # find_table_pattern_heuristics(first_column, last_column)

    for item in first_column:
        pattern_counter = {"length": len(item), "digit": sum(c.isdigit() for c in item),
                           "alpha_upper": sum(c.isupper() for c in item),
                           "alpha_lower": sum(c.islower() for c in item),
                           "special": sum(not c.isalnum() for c in item)}

        first_column_patterns.append(pattern_counter)

    for item in last_column:
        pattern_counter = {"length": len(item), "digit": sum(c.isdigit() for c in item),
                           "alpha_upper": sum(c.isupper() for c in item),
                           "alpha_lower": sum(c.islower() for c in item),
                           "special": sum(not c.isalnum() for c in item)}

        last_column_patterns.append(pattern_counter)

    # Ensure a table with more than 2 row
    if len(first_column) < 3 or len(last_column) < 3:
        return False

    if find_column_pattern(last_column_patterns, last_column) or find_column_pattern(first_column_patterns, first_column):
        return True

    return False


def find_sections_with_patterns(sections):
    relevant_sections = []

    for section in sections:
        first_column = []
        last_column = []
        split_sections = section.split("\n")
        for line in split_sections:
            split_line = line.split()
            first_column.append(split_line[0])
            last_column.append(split_line[-1])

        if first_column == last_column:
            continue

        # if find_table_pattern_heuristics(first_column, last_column):
        #     relevant_sections.append(section)
        #     print("***HEU***")
        if find_table_pattern_statistical(first_column, last_column):
            relevant_sections.append(section)
            print("***STA***")
        elif find_table_pattern_levenshtein(first_column, last_column):
            relevant_sections.append(section)
            print("***LEV***")

    print("No. relevant sections: " + str(len(relevant_sections)))
    return relevant_sections


def extract_relevant_sections(sections):
    sec1 = remove_short_lined_sections(sections)
    print("No. non short-lined sections: " + str(len(sec1)))
    sec2 = find_sections_with_patterns(sec1)

    # TODO: Remove header if there is one

    return sec2


image_path = ""
text = pytesseract.image_to_string(image_path)
# print(text)
# print(30 * "+")

sections = split_into_sections(text)
relevant_sections = extract_relevant_sections(sections)

for sec in relevant_sections:
    print(sec)
    print(30 * "-")
