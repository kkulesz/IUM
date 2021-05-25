log_file = "ab_events.log"


def read_data_from_log_file():
    file = open(log_file, "r")

    session_predictions = {}
    session_results = {}

    prediction_prefix = 'INFO:root:Prediction:'
    result_prefix = 'INFO:root:Result:'

    len_of_prediction_prefix = len(prediction_prefix)
    len_of_result_prefix = len(result_prefix)

    log_line = file.readline()
    while log_line:
        log_line = log_line.replace(" ", "").replace("\n", "")

        if log_line.startswith(prediction_prefix):

            log_line = log_line[len_of_prediction_prefix:]
            model, session_id, prediction = log_line.split(',')
            session_id = str(int(float(session_id)))
            session_predictions[session_id] = [model, prediction]

        elif log_line.startswith(result_prefix):

            log_line = log_line[len_of_result_prefix:]
            session_id, result = log_line.split(',')
            session_id = str(int(float(session_id)))
            session_results[session_id] = result

        log_line = file.readline()
    return session_predictions, session_results


def count_successful_predictions(pred_dict, res_dict):
    sum_of_correct_for_A = 0
    sum_of_correct_for_B = 0
    sum_of_all_A = 0
    sum_of_all_B = 0
    no_of_false_possitive_for_our_model = 0

    session_ids = pred_dict.keys()
    for ses_id in session_ids:
        model = pred_dict[ses_id][0]
        is_correct = pred_dict[ses_id][1] == res_dict[ses_id]

        if model == "A":
            sum_of_all_A += 1
            if is_correct:
                sum_of_correct_for_A += 1
        elif model == "B":
            sum_of_all_B += 1
            if is_correct:
                sum_of_correct_for_B += 1
            if pred_dict[ses_id][1] == 'True' and res_dict[ses_id] == 'False':
                no_of_false_possitive_for_our_model += 1

    proc_of_A = round((sum_of_correct_for_A/sum_of_all_A) * 100, 2)
    proc_of_B = round((sum_of_correct_for_B/sum_of_all_B) * 100, 2)
    return (proc_of_A, sum_of_all_A), (proc_of_B, sum_of_all_B, no_of_false_possitive_for_our_model)


if __name__ == "__main__":
    pred_dict, res_dict = read_data_from_log_file()
    (proc_of_A, sum_of_all_A), (proc_of_B, sum_of_all_B, false_positives) = count_successful_predictions(pred_dict, res_dict)

    print(f"""Naive A: skuteczność={proc_of_A:.2f}% dla {sum_of_all_A} rekordów""")
    print(f"""Our   B: skuteczność={proc_of_B:.2f}% dla {sum_of_all_B} rekordów
    z czego liczba błędów false-positive={false_positives}""")