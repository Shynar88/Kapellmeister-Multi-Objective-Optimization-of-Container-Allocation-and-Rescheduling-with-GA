from ast import literal_eval

def parse_log_data():
    for line in open("fitness.log", "r"):
        info = line.split(" - ")
        dates.append(info[0])
        expr = literal_eval(info[1])
        exec_times.append(expr[0])
        inputs.append(expr[1])
    indexes = tuple(np.arange(1, len(dates) + 1))
    return indexes, dates, exec_times, inputs