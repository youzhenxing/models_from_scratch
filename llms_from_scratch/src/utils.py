
def print_line(text, is_start = True):
    if is_start:
       print('-----> start: {} -------------'.format(text))
    else:
       print('-----> finish: {} ----------------'.format(text))
    return