def get_bad_word_dict():
    lines = open('badwords.txt').readlines()
    lines = [l.lower().strip() for l in lines]
    lines = [l.split(',') for l in lines]
    bad_dict = {}
    for v in lines:
        if len(v) == 1:
            bad_dict[v[0]] =v[0]
        else:
            bad_dict[v[0]] =v[1]

    return bad_dict

if __name__ == '__main__':
    print(get_bad_word_dict())
