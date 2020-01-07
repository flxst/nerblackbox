
import csv


def read_txt(phase):
    file_path = f'datasets/ner/swedish_ner_corpus/{phase}_corpus.txt'

    _rows = []
    with open(file_path) as f:
        for row in f.readlines():
            _rows.append(row.strip().split())

    print(f'> read {file_path}')

    return _rows


def write_csv(phase, rows):
    file_path = f'datasets/ner/swedish_ner_corpus/{phase}.csv'

    with open(file_path, mode='w') as f:

        labels = []
        sentence = []
        for row in rows:
            if len(row) == 2:
                sentence.append(row[0])
                labels.append(row[1] if row[1] != '0' else 'O')  # replace zeros by capital O (!)
                if row[0] == '.':
                    f.write(' '.join(labels) + '\t' + ' '.join(sentence) + '\n')
                    labels = []
                    sentence = []

    print(f'> wrote {file_path}')


def main():
    # train
    rows = read_txt('train')
    # print(rows)
    write_csv('train', rows)

    # valid/test
    rows = read_txt('test')
    split_index = int(len(rows)/2.)
    rows_valid = rows[:split_index]
    rows_test = rows[split_index:]
    write_csv('valid', rows_valid)
    write_csv('test', rows_test)


if __name__ == '__main__':
    main()
