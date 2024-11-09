

import argparse
import time
import csv
import pickle
import operator
import datetime
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica')
    opt = parser.parse_args()
    print(opt)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset = 'train-item-views.csv'


    dataset_path = os.path.join(current_dir, dataset)
    print(f"Attempting to open file: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} does not exist.")
        print("Available files in the current directory:")
        print("\n".join(os.listdir(current_dir)))
        exit(1)

    with open(dataset_path, "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in reader:
            sessid = data['sessionId']
            if curdate and not curid == sessid:
                date = ''
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid
            item = data['itemId'],int(data['timeframe'])
            curdate = ''
            curdate = data['eventdate']

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = ''
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date
    print("-- Reading data @ %ss" % datetime.datetime.now())

    # Filter out length 1 sessions
    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            del sess_clicks[s]
            del sess_date[s]

    # Count number of times each item appears
    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1

    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

    length = len(sess_clicks)
    for s in list(sess_clicks):
        curseq = sess_clicks[s]
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        if len(filseq) < 2:
            del sess_clicks[s]
            del sess_date[s]
        else:
            sess_clicks[s] = filseq

    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]

    for _, date in dates:
        if maxdate < date:
            maxdate = date

    # 7 days for test
    splitdate = 0
    splitdate = maxdate - 86400 * 7

    print('Splitting date', splitdate)      # diginetica: ('Split date', 1464109200.0)
    train_sess = filter(lambda x: x[1] < splitdate, dates)
    test_sess = filter(lambda x: x[1] > splitdate, dates)

    # Sort sessions by date
    train_sess = sorted(train_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    test_sess = sorted(test_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
    print(len(train_sess))    # 186670   
    print(len(test_sess))    # 15979   
    print(train_sess[:3])
    print(test_sess[:3])
    print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

    # Choosing item count >=5 gives approximately the same number of items as reported in paper
    item_dict = {}
    # Convert training sessions to sequences and renumber items to start from 1
    def obtian_tra():
        train_ids = []
        train_seqs = []
        train_dates = []
        item_ctr = 1
        for s, date in train_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr
                    item_ctr += 1
            if len(outseq) < 2:  # Doesn't occur
                continue
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]
        print(item_ctr)     # 43098
        return train_ids, train_dates, train_seqs


    # Convert test sessions to sequences, ignoring items that do not appear in training set
    def obtian_tes():
        test_ids = []
        test_seqs = []
        test_dates = []
        for s, date in test_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
            if len(outseq) < 2:
                continue
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs


    train_ids, train_dates, train_seqs = obtian_tra()
    test_ids, test_dates, test_seqs = obtian_tes()


    def process_seqs(iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        ids = []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
        return out_seqs, out_dates, labs, ids


    train_seqs, train_dates, train_labs, train_ids = process_seqs(train_seqs, train_dates)
    test_seqs, test_dates, test_labs, test_ids = process_seqs(test_seqs, test_dates)
    train = (train_seqs, train_labs)
    test = (test_seqs, test_labs)
    print(len(train_seqs))
    print(len(test_seqs))
    print(train_seqs[:3], train_dates[:3], train_labs[:3])
    print(test_seqs[:3], test_dates[:3], test_labs[:3])
    all = 0

    for seq in train_seqs:
        all += len(seq)
    for seq in test_seqs:
        all += len(seq)
    print('avg length: ', all/(len(train_seqs) + len(test_seqs) * 1.0))
    if opt.dataset == 'diginetica':
        if not os.path.exists('diginetica'):
            os.makedirs('diginetica')
        pickle.dump(train, open('diginetica/train.txt', 'wb'))
        pickle.dump(test, open('diginetica/test.txt', 'wb'))
        pickle.dump(train_seqs, open('diginetica/all_train_seq.txt', 'wb'))

    print('Done.')
