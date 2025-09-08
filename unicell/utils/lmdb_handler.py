import os
import lmdb
import json


class LmdbHelper(object):
    def __init__(self, lmdb_path, vocab_dir, write_frequency=50000, map_size=1e9, is_write=True):
        self.lmdb_path = lmdb_path
        self.vocab_dir = vocab_dir
        self.write_frequency = write_frequency
        self.db = None
        self.txn = None
        self.is_write = is_write
        self.map_size = map_size

    def get_txn(self, write=True):
        return self.db.begin(write=write)

    def init_db(self, map_size, is_write):
        self.db = lmdb.open(self.lmdb_path, map_size=map_size)
        self.txn = self.get_txn(is_write)

    def get_length(self):
        if self.db is None:
            self.init_db(self.map_size, self.is_write)
        res = self.txn.get(b'__len__')
        length = int(res.decode("utf-8")) if res else 0
        return length

    def write_lmdb(self, record, db_length):
        index = db_length
        self.txn.put(str(index).encode(), json.dumps(record).encode())
        return index

    def read_lmdb(self, index):
        res = json.loads(self.txn.get(str(index).encode()))
        return res

    def matrix2lmdb(self, mat, labels):
        """
        mat : express matrix, numpy array
        labels: celltype  of the cell
        """
        cur_len = self.get_length() + 1 if os.path.exists(self.lmdb_path) else 0
        for i in range(mat.shape[0]):
            x = mat[i, :]
            label = labels[i]
            record = {'x': x, 'label': label}
            self.write_lmdb(record, cur_len)
            cur_len += 1
            if cur_len % self.write_frequency == 0:
                self.txn.commit()
                self.txn = self.get_txn(write=True)
                print('write to lmbd: {}'.format(cur_len))
        self.txn.put(b'__len__', str(cur_len).encode())
        print('writing lmbd end! the length of lmdb is {}.'.format(cur_len))
        return cur_len

