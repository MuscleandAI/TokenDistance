import jieba
import numpy as np

class TokenDistance():
    def __init__(self, idf_path):
        idf_dict = {}
        tmp_idx_list = []
        with open(idf_path, encoding="utf8") as f:
            for line in f:
                ll = line.strip().split(" ")
                idf_dict[ll[0]] = float(ll[1])
                tmp_idx_list.append(float(ll[1]))
        self._idf_dict = idf_dict
        self._median_idf = np.median(tmp_idx_list)
    
    def predict_jaccard(self, q1, q2):
        # jaccard距离，根据idf加权
        if len(q1) < 1 or len(q2) < 1:
            return 0

        q1 = set(list(jieba.cut(q1)))
        q2 = set(list(jieba.cut(q2)))
        print(q1.intersection(q2))
        print(q1.union(q2))

        numerator = sum([self._idf_dict.get(word, self._median_idf) for word in q1.intersection(q2)])
        denominator  = sum([self._idf_dict.get(word, self._median_idf) for word in q1.union(q2)])
        return numerator / denominator

    def predict_left(self, q1, q2):
        # 单向相似度，分母为q1，根据idf加权
        if len(q1) < 1 or len(q2) < 1:
            return 0
        
        q1 = set(list(jieba.cut(q1)))
        q2 = set(list(jieba.cut(q2)))

        numerator = sum([self._idf_dict.get(word, self._median_idf) for word in q1.intersection(q2)])
        denominator  = sum([self._idf_dict.get(word, self._median_idf) for word in q1])
        return numerator / denominator

    def predict_cqrctr(self, q1, q2):
        # cqr*ctr
        if len(q1) < 1 or len(q2) < 1:
            return 0

        cqr = self.predict_left(q1, q2)
        ctr = self.predict_left(q2, q1)

        return cqr * ctr
    
if __name__ == "__main__":
    import sys
    q1 = '华为有限公司'
    q2 ="华为云计算有限公司"
    

    token_distance = TokenDistance("../dict/idf.txt")
    print(q1, q2)
    print(token_distance.predict_jaccard(q1, q2))
    print(token_distance.predict_left(q1, q2))
    print(token_distance.predict_cqrctr(q1, q2))