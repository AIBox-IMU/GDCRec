import pdb
import logging
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy


class Tester(object):
    def __init__(self, args, model, dataloader):
        self.args = args
        self.model = model
        self.history_dic = dataloader.historical_dict
        self.history_csr = dataloader.train_csr
        self.dataloader = dataloader.dataloader_test
        self.test_dic = dataloader.test_dic
        self.cate = np.array(list(dataloader.category_dic.values()))
        self.metrics = args.metrics


    def judge(self, users, items):

        results = {metric: 0.0 for metric in self.metrics}
        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            print(len(items))
            for i in range(len(items)):
                results[metric] += f(items[i], test_pos = self.test_dic[users[i]], num_test_pos = len(self.test_dic[users[i]]), count = stat[i], model = self.model)
        return results


    def ground_truth_filter(self, users, items):
        batch_size, k = items.shape
        res = []
        for i in range(len(users)):
            gt_number = len(self.test_dic[users[i]])
            if gt_number >= k:
                res.append(items[i])
            else:
                res.append(items[i][:gt_number])
        return res


    def test(self):
        results = {}
        h = self.model.get_embedding()
        count = 0

        for k in self.args.k_list:
            results[k] = {metric: 0.0 for metric in self.metrics}

        for batch in tqdm(self.dataloader):

            users = batch[0]
            count += users.shape[0]
            # count += len(users)
            scores = self.model.get_score(h, users)
            users = users.tolist()
            mask = torch.tensor(self.history_csr[users].todense(), device = scores.device).bool()

            _, recommended_items = torch.topk(scores, k = max(self.args.k_list))
            recommended_items = recommended_items.cpu()
            for k in self.args.k_list:

                results_batch = self.judge(users, recommended_items[:, :k])

                for metric in self.metrics:
                    results[k][metric] += results_batch[metric]

        for k in self.args.k_list:
            for metric in self.metrics:
                results[k][metric] = results[k][metric] / count
        self.show_results(results)

    def show_results(self, results):
        for metric in self.metrics:
            for k in self.args.k_list:
                logging.info('For top{}, metric {} = {}'.format(k, metric, results[k][metric]))


    def stat(self, items):
        stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]
        return stat

class Metrics(object):

    def __init__(self):
        pass

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'precision': Metrics.precision,
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage
        }

        return metrics_map[metric]

    @staticmethod
    def precision(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count / len(items)

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0
    @staticmethod
    def coverage(items, **kwargs):
        item_categories = {}
        with open("D:\Code\PythonProject\Article\DGRec\datasets\Beauty\item_category.txt", 'r',
                  encoding='utf-8-sig') as file:
            for line in file:
                item_id, category = line.strip().split(',')
                item_categories[int(item_id)] = int(category)
        item_category_list1 = []
        items_list = items.tolist()
        for item in items_list:
            category1 = item_categories.get(item)
            if category1 and category1 not in item_category_list1:
                item_category_list1.append(category1)

        test_pos = kwargs['test_pos']
        item_category_list2 = []
        for item in test_pos:
            category2 = item_categories.get(item)
            if category2 and category2 not in item_category_list2:
                item_category_list2.append(category2)

        set1 = set(item_category_list1)
        set2 = set(item_category_list2)

        common_elements = set2.intersection(set1)
        num_common_elements = len(common_elements)
        coverage = num_common_elements / len(set2) if len(set2) > 0 else 0

        return coverage


    @staticmethod
    def ndcg(items, **kwargs):
        test_pos = kwargs['test_pos']
        rels = np.isin(items, test_pos).astype(int)

        def dcg(rels):
            return np.sum(rels / np.log2(np.arange(2, len(rels) + 2)))

        dcg_value = dcg(rels)
        dcg_value=dcg_value
        ideal_rels = sorted(rels, reverse=True)
        idcg_value = dcg(ideal_rels)

        return dcg_value / idcg_value if idcg_value > 0 else 0.0
