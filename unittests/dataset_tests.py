import unittest, json


class TestCollectionState(unittest.TestCase):
    def test_dataset(self):
        with open(
            "/home/tao/lab/silco_journal/data/list/cl_voc_v2.json", "r"
        ) as f:
            tmp = json.load(f)

            def get_img_path_set(split):
                _list = []
                for key, value in tmp[split].items():
                    _list.extend(value)
                img_path_list = [tmp["img_path"] for tmp in _list]
                return set(img_path_list)

            train_img_path_set = get_img_path_set("train")
            val_img_path_set = get_img_path_set("val")
            test_img_path_set = get_img_path_set("test")
        self.assertTrue(
            len(train_img_path_set.intersection(val_img_path_set)) == 0
        )
        self.assertTrue(
            len(train_img_path_set.intersection(test_img_path_set)) == 0
        )
        self.assertTrue(
            len(val_img_path_set.intersection(test_img_path_set)) == 0
        )

    def test_torch_similarity(self):
        import torch
        import numpy as np

        def sim_matrix(_a, _b, eps=1e-8):
            """
            a: BxT1xC
            b: BxT2xC
            added eps for numerical stability
            return BxT1xT2, means similarity score
            https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
            """
            bb, _, _ = _a.shape
            sim_mt_list = []
            for _bb in range(bb):
                a = _a[_bb]
                b = _b[_bb]
                a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
                a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
                b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
                sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
                sim_mt_list.append(sim_mt.unsqueeze(0))

            return torch.cat(sim_mt_list, 0)

        self.assertTrue(
            torch.sum(
                sim_matrix(
                    torch.from_numpy(
                        np.array([1] * 3, np.float).reshape(1, 3, 1)
                    ),
                    torch.from_numpy(
                        np.array([1] * 1, np.float).reshape(1, 1, 1)
                    ),
                )
            ).item()
            == 3
        )

        self.assertTrue(
            torch.sum(
                sim_matrix(
                    torch.from_numpy(
                        np.array([1] * 30, np.float).reshape(10, 3, 1)
                    ),
                    torch.from_numpy(
                        np.array([1] * 10, np.float).reshape(10, 1, 1)
                    ),
                )
            ).item()
            == 30
        )

        self.assertTrue(
            torch.sum(
                sim_matrix(
                    torch.from_numpy(
                        np.array([1] * 60, np.float).reshape(10, 3, 2)
                    ),
                    torch.from_numpy(
                        np.array([1] * 20, np.float).reshape(10, 1, 2)
                    ),
                )
            ).item()
            == 30
        )


if __name__ == "__main__":
    unittest.main()
