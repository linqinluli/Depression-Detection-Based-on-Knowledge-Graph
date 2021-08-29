#%%
if __name__ == '__main__':

        from numpy.core.fromnumeric import shape
        from model import DEPredictor
        from util import predict_one, read_data, feature_extract
        #%%
        x = read_data('train', 777)
        print('ok')
        #%%
        _, label, score_res, max_post_list = feature_extract(x[485:487])
        #%%
        from torch.utils.data import Dataset, DataLoader
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from model import DEPredictor
        from RSDDDataset import RSDDDataset
        property_num = score_res.shape[1]

        predictor = DEPredictor(property_num).cuda()
        predictor.load_state_dict(torch.load('lr_0.0001.pt'))
        val_x = score_res
        val_y = label
        # val_y[1] = 0
        val_data = RSDDDataset(val_x, val_y, max_post_list)
        predict_dataloader = DataLoader(val_data, batch_size=1)
        #%%
        from util import predict_one
        for batch, (feature, label,post) in enumerate(predict_dataloader):
                x, y = feature.cuda(), label.cuda()
                valpred = predictor(x).cpu().detach().numpy()
                if(y.cpu().detach().numpy()[0]==1):
                        predict_one(x[0],valpred[0][0],post)
        