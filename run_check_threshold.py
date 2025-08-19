import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats


PATH_DICT = {
    'proposed': {
        'path_root': './models_saved/20200507-100156_downstream_non-linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_c1cfa4361f3049f8b1a68b1f56819f76_Epoch_59_max-valid-LVH-AUROC_1_final_pred_result.csv',
            'model_68e66fd605814b2490cdafb9b91f45d7_Epoch_100_max-valid-LVH-AUROC_3_final_pred_result.csv',
            'model_13d33374af944e29a1d47e24b8ceb468_Epoch_96_max-valid-LVH-AUROC_2_final_pred_result.csv'
        ]
    },
    'proposed_triplet': {
        'path_root': './models_saved/20200511-111908_downstream_non-linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_c4142e3754a14e57adb3aa84dbc22200_Epoch_86_max-valid-LVH-AUROC_1_final_pred_result.csv',
            'model_7e96d0b657cb4224ae9fe91cb86c30d6_Epoch_98_max-valid-LVH-AUROC_2_final_pred_result.csv',
            'model_20892832911444a28bffb05fd7c7e81b_Epoch_85_max-valid-LVH-AUROC_3_final_pred_result.csv'
        ]
    },
    'proposed_ft_all': {
        'path_root': './models_saved/20200521-014648_downstream_non-linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_dd7c1ed5a9c041a084a2cb1427ad31e6_Epoch_31_max-valid-LVH-AUROC_2_final_pred_result.csv',
            'model_8e745648c32844408b26df1db57a327d_Epoch_17_max-valid-LVH-AUROC_1_final_pred_result.csv',
            'model_4eb7b9202f364d4780c6b78f562683dc_Epoch_28_max-valid-LVH-AUROC_3_final_pred_result.csv'
        ]
    },
    'dae': {
        'path_root': './models_saved/20200521-192253_downstream_non-linear_lvh_dae',
        'path': [
            'model_e4e24b19685e44aea536ec1e7683bd41_Epoch_83_max-valid-LVH-AUROC_3_final_pred_result.csv',
            'model_88b636a18a464894805368827e291dce_Epoch_97_max-valid-LVH-AUROC_2_final_pred_result.csv',
            'model_02294e8963614e1ca4a7302aefc033ae_Epoch_78_max-valid-LVH-AUROC_1_final_pred_result.csv'
        ]
    },
    'cdae': {
        'path_root': './models_saved/20200521-143058_downstream_non-linear_lvh_cdae',
        'path': [
            'model_5bfc72c94b0b47fd956511e8e9b4ff0e_Epoch_98_max-valid-LVH-AUROC_1_final_pred_result.csv',
            'model_37a2cb2930984def8fca03ee09e41b5d_Epoch_96_max-valid-LVH-AUROC_3_final_pred_result.csv',
            'model_067cbc10f1b74dd59a649d1a46d64c24_Epoch_99_max-valid-LVH-AUROC_2_final_pred_result.csv'
        ]
    },
    'emotion': {
        'path_root': './models_saved/20200426-171945_downstream_non-linear_lvh_emotion_ssl_original',
        'path': [
            'model_e6547c6ab0974e51bf1c5cb1702e78c2_Epoch_75_max-valid-LVH-AUROC_1_final_pred_result.csv',
            'model_1f3d15ed15ae442b8543d4b3d6430006_Epoch_55_max-valid-LVH-AUROC_2_final_pred_result.csv',
            'model_1119c58a57ff4982966b6b7394e9ddb6_Epoch_77_max-valid-LVH-AUROC_3_final_pred_result.csv'
        ]
    },
    'unsup_mts': {
        'path_root': './models_saved/20200419-201832_downstream_non-linear_lvh_unsup_mts_causalcnn',
        'path': [
            'model_c96575e5de2743b8b14be6b79577d268_Epoch_72_max-valid-LVH-AUROC_2_final_pred_result.csv',
            'model_eb340b99a4444952bc84914354eba5df_Epoch_56_max-valid-LVH-AUROC_1_final_pred_result.csv',
            'model_ee69a7b9fcf24c95a07b0e3c7e839e43_Epoch_61_max-valid-LVH-AUROC_3_final_pred_result.csv'
        ]
    },
    'afl_proposed': {
        'path_root': './models_saved/20200718-122933_downstream_non-linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_10533cffb20b4432b572e6d666152840_Epoch_92_max-valid-AFL-AUROC_1_final_pred_result.csv',
            'model_5ff6f5253f504768a3f3123904225ac2_Epoch_99_max-valid-AFL-AUROC_3_final_pred_result.csv',
            'model_6a377071ae9b489193dace75711cef8b_Epoch_97_max-valid-AFL-AUROC_2_final_pred_result.csv'
        ]
    },
    'afl_unsup_mts': {
        'path_root': './models_saved/20200716-141902_downstream_non-linear_lvh_unsup_mts_causalcnn',
        'path': [
            'model_e0942158ddc447abb0c6a3cf545bf91c_Epoch_94_max-valid-AFL-AUROC_1_final_pred_result.csv',
            'model_830073da25f64659b9255092629c6a43_Epoch_100_max-valid-AFL-AUROC_2_final_pred_result.csv',
            'model_09975fafe1814d34ac83bd29b60f918d_Epoch_99_max-valid-AFL-AUROC_3_final_pred_result.csv'
        ]
    },
    '1avb_proposed': {
        'path_root': './models_saved/20200522-010539_downstream_non-linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_77705965cb10469096f3dfdfc046f82e_Epoch_99_max-valid-1AVB-AUROC_3_final_pred_result.csv',
            'model_965a653259ab4d35bbbf274fc5cecc45_Epoch_88_max-valid-1AVB-AUROC_2_final_pred_result.csv',
            'model_d11f5ff71c874913959c4a7128169977_Epoch_98_max-valid-1AVB-AUROC_1_final_pred_result.csv'
        ]
    },
    '1avb_unsup_mts': {
        'path_root': './models_saved/20200522-213435_downstream_non-linear_lvh_unsup_mts_causalcnn',
        'path': [
            'model_7c8a354407ec4c1593d078f6304a961a_Epoch_70_max-valid-1AVB-AUROC_3_final_pred_result.csv',
            'model_7ba43083e35f4bdf8b2e1e7cae67216c_Epoch_62_max-valid-1AVB-AUROC_1_final_pred_result.csv',
            'model_37055309ee8c4864a22bd0103762fd83_Epoch_87_max-valid-1AVB-AUROC_2_final_pred_result.csv'
        ]
    },
    'lvh_cdae': {
        'path_root': './models_saved/20200720-172030_downstream_linear_lvh_cdae',
        'path': [
            'model_final_pred_result.csv',
        ]
    },
    'lvh_dae': {
        'path_root': './models_saved/20200720-172053_downstream_linear_lvh_dae',
        'path': [
            'model_final_pred_result.csv',
        ]
    },
    'lvh_proposed_MSloss': {
        'path_root': './models_saved/20200720-170308_downstream_linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_final_pred_result.csv',
        ]
    },
    'lvh_proposed_Tripletloss': {
        'path_root': './models_saved/20200721-001609_downstream_linear_lvh_MSwKM_causalcnn',
        'path': [
            'model_final_pred_result.csv',
        ]
    },
    'lvh_unsup_mts': {
        'path_root': './models_saved/20200721-001307_downstream_linear_lvh_unsup_mts_causalcnn',
        'path': [
            'model_final_pred_result.csv',
        ]
    },
    'lvh_emotion': {
        'path_root': './models_saved/20200721-103422_downstream_linear_lvh_emotion_ssl_original',
        'path': [
            'model_final_pred_result.csv',
        ]
    }
}


if __name__ == "__main__":
    for key, _ in PATH_DICT.items():
        path_root = PATH_DICT[key]['path_root']
        pred_prob_list, pred_list, valuable_auroc_list = [], [], []
        tpr_thres = 0.15
        for path in PATH_DICT[key]['path']:
            df = pd.read_csv(f'{path_root}/{path}')
            y = np.array(df['truth'])
            pred_prob_list.append(df['pred_prob'])
            pred_list.append(df['pred'])
            valuable_auroc_list.append(metrics.roc_auc_score(y, np.array(df['pred_prob']), max_fpr=0.05))
        avg_pred_prob = np.average(np.array(pred_prob_list), axis=0)
        avg_pred = stats.mode(np.array(pred_list), axis=0)[0].reshape(-1)
        fpr_all, tpr_all, thresholds = metrics.roc_curve(y, avg_pred_prob, pos_label=1)
        fpr_candidate, tpr_candidate, thes_candidate = [], [], []
        for fpr, tpr, thes in zip(fpr_all, tpr_all, thresholds):
            if (fpr - tpr_thres) < 0:
                fpr_candidate.append(fpr)
                tpr_candidate.append(tpr)
                thes_candidate.append(thes)
        max_index = np.argmax(fpr_candidate)
        average_precision = metrics.average_precision_score(y, avg_pred)
        average_cohen_kappa_score = metrics.cohen_kappa_score(y, avg_pred)
        average_MCC_score = metrics.matthews_corrcoef(y, avg_pred)
        average_valuable_auroc_list = np.average(np.array(valuable_auroc_list), axis=0)
        print(f'{key} / Average AUROC score fpr<0.05: {average_valuable_auroc_list}')
        print(f'{key} / Average AUROC score fpr<0.05 by average prob: {metrics.roc_auc_score(y, avg_pred_prob, max_fpr=0.05)}')
        # print(f'Average MCC score: ', average_MCC_score)
        # print(f'Average precision-recall score: ', average_precision)
        # print(f'Average kappa score: ', average_cohen_kappa_score)
        print(f'{key} / :', 1 - fpr_candidate[max_index], tpr_candidate[max_index], thes_candidate[max_index])
        precision, recall, _ = metrics.precision_recall_curve(y, avg_pred_prob)
        # plt.figure()
        # lw = 0.5
        # plt.plot(fpr_all, tpr_all, lw=lw, label='proposed')
        # plt.plot(fpr_all_mts, tpr_all_mts, lw=lw, label='baseline')
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.savefig('auc_roc.png')

    # path_root = PATH_DICT['unsup_mts']['path_root']
    # pred_prob_list, pred_list = [], []
    # for path in PATH_DICT['unsup_mts']['path']:
    #     df = pd.read_csv(f'{path_root}/{path}')
    #     pred_prob_list.append(df['pred_prob'])
    #     pred_list.append(df['pred'])
    # y = np.array(df['truth'])
    # avg_pred_prob = np.average(np.array(pred_prob_list), axis=0)
    # avg_pred = stats.mode(np.array(pred_list), axis=0)[0].reshape(-1)
    # fpr_all_mts, tpr_all_mts, thresholds = metrics.roc_curve(y, avg_pred_prob, pos_label=1)
    # fpr_candidate, tpr_candidate, thes_candidate = [], [], []
    # for fpr, tpr, thes in zip(fpr_all_mts, tpr_all_mts, thresholds):
    #     if (fpr - tpr_thres) < 0:
    #         fpr_candidate.append(fpr)
    #         tpr_candidate.append(tpr)
    #         thes_candidate.append(thes)
    # max_index = np.argmax(fpr_candidate)
    # average_precision_mts = metrics.average_precision_score(y, avg_pred)
    # average_cohen_kappa_score = metrics.cohen_kappa_score(y, avg_pred)
    # average_MCC_score = metrics.matthews_corrcoef(y, avg_pred)
    # print('Average MCC score: ', average_MCC_score)
    # print('Average precision-recall score: ', average_precision_mts)
    # print('Average kappa score: ', average_cohen_kappa_score)
    # print(1 - fpr_candidate[max_index], tpr_candidate[max_index], thes_candidate[max_index])
    # precision_mts, recall_mts, _ = metrics.precision_recall_curve(y, avg_pred_prob)
    # plt.step(recall, precision, where='post', label='proposed')
    # plt.step(recall_mts, precision_mts, where='post', label='baseline')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Average precision score')
    # plt.legend(loc="lower right")
    # plt.savefig('PR_curve.png')

    # plt.figure()
    # lw = 0.5
    # plt.plot(fpr_all, tpr_all, lw=lw, label='proposed')
    # plt.plot(fpr_all_mts, tpr_all_mts, lw=lw, label='baseline')
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig('auc_roc.png')
